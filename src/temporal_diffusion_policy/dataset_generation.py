from temporal_diffusion_policy.temporal_encodings import binary_idle_indicator, velocity_vmax as get_velocity_vmax
import torch
import numpy as np
import zarr
from temporal_diffusion_policy.temporal_encodings import (
    progress_positional_encoding_transformer, 
    progress_saturalising_encoding, 
    IdlenessEncoder, 
    calculate_growth_rate_parameter
)

def iter_episodes(episode_ends):
    start = 0
    for end in episode_ends:
        yield start, end
        start = end

def create_sample_indices(
        episode_ends:np.ndarray, sequence_length:int,
        pad_before: int=0, pad_after: int=0):
    indices = list()
    for i in range(len(episode_ends)):
        start_idx = 0
        if i > 0:
            start_idx = episode_ends[i-1]
        end_idx = episode_ends[i]
        episode_length = end_idx - start_idx

        min_start = -pad_before
        max_start = episode_length - sequence_length + pad_after

        # range stops one idx before end
        for idx in range(min_start, max_start+1):
            buffer_start_idx = max(idx, 0) + start_idx
            buffer_end_idx = min(idx+sequence_length, episode_length) + start_idx
            start_offset = buffer_start_idx - (idx+start_idx)
            end_offset = (idx+sequence_length+start_idx) - buffer_end_idx
            sample_start_idx = 0 + start_offset
            sample_end_idx = sequence_length - end_offset
            indices.append([
                buffer_start_idx, buffer_end_idx,
                sample_start_idx, sample_end_idx])
    indices = np.array(indices)
    return indices


def sample_sequence(train_data, sequence_length,
                    buffer_start_idx, buffer_end_idx,
                    sample_start_idx, sample_end_idx):
    result = dict()
    for key, input_arr in train_data.items():
        sample = input_arr[buffer_start_idx:buffer_end_idx]
        data = sample
        if (sample_start_idx > 0) or (sample_end_idx < sequence_length):
            data = np.zeros(
                shape=(sequence_length,) + input_arr.shape[1:],
                dtype=input_arr.dtype)
            if sample_start_idx > 0:
                data[:sample_start_idx] = sample[0]
            if sample_end_idx < sequence_length:
                data[sample_end_idx:] = sample[-1]
            data[sample_start_idx:sample_end_idx] = sample
        result[key] = data
    return result

# normalize data
def get_data_stats(data):
    data = data.reshape(-1,data.shape[-1]) # (..., D) -> (N, D)
    stats = {
        'min': np.min(data, axis=0),
        'max': np.max(data, axis=0),
        'q01': np.quantile(data, 0.01, axis=0),
        'q99': np.quantile(data, 0.99, axis=0)
    }
    return stats

class NormalizationType:
    NONE = "NONE"
    MIN_MAX = "MIN_MAX"
    Q01_Q99 = "Q01_Q99"
    MIN_Q99 = "MIN_Q99" # minimum for lower bound, q99 for upper bound.


def apply_stats_overrides(stats: dict, override: dict):
    if 'min' not in override or 'max' not in override:
        raise ValueError("stats override must include both 'min' and 'max'")

    override_min = np.asarray(override['min'], dtype=np.float32)
    override_max = np.asarray(override['max'], dtype=np.float32)
    if override_min.shape != stats['min'].shape or override_max.shape != stats['max'].shape:
        raise ValueError(
            f"stats override shape mismatch: expected {stats['min'].shape}, got min={override_min.shape}, max={override_max.shape}"
        )

    if 'mask' in override and override['mask'] is not None:
        mask = np.asarray(override['mask'], dtype=bool)
        if mask.shape != stats['min'].shape:
            raise ValueError(
                f"stats override mask shape mismatch: expected {stats['min'].shape}, got {mask.shape}"
            )
    else:
        mask = np.isfinite(override_min) & np.isfinite(override_max)

    stats['min'] = np.where(mask, override_min, stats['min'])
    stats['max'] = np.where(mask, override_max, stats['max'])


def normalize_data(data, stats, normalisation_type: NormalizationType = NormalizationType.MIN_MAX):
    # nomalize to [0,1]
    if normalisation_type == NormalizationType.MIN_MAX:
        minimum = stats['min']
        maximum = stats['max']
    elif normalisation_type == NormalizationType.Q01_Q99:
        minimum = stats['q01']
        maximum = stats['q99']
    elif normalisation_type == NormalizationType.MIN_Q99:
        minimum = stats['min']
        maximum = stats['q99']
    else:
        raise ValueError(f"Unsupported normalisation type: {normalisation_type}")
    
    if np.allclose(minimum, maximum):
        return np.zeros_like(data)

    ndata = (data - minimum) / (maximum - minimum + 1e-8)
    # normalize to [-1, 1]
    ndata = ndata * 2 - 1

    if normalisation_type == NormalizationType.Q01_Q99 or normalisation_type == NormalizationType.MIN_Q99:
        # clip to [-1, 1]
        ndata = np.clip(ndata, -1.0, 1.0)

    return ndata

def unnormalize_data(ndata, stats, normalisation_type: NormalizationType = NormalizationType.MIN_MAX):
    if normalisation_type == NormalizationType.MIN_MAX:
        minimum = stats['min']
        maximum = stats['max']
    elif normalisation_type == NormalizationType.Q01_Q99:
        minimum = stats['q01']
        maximum = stats['q99']
    elif normalisation_type == NormalizationType.MIN_Q99:
        minimum = stats['min']
        maximum = stats['q99']
    else:
        raise ValueError(f"Unsupported normalisation type: {normalisation_type}")

    ndata = (ndata + 1) / 2
    data = ndata * (maximum - minimum + 1e-8) + minimum
    return data


# dataset
class TemporalConditioningDataset(torch.utils.data.Dataset):
    def __init__(self,
        dataset_path: str,
        pred_horizon: int,
        obs_horizon: int,
        action_horizon: int,
        dataset_keys_and_normalisation: dict[str, str] = {
            'action': NormalizationType.MIN_MAX,
            'agent_pose': NormalizationType.MIN_MAX,
            'agent_velocity': NormalizationType.Q01_Q99},
        normalization_mask: dict[str, np.ndarray] = None, # dict of key to boolean mask of shape (D,) for which dimensions to apply normalization to. If None, apply to all dimensions.
        normalization_stats_overrides: dict[str, dict] = None, # dict of key -> {'min': (D,), 'max': (D,), optional 'mask': (D,)}
        keys_to_exclude_from_stats_calculation: list[str] = ['full_image', 'hand_image', 'full_image_raw', 'done'], # for efficiency
        use_idleness_encoding: bool = False,
        use_direct_idleness_conditioning: bool = False,
        direct_idleness_threshold: float = 0.03,
        use_sinusoidal_progress_encoding: bool = False,
        sinusoidal_progress_encoding_parameters: dict = {'d_model': 4, 'power_value': 10000.0, 'normalize_by_episode_length': False},
        use_saturating_progress_encoding: bool = False,
        saturating_progress_encoding_omega: float = None, # omega in the paper. Set to None if auto-calc based on max steps.
        idleness_encoding_alpha: float = None, # omega in the paper. Set to None to auto-calc based on max velocity.
        idleness_encoding_rest_thresh: float = 0.03, # velocity threshold for being considered "at rest". In normalized velocity units (i.e. 0.03 means 3% of max velocity)
        return_images: bool = True,
    ):

        self.use_idleness_encoding = use_idleness_encoding
        self.use_direct_idleness_conditioning = use_direct_idleness_conditioning
        self.return_images = return_images
        self.dataset_keys_and_normalisation = dataset_keys_and_normalisation
        self.normalization_mask = normalization_mask
        self.normalization_stats_overrides = normalization_stats_overrides

        # read from zarr dataset
        dataset_root = zarr.open(dataset_path, 'r')


        # (N, D)
        train_data = dict()
        stats = dict()

        keys_to_passthrough = list(dataset_root['data'].keys())

        print("Pre-processing data - this can take some time...")
        for key in keys_to_passthrough:
            
            if key in self.dataset_keys_and_normalisation:
                normalise_type = self.dataset_keys_and_normalisation[key]
            else:
                normalise_type = NormalizationType.NONE
            
             # ensure at least 2D
            data = dataset_root['data'][key][:]
            if data.ndim == 1:
                data = data[:, None]
            
            if key not in keys_to_exclude_from_stats_calculation:
                stats[key] = get_data_stats(data)
                if self.normalization_stats_overrides is not None and key in self.normalization_stats_overrides:
                    apply_stats_overrides(stats[key], self.normalization_stats_overrides[key])
            
            if normalise_type == NormalizationType.NONE:
                train_data[key] = data
            else:
                print(f"Normalising key: '{key}'")
                normalized = normalize_data(data, stats[key], normalisation_type=normalise_type)
                if self.normalization_mask is not None and key in self.normalization_mask:
                    mask = self.normalization_mask[key]
                    if mask.shape[0] != normalized.shape[1]:
                        raise ValueError(f"Normalization mask for key '{key}' has shape {mask.shape}, but data has shape {normalized.shape}")
                    normalized = np.where(mask, normalized, data)
                train_data[key] = normalized
        print("Pre-processing done")

        episode_ends = dataset_root['meta']['episode_ends'][:]


        if use_sinusoidal_progress_encoding:
            step_count = dataset_root['data']['step_count'][:]
            power_value = sinusoidal_progress_encoding_parameters['power_value']
            # https://www.tensorflow.org/tutorials/text/transformer#positional_encoding
            d_model = sinusoidal_progress_encoding_parameters['d_model']  # dimension of the positional encoding (half of this is sin, half is cos)
            step_progress_pe = np.zeros((len(step_count), d_model), dtype=np.float32)  # (N, d_model)

            for start_idx, end_idx in iter_episodes(episode_ends):
                episode_length = end_idx - start_idx
                t = np.arange(episode_length, dtype=np.float32)  # (T,)
                if sinusoidal_progress_encoding_parameters['normalize_by_episode_length']:
                    step_progress_pe[start_idx:end_idx, :] = progress_positional_encoding_transformer(t=t, d_model=d_model, power_value=power_value, T=episode_length)
                else:
                    step_progress_pe[start_idx:end_idx, :] = progress_positional_encoding_transformer(t=t, d_model=d_model, power_value=power_value)

            train_data['sinusoidal_progress_encoding'] = step_progress_pe # (N, 2*F)
            
        if use_saturating_progress_encoding:
            step_count = dataset_root['data']['step_count'][:]
            at_progress_ctr = np.zeros_like(step_count, dtype=np.float32)
            
            if saturating_progress_encoding_omega is not None:
                omega = saturating_progress_encoding_omega
            else:
                omega = calculate_growth_rate_parameter(max_steps=stats['step_count']['max'], target_percent=0.99)
            self.in_use_saturating_progress_encoding_omega = omega

            # then we want to loop over episodes, to prevent counting across episode boundaries
            for start_idx, end_idx in iter_episodes(episode_ends):
                episode_step_count = step_count[start_idx:end_idx]
                # max_steps = max(episode_step_count)
                progress = progress_saturalising_encoding(episode_step_count, omega=omega)
                at_progress_ctr[start_idx:end_idx] = progress
                # print(*np.vstack([episode_step_count.squeeze(), at_progress_ctr.squeeze()]).T, sep="\n")
            # add to train data
            train_data['saturating_progress_encoding'] = at_progress_ctr[:, None]

        if use_idleness_encoding:
            # idle = idleness > 0.95

            # first we want max over the whole dataset
            raw_velocity = dataset_root['data']['agent_velocity'][:]
            if raw_velocity.ndim == 1:
                raw_velocity = raw_velocity[:, None]
            # max_velocity = stats['agent_velocity']['max']
            at_rest_ctr = np.zeros_like(raw_velocity, dtype=np.float32)

            normalised_velocity = normalize_data(raw_velocity, stats['agent_velocity'], normalisation_type=NormalizationType.MIN_MAX)

            if idleness_encoding_alpha is not None:
                alpha = idleness_encoding_alpha
            else:
                alpha = calculate_growth_rate_parameter(max_steps=stats['step_count']['max']) # calculate alpha based on max steps, where we want 99% by the end
            
            idleness_encoder = IdlenessEncoder(vmax=stats['agent_velocity']['max'], rest_thresh=idleness_encoding_rest_thresh, alpha=alpha)
            self.in_use_idleness_encoding_alpha = alpha

            # then we want to loop over episodes, to prevent counting across episode boundaries
            for start_idx, end_idx in iter_episodes(episode_ends):

                resting = idleness_encoder.run_episode(raw_velocity[start_idx:end_idx, 0])
                at_rest_ctr[start_idx:end_idx, 0] = resting

                # assert np.isclose(resting.squeeze(), resting2).all()

                # print(*np.vstack([raw_velocity.squeeze(), at_rest_ctr.squeeze()]).T, sep="\n")
            # add to train data
            train_data['idleness'] = at_rest_ctr

        if use_direct_idleness_conditioning:
            raw_velocity = dataset_root['data']['agent_velocity'][:]
            if raw_velocity.ndim == 1:
                raw_velocity = raw_velocity[:, None]
            self.direct_idleness_vmax = get_velocity_vmax(raw_velocity)
            train_data['direct_idleness'] = binary_idle_indicator(
                raw_velocity,
                self.direct_idleness_vmax,
                direct_idleness_threshold,
            )


        # compute start and end of each state-action sequence
        # also handles padding
        indices = create_sample_indices(
            episode_ends=episode_ends,
            sequence_length=pred_horizon,
            pad_before=obs_horizon-1,
            pad_after=action_horizon-1)

        train_data_keys_without_action = list(train_data.keys())
        train_data_keys_without_action.remove('action')

        self.indices = indices
        self.stats = stats
        self.full_train_data = train_data
        self.train_data_keys_without_action = train_data_keys_without_action
        self.pred_horizon = pred_horizon
        self.action_horizon = action_horizon
        self.obs_horizon = obs_horizon
        self.episode_ends = episode_ends

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        # get the start/end indices for this datapoint
        buffer_start_idx, buffer_end_idx, sample_start_idx, sample_end_idx = self.indices[idx]

        # get nomralized data using these indices
        nsample = sample_sequence(
            train_data=self.full_train_data,
            sequence_length=self.pred_horizon,
            buffer_start_idx=buffer_start_idx,
            buffer_end_idx=buffer_end_idx,
            sample_start_idx=sample_start_idx,
            sample_end_idx=sample_end_idx
        )
        
        for key in self.train_data_keys_without_action:
            nsample[key] = nsample[key][:self.obs_horizon, :]

        if not self.return_images:
            window_start = buffer_start_idx - sample_start_idx
            episode_id = int(np.searchsorted(
                self.episode_ends, buffer_start_idx, side='right'
            ))
            episode_start = 0 if episode_id == 0 else int(self.episode_ends[episode_id - 1])
            episode_end = int(self.episode_ends[episode_id])
            nsample['policy_image_indices'] = np.clip(
                window_start + np.arange(self.obs_horizon),
                episode_start, episode_end - 1,
            ).astype(np.int64)
            nsample.pop('full_image', None)
            nsample.pop('hand_image', None)


        # for key in nsample.keys():
        #     nsample[key] = nsample[key][:self.obs_horizon, :]
        

        # nsample['agent_pose'] = nsample['agent_pose'][:self.obs_horizon, :]
        # nsample['full_image'] = nsample['full_image'][:self.obs_horizon, :]
        # if 'idleness' in nsample:
        #     nsample['idleness'] = nsample['idleness'][:self.obs_horizon, :]
        # nsample['goal_state'] = nsample['goal_state'][:self.obs_horizon, :]
        # nsample['step_count'] = nsample['step_count'][:self.obs_horizon, :] 
        # nsample['step_count_normalised'] = nsample['step_count_normalised'][:self.pred_horizon, :] 
        

        return nsample
