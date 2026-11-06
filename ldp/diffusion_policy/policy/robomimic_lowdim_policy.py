from typing import Dict
import torch
from diffusion_policy.model.common.normalizer import LinearNormalizer
from diffusion_policy.policy.base_lowdim_policy import BaseLowdimPolicy
import torch.nn as nn

from robomimic.algo import algo_factory
from robomimic.algo.algo import PolicyAlgo
import robomimic.utils.obs_utils as ObsUtils
from diffusion_policy.common.robomimic_config_util import get_robomimic_config
from diffusion_policy.common.pytorch_util import dict_apply, replace_submodules
import robomimic.models.base_nets as rmbn
import diffusion_policy.model.vision.crop_randomizer as dmvc

class RobomimicLowdimPolicy(BaseLowdimPolicy):
    def __init__(self, 
            algo_name='bc_rnn',
            obs_type='low_dim',
            task_name='square',
            dataset_type='ph',
            crop_shape=(76,76),
            obs_encoder_dir=None,
            shape_meta=None,
            past_action_pred=False,
            obs_encoder_group_norm= True,
            eval_fixed_crop= True
        ):
        super().__init__()
        action_dim = shape_meta["action"]["shape"][0]
        obs_dim = shape_meta["obs"]["embedding"]["shape"][0]
        del shape_meta["obs"]["embedding"]
        # key for robomimic obs input
        # previously this is 'object', 'robot0_eef_pos' etc
        obs_key = 'obs'
        self.obs_encoder_dir = obs_encoder_dir
        self.shape_meta = shape_meta
        self.past_action_pred = past_action_pred

        config = get_robomimic_config(
            algo_name=algo_name,
            hdf5_type="low_dim",
            task_name=task_name,
            dataset_type=dataset_type)
        with config.unlocked():
            config.observation.modalities.obs.low_dim = [obs_key]
        
        ObsUtils.initialize_obs_utils_with_config(config)
        model: PolicyAlgo = algo_factory(
                algo_name=config.algo_name,
                config=config,
                obs_key_shapes={obs_key: [obs_dim]},
                ac_dim=action_dim,
                device='cpu',
            )
        self.model = model
        self.nets = model.nets
        self.normalizer = LinearNormalizer()
        self.obs_key = obs_key
        self.config = config

        action_shape = shape_meta['action']['shape']
        assert len(action_shape) == 1
        action_dim = action_shape[0]
        obs_shape_meta = shape_meta['obs']
        obs_config = {
            'low_dim': [],
            'rgb': [],
            'depth': [],
            'scan': []
        }
        obs_key_shapes = dict()
        for key, attr in obs_shape_meta.items():
            shape = attr['shape']
            obs_key_shapes[key] = list(shape)

            type = attr.get('type', 'low_dim')
            if type == 'rgb':
                obs_config['rgb'].append(key)
            elif type == 'low_dim':
                obs_config['low_dim'].append(key)
            else:
                raise RuntimeError(f"Unsupported obs type: {type}")

        # get raw robomimic config
        config = get_robomimic_config(
            algo_name='bc_rnn',
            hdf5_type='image',
            task_name='square',
            dataset_type='ph')
        
        with config.unlocked():
            # set config with shape_meta
            config.observation.modalities.obs = obs_config

            if crop_shape is None:
                for key, modality in config.observation.encoder.items():
                    if modality.obs_randomizer_class == 'CropRandomizer':
                        modality['obs_randomizer_class'] = None
            else:
                # set random crop parameter
                ch, cw = crop_shape
                for key, modality in config.observation.encoder.items():
                    if modality.obs_randomizer_class == 'CropRandomizer':
                        modality.obs_randomizer_kwargs.crop_height = ch
                        modality.obs_randomizer_kwargs.crop_width = cw

        # init global state
        ObsUtils.initialize_obs_utils_with_config(config)

        # load model
        policy: PolicyAlgo = algo_factory(
                algo_name=config.algo_name,
                config=config,
                obs_key_shapes=obs_key_shapes,
                ac_dim=action_dim,
                device='cpu',
            )

        obs_encoder = policy.nets['policy'].nets['encoder'].nets['obs']

        if obs_encoder_group_norm:
            # replace batch norm with group norm
            replace_submodules(
                root_module=obs_encoder,
                predicate=lambda x: isinstance(x, nn.BatchNorm2d),
                func=lambda x: nn.GroupNorm(
                    num_groups=x.num_features//16, 
                    num_channels=x.num_features)
            )
            # obs_encoder.obs_nets['agentview_image'].nets[0].nets
        
        # obs_encoder.obs_randomizers['agentview_image']
        if eval_fixed_crop:
            replace_submodules(
                root_module=obs_encoder,
                predicate=lambda x: isinstance(x, rmbn.CropRandomizer),
                func=lambda x: dmvc.CropRandomizer(
                    input_shape=x.input_shape,
                    crop_height=x.crop_height,
                    crop_width=x.crop_width,
                    num_crops=x.num_crops,
                    pos_enc=x.pos_enc
                )
            )
        self.obs_encoder = obs_encoder


    def to(self,*args,**kwargs):
        device, dtype, non_blocking, convert_to_format = torch._C._nn._parse_to(*args, **kwargs)
        if device is not None:
            self.model.device = device
        super().to(*args,**kwargs)
    
    # =========== inference =============
    def predict_action(self, obs_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        # TODO get all vision

        if "embedding" in obs_dict:
            obs = obs_dict["embedding"]
        else:
            nobs = self.normalizer.normalize(obs_dict)
            this_nobs = dict_apply(nobs, lambda x: x[:,:,...].reshape(-1,*x.shape[2:]))
            B, To, _ = nobs["robot0_eef_pos"].shape
            obs = self.obs_encoder(this_nobs)
            obs = obs.reshape(B, To, -1)
            obs = obs[:,[-1]]
        
        assert obs.shape[1] == 1
        robomimic_obs_dict = {self.obs_key: obs[:,0,:]}
        naction = self.model.get_action(robomimic_obs_dict)
        action = self.normalizer['action'].unnormalize(naction)
        # (B, Da)
        result = {
            'action': action[:,None,:] # (B, 1, Da)
        }
        return result
    
    def reset(self):
        self.model.reset()
        
    # =========== training ==============
    def set_normalizer(self, normalizer: LinearNormalizer):
        self.normalizer.load_state_dict(normalizer.state_dict())
    
    def train_on_batch(self, batch, epoch, validate=False):
        naction = self.normalizer["action"].normalize(batch["action"])
        nobs = batch["obs"]["embedding"]
        robomimic_batch = {
            'obs': {self.obs_key: nobs},
            'actions': naction
        }
        input_batch = self.model.process_batch_for_training(
            robomimic_batch)
        info = self.model.train_on_batch(
            batch=input_batch, epoch=epoch, validate=validate)
        # keys: losses, predictions
        return info

    def get_optimizer(self):
        return self.model.optimizers['policy']
