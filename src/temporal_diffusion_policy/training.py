from dataclasses import dataclass, asdict

import os
import time
import random
from typing import Literal, Union
import draccus
import numpy as np
import torch
import torch.nn as nn
import wandb
import atexit

from sklearn.model_selection import train_test_split
from torch.utils.data import Subset, DataLoader

from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from diffusers.schedulers.scheduling_ddim import DDIMScheduler
from diffusers.training_utils import EMAModel
from diffusers.optimization import get_scheduler
from temporal_diffusion_policy.conditional_flow_matching import ConditionalFlowMatcher
from tqdm.auto import tqdm

from temporal_diffusion_policy.env_mappings import get_env_settings_from_dataset_name
from temporal_diffusion_policy.model import ConditionalUnet1D
from temporal_diffusion_policy.dataset_generation import NormalizationType, TemporalConditioningDataset
from temporal_diffusion_policy.utils import best_effort_notify, get_pc_hostname, write_post_train_info_to_screen_specific_tmpfile
from temporal_diffusion_policy.vision_encoder import get_resnet,get_resnet_1_channel, replace_bn_with_gn


@dataclass
class ExperimentConfig:
    wandb_entity: str | None = None  # Use the logged-in account or WANDB_ENTITY.
    wandb_project: str = "TemporalDiffusionPolicy"
    policy_type: Literal["diffusion", "flow_matching"] = "diffusion"
    flow_matching_sigma: float = 0.0 # between [0, 1) for flow matching. Ignored if policy_type is "diffusion"]
    # Flow matching implementation taken from https://github.com/HRI-EU/flow_matching/blob/master/examples/flow_pusht.py

    # Policy parameters
    batch_size: int = 64 # Adjust to the available GPU memory.
    lr: float = 1e-4 # use 1e-3 for higher batch size, 1e-4 is suitable for 64 batch size.
    num_epochs: int = 200 # 200 epochs for the paper configuration.
    num_diffusion_iters: int = 100
    num_images: int = 1
    save_every: int = 50
    val_every: int = 10
    validation_split: float = 0.2  # Fraction held out for validation; 0.0 trains on all data and skips validation.
    validation_seed: int | None = None  # Override the split seed independently of training randomness.
    pred_horizon: int = 16
    obs_horizon: int = 2
    action_horizon: int = 8
    save_ema_weights: bool = False  # Whether to save network weights or EMA weights.
    cond_predict_scale: bool = False  # Match the WaitAtGoal policy architecture.
    # Dataset parameters
    dataset_path: str = "data/demonstration/wait_at_goal_dataset_v4.zarr"
    checkpoint_root: str = "data/checkpoints"
    override_checkpoint_name: str = ""
    use_wandb:  bool = True
    num_dataloader_workers: int = 2
    cache_images_on_gpu: bool = False  # Faster input path; consumes GPU RAM.
    seed: int | None = None  # Seed Python, NumPy, and PyTorch when provided.

    use_ddim_sampling: bool = False # If using ddim, inference can run with less timesteps
    
    # Additional experiment parameters
    use_idleness_encoding: bool = False # Idleness (ID)
    use_direct_idleness_conditioning: bool = False # Instantaneous binary idleness fed directly to the policy.
    direct_idleness_threshold: float = 0.03 # Idle when |v| / vmax is below this threshold.
    use_sinusoidal_progress_encoding: bool = False # Sinusoidal Progress (SiP)
    sinusoidal_progress_encoding_dim: int = 128
    use_saturating_progress_encoding: bool = False # Saturating Progress (SaP)
    saturating_progress_encoding_omega: Union[float, None] = None # if None, will be computed based on dataset max step count
    idleness_encoding_alpha: Union[float, None] = None # # if None, will be computed based on dataset max velocity
    idleness_encoding_rest_thresh: float = 0.03 # velocity threshold for being considered "at rest". In normalized velocity units (i.e. 0.03 means 3 percent of max velocity)
    use_temporal_time_signal: bool = False # normalized step_count conditioning
    use_agent_velocity_conditioning: bool = False # normalized agent_velocity conditioning


@draccus.wrap()
def main(cfg: ExperimentConfig):

    cfg.wandb_entity = cfg.wandb_entity or os.environ.get("WANDB_ENTITY")
    if cfg.direct_idleness_threshold <= 0:
        raise ValueError("direct_idleness_threshold must be positive")
    if cfg.seed is not None:
        random.seed(cfg.seed)
        np.random.seed(cfg.seed)
        torch.manual_seed(cfg.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(cfg.seed)
        print(f"Using training seed: {cfg.seed}")
    env_settings = get_env_settings_from_dataset_name(cfg.dataset_path)

    # Ensure dataset path exists
    if not os.path.exists(cfg.dataset_path):
        raise FileNotFoundError(f"Dataset path {cfg.dataset_path} does not exist.")
    
    wandb_log_config = asdict(cfg)
    wandb_log_config['img_dim'] = env_settings.img_dim
    current_timestamp = time.strftime("%Y%m%d_%H%M%S")
    checkpoint_name = cfg.override_checkpoint_name or current_timestamp
    if cfg.seed is not None:
        seed_suffix = f"_seed{cfg.seed}"
        if not checkpoint_name.endswith(seed_suffix):
            checkpoint_name += seed_suffix
    checkpoint_dir = os.path.join(cfg.checkpoint_root, checkpoint_name)

    wandb_log_config['checkpoint_dir'] = checkpoint_dir
    wandb_log_config['host'] = get_pc_hostname()


    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    
    config_path = os.path.join(checkpoint_dir, 'config.txt')

    def write_config() -> None:
        """Persist runtime fields such as the W&B run identity alongside config."""
        with open(config_path, 'w') as f:
            for key, value in wandb_log_config.items():
                f.write(f"{key}: {value}\n")

    # Save a configuration immediately so an active run is discoverable.
    write_config()

    #|o|o|                             observations: 2
    #| |a|a|a|a|a|a|a|a|               actions executed: 8
    #|p|p|p|p|p|p|p|p|p|p|p|p|p|p|p|p| actions predicted: 16

    # create dataset from file
    # Observations, actions and normalisations
    dataset_keys_and_normalisations = {
        'full_image': NormalizationType.NONE, 
        'hand_image': NormalizationType.NONE,
        'action': NormalizationType.MIN_MAX, # 
        'agent_pose': NormalizationType.MIN_MAX, # Observations that will be normalized to [-1, 1] using min-max normalization
        
    }

    if cfg.use_idleness_encoding or cfg.use_direct_idleness_conditioning or cfg.use_agent_velocity_conditioning:
        dataset_keys_and_normalisations['agent_velocity'] = NormalizationType.MIN_MAX
    if cfg.use_temporal_time_signal:
        dataset_keys_and_normalisations['step_count'] = NormalizationType.MIN_MAX
    

    print("Creating Dataset...")
    dataset = TemporalConditioningDataset(
        dataset_path=cfg.dataset_path,
        pred_horizon=cfg.pred_horizon,
        obs_horizon=cfg.obs_horizon,
        action_horizon=cfg.action_horizon,
        dataset_keys_and_normalisation=dataset_keys_and_normalisations,
        normalization_mask=env_settings.normalization_mask,
        normalization_stats_overrides=env_settings.normalization_stats_overrides,
        use_idleness_encoding=cfg.use_idleness_encoding,
        use_direct_idleness_conditioning=cfg.use_direct_idleness_conditioning,
        direct_idleness_threshold=cfg.direct_idleness_threshold,
        use_sinusoidal_progress_encoding=cfg.use_sinusoidal_progress_encoding,
        sinusoidal_progress_encoding_parameters={'d_model': cfg.sinusoidal_progress_encoding_dim, 'power_value': 10000.0, 'normalize_by_episode_length': False},
        use_saturating_progress_encoding=cfg.use_saturating_progress_encoding,
        idleness_encoding_alpha=cfg.idleness_encoding_alpha,
        saturating_progress_encoding_omega=cfg.saturating_progress_encoding_omega,
        idleness_encoding_rest_thresh=cfg.idleness_encoding_rest_thresh,
        return_images=not cfg.cache_images_on_gpu,
    )
    # save training data statistics (min, max) for each dim
    stats = dataset.stats
    
    if cfg.use_saturating_progress_encoding:
        wandb_log_config['in_use_saturating_progress_encoding_omega'] = dataset.in_use_saturating_progress_encoding_omega
    
    if cfg.use_idleness_encoding:
        wandb_log_config['in_use_idleness_encoding_alpha'] = dataset.in_use_idleness_encoding_alpha
    if cfg.use_direct_idleness_conditioning:
        wandb_log_config['direct_idleness_vmax'] = dataset.direct_idleness_vmax

    if cfg.use_wandb:
        run = wandb.init(entity=cfg.wandb_entity, project=cfg.wandb_project, config=wandb_log_config)
        wandb_log_config['wandb_entity'] = run.entity
        wandb_log_config['wandb_run_name'] = run.name
        wandb_log_config['wandb_run_id'] = run.id
        write_config()
        
    if not 0.0 <= cfg.validation_split < 1.0:
        raise ValueError(
            f"validation_split must be in [0.0, 1.0), got {cfg.validation_split}."
        )
    if cfg.validation_split > 0.0:
        train_indices, val_indices = train_test_split(
            np.arange(len(dataset)),
            test_size=cfg.validation_split,
            random_state=(
                cfg.validation_seed if cfg.validation_seed is not None
                else cfg.seed if cfg.seed is not None else 42
            ),
        )
        train_dataset = Subset(dataset, train_indices)
        val_dataset = Subset(dataset, val_indices)
    else:
        train_dataset = dataset
        val_dataset = None

    dataloader_workers = cfg.num_dataloader_workers

    # train/val dataloaders
    train_loader = DataLoader(
        train_dataset, batch_size=cfg.batch_size, num_workers=dataloader_workers,
        shuffle=True, pin_memory=dataloader_workers > 0, persistent_workers=False)

    val_loader = None
    if val_dataset is not None:
        val_loader = DataLoader(
            val_dataset, batch_size=cfg.batch_size, num_workers=dataloader_workers,
            shuffle=False, pin_memory=dataloader_workers > 0, persistent_workers=False)
    
    # visualize data in batch
    batch = next(iter(train_loader))
    print("batch['agent_pose'].shape:", batch['agent_pose'].shape)
    print("batch['action'].shape:", batch['action'].shape)
    print("batch['full_image'].shape:" if not cfg.cache_images_on_gpu else "batch['policy_image_indices'].shape:", (batch['policy_image_indices'] if cfg.cache_images_on_gpu else batch['full_image']).shape)

    ######NETWORK#######
    # construct ResNet encoder
    # if you have multiple camera views, use seperate encoder weights for each view.
    resnet_type = env_settings.resnet_type
    vision_encoder = get_resnet(resnet_type) # e.g. 'resnet18'

    if cfg.num_images > 1:
        vision_encoder_hand = get_resnet(resnet_type)

    # IMPORTANT!
    # replace all BatchNorm with GroupNorm to work with EMA
    # performance will tank if you forget to do this!
    vision_encoder = replace_bn_with_gn(vision_encoder)
    if cfg.num_images > 1:
        vision_encoder_hand = replace_bn_with_gn(vision_encoder_hand)

    # ResNet18 has output dim of 512
    vision_feature_dim = vision_encoder.inplanes # 512
    
    lowdim_obs_dim = env_settings.agent_pose_dim

    if cfg.use_idleness_encoding:
        lowdim_obs_dim += 1  # add 1 dim for conditioning on idleness

    if cfg.use_direct_idleness_conditioning:
        lowdim_obs_dim += 1  # instantaneous binary idleness input

    if cfg.use_agent_velocity_conditioning:
        velocity_dim = int(np.asarray(dataset.full_train_data['agent_velocity']).shape[-1])
        lowdim_obs_dim += velocity_dim
    
    if cfg.use_saturating_progress_encoding:
        lowdim_obs_dim += 1  # add 1 dim for conditioning on saturating progress
    
    if cfg.use_sinusoidal_progress_encoding:
        lowdim_obs_dim += cfg.sinusoidal_progress_encoding_dim  # add 1 dim for conditioning on sinusoidal progress 

    if cfg.use_temporal_time_signal:
        lowdim_obs_dim += 1

    # observation feature has 514 dims in total per step
    obs_dim = vision_feature_dim 

    if cfg.num_images > 1:
        obs_dim += vision_encoder_hand.inplanes # add another 512 dims if we have a second camera view (e.g. hand cam)

    obs_dim += lowdim_obs_dim
    
    action_dim = env_settings.action_dim

    # create network object
    noise_pred_net = ConditionalUnet1D(
        input_dim=action_dim,
        global_cond_dim=obs_dim*cfg.obs_horizon,
        cond_predict_scale=cfg.cond_predict_scale,
    )

    # the final arch has 2 parts
    nets = nn.ModuleDict({
        'vision_encoder': vision_encoder,
        'noise_pred_net': noise_pred_net
    })
    if cfg.num_images > 1:
        nets['vision_encoder_hand'] = vision_encoder_hand

    # demo
    with torch.no_grad():
        # example inputs
        # batch size is first dimension
        demo_batch_size = 1

        image = torch.zeros((demo_batch_size, cfg.obs_horizon, 3, env_settings.img_dim, env_settings.img_dim))
        image_features = nets['vision_encoder'](image.flatten(end_dim=1))
        image_features = image_features.reshape(*image.shape[:2], -1)

        if cfg.num_images > 1:
            hand_image = torch.zeros((demo_batch_size, cfg.obs_horizon, 3, env_settings.img_dim, env_settings.img_dim))
            hand_image_features = nets['vision_encoder_hand'](hand_image.flatten(end_dim=1))
            hand_image_features = hand_image_features.reshape(*hand_image.shape[:2], -1)
            image_features = torch.cat([image_features, hand_image_features], dim=-1)

        agent_pose = torch.zeros((demo_batch_size, cfg.obs_horizon, env_settings.agent_pose_dim))
        
        obs_data = [image_features, agent_pose]

        if cfg.use_idleness_encoding:
            idleness = torch.zeros((demo_batch_size, cfg.obs_horizon, 1))
            obs_data.append(idleness)

        if cfg.use_direct_idleness_conditioning:
            direct_idleness = torch.zeros((demo_batch_size, cfg.obs_horizon, 1))
            obs_data.append(direct_idleness)

        if cfg.use_agent_velocity_conditioning:
            velocity = torch.zeros((demo_batch_size, cfg.obs_horizon, velocity_dim))
            obs_data.append(velocity)
        
        if cfg.use_saturating_progress_encoding:
            saturating_progress_encoding = torch.zeros((demo_batch_size, cfg.obs_horizon, 1))
            obs_data.append(saturating_progress_encoding)
        
        if cfg.use_sinusoidal_progress_encoding:
            geometric_progress_pe = torch.zeros((demo_batch_size, cfg.obs_horizon, cfg.sinusoidal_progress_encoding_dim))
            obs_data.append(geometric_progress_pe)

        if cfg.use_temporal_time_signal:
            time_signal = torch.zeros((demo_batch_size, cfg.obs_horizon, 1))
            obs_data.append(time_signal)


        obs = torch.cat(obs_data, dim=-1)
        print(obs.shape)

        noised_action = torch.randn((demo_batch_size, cfg.pred_horizon, action_dim))
        diffusion_iter = torch.zeros((1,))

        noise = nets['noise_pred_net'](
            sample=noised_action,
            timestep=diffusion_iter,
            global_cond=obs.flatten(start_dim=1))

    noise_scheduler = None
    flow_matcher = None
    if cfg.policy_type == "diffusion":
        if cfg.use_ddim_sampling:
            noise_scheduler = DDIMScheduler(
                num_train_timesteps=cfg.num_diffusion_iters,
                beta_schedule='squaredcos_cap_v2',
                clip_sample=True,
                prediction_type='epsilon'
            )
        else:
            noise_scheduler = DDPMScheduler(
                num_train_timesteps=cfg.num_diffusion_iters,
                beta_schedule='squaredcos_cap_v2',
                clip_sample=True,
                prediction_type='epsilon'
            )
    elif cfg.policy_type == "flow_matching":
        flow_matcher = ConditionalFlowMatcher(sigma=cfg.flow_matching_sigma)
    
    # device transfer
    device = torch.device('cuda')
    _ = nets.to(device)

    raw_image_gpu_cache = None
    if cfg.cache_images_on_gpu:
        raw_image_gpu_cache = {}
        image_keys = ['full_image'] + (['hand_image'] if cfg.num_images > 1 else [])
        for key in image_keys:
            cached = torch.from_numpy(np.ascontiguousarray(dataset.full_train_data[key])).to(device)
            raw_image_gpu_cache[key] = cached
            print(f"Cached raw {key} images on GPU ({cached.nbytes / 1024 ** 3:.2f} GiB).")


    # Exponential Moving Average
    ema = EMAModel(
        parameters=nets.parameters(),
        power=0.75)

    # Standard ADAM optimizer
    optimizer = torch.optim.AdamW(
        params=(
            list(nets.parameters())
            + ([])
        ),
        lr=cfg.lr, weight_decay=1e-6)

    # Cosine LR schedule with linear warmup
    lr_scheduler = get_scheduler(
        name='cosine',
        optimizer=optimizer,
        num_warmup_steps=500,
        num_training_steps=len(train_loader) * cfg.num_epochs
    )

    # Ensure the input and model weights are of the same type
    for name, param in nets.named_parameters():
        if param.dtype != torch.float32:
            raise TypeError(f"Parameter {name} is not of type torch.float32")

    def run_forward_pass(nbatch):
        if raw_image_gpu_cache is None:
            nimage = nbatch['full_image'][:, :cfg.obs_horizon].to(device, non_blocking=True)
        else:
            image_indices = nbatch['policy_image_indices'][:, :cfg.obs_horizon].to(device, non_blocking=True)
            nimage = raw_image_gpu_cache['full_image'][image_indices]
        nagent_pose = nbatch['agent_pose'][:, :cfg.obs_horizon].to(device, non_blocking=True)
        naction = nbatch['action'].to(device, non_blocking=True)

        image_features = nets['vision_encoder'](nimage.flatten(end_dim=1))
        image_features = image_features.reshape(*nimage.shape[:2], -1)
        if cfg.num_images > 1:
            nhand_image = (
                nbatch['hand_image'][:, :cfg.obs_horizon].to(device, non_blocking=True)
                if raw_image_gpu_cache is None
                else raw_image_gpu_cache['hand_image'][image_indices]
            )
            hand_image_features = nets['vision_encoder_hand'](
                nhand_image.flatten(end_dim=1)
            ).reshape(*nhand_image.shape[:2], -1)
            image_features = torch.cat([image_features, hand_image_features], dim=-1)

        obs_features = [image_features, nagent_pose]
        if cfg.use_idleness_encoding:
            obs_features.append(nbatch['idleness'][:, :cfg.obs_horizon].to(device, non_blocking=True))
        if cfg.use_direct_idleness_conditioning:
            obs_features.append(nbatch['direct_idleness'][:, :cfg.obs_horizon].to(device, non_blocking=True))
        if cfg.use_agent_velocity_conditioning:
            obs_features.append(nbatch['agent_velocity'][:, :cfg.obs_horizon].to(device, dtype=torch.float32, non_blocking=True))
        if cfg.use_saturating_progress_encoding:
            obs_features.append(
                nbatch['saturating_progress_encoding'][:, :cfg.obs_horizon].to(device, non_blocking=True)
            )
        if cfg.use_sinusoidal_progress_encoding:
            obs_features.append(
                nbatch['sinusoidal_progress_encoding'][:, :cfg.obs_horizon].to(device, non_blocking=True)
            )
        if cfg.use_temporal_time_signal:
            obs_features.append(nbatch['step_count'][:, :cfg.obs_horizon].to(device, dtype=torch.float32, non_blocking=True))

        obs_cond = torch.cat(obs_features, dim=-1).flatten(start_dim=1)
        if cfg.policy_type == "diffusion":
            noise = torch.randn_like(naction)
            timesteps = torch.randint(
                0,
                noise_scheduler.config.num_train_timesteps,
                (naction.shape[0],),
                device=device,
            ).long()
            noisy_actions = noise_scheduler.add_noise(naction, noise, timesteps)
            target = noise
        elif cfg.policy_type == "flow_matching":
            x0 = torch.randn_like(naction)
            timesteps, noisy_actions, target = (
                flow_matcher.sample_location_and_conditional_flow(x0, naction)
            )

        model_input = noisy_actions
        model_output = nets['noise_pred_net'](
            model_input,
            timesteps,
            global_cond=obs_cond,
        )
        # In diffusion, loss is between noise_pred and noise. In flow matching, loss is between predicted flow and target flow.
        loss_diffusion = nn.functional.mse_loss(model_output, target)
        loss_total = loss_diffusion
        return {
            'loss_total': loss_total,
            'loss_diffusion': loss_diffusion,
        }

    best_validation_loss = float('inf')
    best_validation_checkpoint_path = None

    with tqdm(range(cfg.num_epochs), desc='Epoch') as tglobal:
        # epoch loop
        for epoch_idx in tglobal:
            nets.train()
            epoch_loss = list()
            epoch_diffusion_loss = list()
            # batch loop
            with tqdm(train_loader, desc='Batch', leave=False) as tepoch:
                for nbatch in tepoch:
                    losses = run_forward_pass(nbatch)
                    loss = losses['loss_total']

                    # optimize
                    loss.backward()
                    optimizer.step()
                    optimizer.zero_grad()
                    # step lr scheduler every batch
                    lr_scheduler.step()

                    # update Exponential Moving Average of the model weights
                    ema.step(nets.parameters())

                    # logging
                    loss_cpu = loss.item()
                    epoch_loss.append(loss_cpu)
                    epoch_diffusion_loss.append(losses['loss_diffusion'].item())
                    tepoch.set_postfix(loss=loss_cpu)
            tglobal.set_postfix(loss=np.mean(epoch_loss))

            # Validation
            ran_validation = val_loader is not None and (epoch_idx + 1) % cfg.val_every == 0
            if ran_validation:
                val_losses = []
                val_diffusion_losses = []
                nets.eval()
                with torch.no_grad():
                    with tqdm(val_loader, desc='Val', leave=False) as tval:
                        for nbatch in tval:
                            losses = run_forward_pass(nbatch)
                            val_losses.append(losses['loss_total'].item())
                            val_diffusion_losses.append(losses['loss_diffusion'].item())
                            
                val_loss_mean = np.mean(val_losses)
                val_diffusion_loss_mean = np.mean(val_diffusion_losses)
                nets.train()

            log_dict = {
                "epoch": epoch_idx,
                "train_loss": np.mean(epoch_loss),
                "train/loss_diffusion": np.mean(epoch_diffusion_loss),
                "train/loss_total": np.mean(epoch_loss),
                "lr": lr_scheduler.get_last_lr()[0]
            }

            if ran_validation:
                log_dict["val_loss"] = val_loss_mean
                log_dict.update({
                    "val/loss_diffusion": val_diffusion_loss_mean,
                    "val/loss_total": val_loss_mean,
                })
            if cfg.use_wandb:
                wandb.log(log_dict)
            if (epoch_idx + 1) % cfg.save_every == 0:
                if ran_validation and val_loss_mean >= best_validation_loss:
                    print(
                        f"Skipping checkpoint at epoch {epoch_idx + 1}: "
                        f"validation loss {val_loss_mean:.4f} is not better than "
                        f"best {best_validation_loss:.4f}."
                    )
                else:
                    checkpoint_suffix = f"valloss_{val_loss_mean:.4f}" if ran_validation else "no_val"
                    checkpoint_path = os.path.join(checkpoint_dir, f"epoch_{epoch_idx + 1}_{checkpoint_suffix}.pth")
                    if cfg.save_ema_weights:
                        ema.store(nets.parameters())
                        ema.copy_to(nets.parameters())
                    torch.save({
                        'epoch': epoch_idx + 1,
                        'model_state_dict': nets.state_dict(),
                        'loss': epoch_loss,
                        'val_loss': val_loss_mean if ran_validation else None,
                        'stats': stats,
                        'policy_type': cfg.policy_type,
                        'train_cfg': wandb_log_config,
                    }, checkpoint_path)
                    if cfg.save_ema_weights:
                        ema.restore(nets.parameters())
                    if ran_validation:
                        if best_validation_checkpoint_path is not None:
                            os.remove(best_validation_checkpoint_path)
                            print(
                                "Deleted previous higher-validation-loss checkpoint: "
                                f"{best_validation_checkpoint_path}"
                            )
                        best_validation_loss = val_loss_mean
                        best_validation_checkpoint_path = checkpoint_path

    checkpoint_path = os.path.join(checkpoint_dir, f"final.pth")
    if cfg.save_ema_weights:
        ema.store(nets.parameters())
        ema.copy_to(nets.parameters())
    torch.save({
        'epoch': cfg.num_epochs,
        'model_state_dict': nets.state_dict(),
        'loss': epoch_loss,
        'stats': stats,
        'policy_type': cfg.policy_type,
        'train_cfg': wandb_log_config
    }, checkpoint_path)
    if cfg.save_ema_weights:
        ema.restore(nets.parameters())
    if cfg.use_wandb:
        wandb.save(checkpoint_path)
    
    write_post_train_info_to_screen_specific_tmpfile(wandb_log_config)
    
    if cfg.use_wandb:
        best_effort_notify(f"Training complete for run {run.name}")
        atexit.register(wandb.finish)
        
        
if __name__ == '__main__':
    main()
