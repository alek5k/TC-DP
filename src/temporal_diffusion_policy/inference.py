from temporal_diffusion_policy.temporal_encodings import binary_idle_indicator, velocity_vmax
from dataclasses import asdict, dataclass
from pathlib import Path
import time
from typing import Literal, Optional, Sequence, Union
import cv2
import draccus
import numpy as np
import torch
import torch.nn as nn
import collections
import os
import re
import threading

import wandb

from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from diffusers.schedulers.scheduling_ddim import DDIMScheduler
from tqdm.auto import tqdm

from skvideo.io import vwrite
from temporal_diffusion_policy.env_mappings import create_pygame_env
from temporal_diffusion_policy.env_mappings import EnvName, EnvType, get_env_settings_from_dataset_name, get_env_settings_from_env_name
from temporal_diffusion_policy.replay_buffer import ReplayBuffer
from temporal_diffusion_policy.utils import best_effort_notify
from temporal_diffusion_policy.model import ConditionalUnet1D
from temporal_diffusion_policy.vision_encoder import get_resnet,get_resnet_1_channel, replace_bn_with_gn
from temporal_diffusion_policy.dataset_generation import NormalizationType, normalize_data, unnormalize_data
import pygame
from temporal_diffusion_policy.temporal_encodings import (
    calculate_growth_rate_parameter, 
    progress_positional_encoding_manual, 
    progress_positional_encoding_transformer, 
    progress_saturalising_encoding, 
    IdlenessEncoder, 
    PauseInducedPerturbation
)


def copy_env_state(from_env, to_env):
    to_env.sim.set_state(from_env.sim.get_state())
    to_env.sim.forward()

# checkpoint_path: Optional[str] = "data/checkpoints/20260325_133722"
# env_name: str = EnvName.LIFTQA_REAL
# num_episodes: int = 50
# prompt_user_before_each_episode: bool = True

@dataclass
class InferenceConfig:
    wandb_entity: str | None = None  # Use the logged-in account or WANDB_ENTITY.
    wandb_project: str = "TemporalDiffusionPolicy"
    checkpoint_path: Optional[str] = None
    wandb_run_name: Optional[str] = None # wandb run will be used to resolve checkpoint path
    get_checkpoint_from_screen_tmpfile: bool = False # get checkpoint path from screen-specific tmpfile, instead of wandb_run_name or checkpoint_path
    checkpoint_file: str = "final.pth"
    env_name: str = 'WaitAtGoal'
    num_episodes: int = 200
    save_video: bool = False
    render: bool = False
    video_folder: str = 'outputs/videos'
    policy_type: Literal["diffusion", "flow_matching"] = "diffusion"
    num_diffusion_iters: int = 100
    num_flow_matching_steps: int = 16
    num_images: int = 1
    image_key: str = "full_image"
    hand_image_key: str = "hand_image"
    agent_pose_key: str = "agent_pose"
    pred_horizon: int = 16
    obs_horizon: int = 2
    action_horizon: int = 8
    max_steps: int = 400
    seed: int = 200
    use_deterministic_rng: bool = True
    is_part_of_parallel_eval: bool = False
    parallel_eval_id: str = ""
    inference_dataset_root: str = "data/inference"
    generated_dataset_suffix: str = "" # extra identifier to append to dataset name
    use_wandb: bool = False
    cond_predict_scale: bool = False  # Match the WaitAtGoal policy architecture.
    prompt_user_before_each_episode: bool = False
    enable_enter_to_next_episode: bool = False # if True, pressing Enter during inference ends current episode and starts the next one

    use_ddim_sampling: bool = False # If using ddim, inference can run with less timesteps 

    # Paper proposed encodings
    use_idleness_encoding: bool = False # Idleness (ID)
    use_direct_idleness_conditioning: bool = False # Instantaneous binary idleness fed directly to the policy.
    direct_idleness_threshold: float = 0.03 # Idle when |v| / vmax is below this threshold.
    use_sinusoidal_progress_encoding: bool = False # Sinusoidal Progress (SiP)
    sinusoidal_progress_encoding_dim: int = 128
    use_saturating_progress_encoding: bool = False # Saturating Progress (SaP)
    saturating_progress_encoding_omega: Union[float, None] = None # if None, will attempt to load from train checkpoint, otherwise will be computed based on dataset max step count
    idleness_encoding_alpha: Union[float, None] = None # if None, will attempt to load from train checkpoint, otherwise will be computed based on dataset max velocity
    idleness_encoding_rest_thresh: float = 0.03 # velocity threshold for being considered "at rest". In normalized velocity units (i.e. 0.03 means 3 percent of max velocity)
    use_temporal_time_signal: bool = False # normalized step_count conditioning
    use_agent_velocity_conditioning: bool = False # normalized agent_velocity conditioning
    
    # PIP settings
    use_pause_induced_perturbation: bool = False # https://arxiv.org/pdf/2508.15669    
    pip_sigma: Optional[float | None] = None # if none, will use best effort defaults
    pip_window_size: Optional[int | None] = None # if none, will use best effort defaults
    pip_epsilon: Optional[float | None] = None # if none, will use best effort defaults
    
    # Experimental settings
    test_blank_image: bool = False # for ablation, use blank image as input


def generate_short_label_from_inference_config(cfg: InferenceConfig) -> str:
    label_str = ""
    label_str += "+T " if cfg.use_temporal_time_signal else ""
    label_str += "+V " if cfg.use_agent_velocity_conditioning else ""
    label_str += "+DI " if cfg.use_direct_idleness_conditioning else ""
    label_str += "+ID " if cfg.use_idleness_encoding else ""
    label_str += "+SaP " if cfg.use_saturating_progress_encoding else ""
    label_str += f"+SiP{cfg.sinusoidal_progress_encoding_dim}" if cfg.use_sinusoidal_progress_encoding else ""
    label_str += "PIP" if cfg.use_pause_induced_perturbation else ""
    label_str = label_str.replace(" ", "_").strip("_")
    return label_str




@draccus.wrap()
def main(cfg: InferenceConfig):
    t_start = time.time()

    if cfg.checkpoint_path is None and cfg.wandb_run_name is None and not cfg.get_checkpoint_from_screen_tmpfile:
        raise ValueError("Exactly one of --checkpoint_path or --wandb_run_name or --get_checkpoint_from_screen_tmpfile must be specified.")
    
    if cfg.wandb_run_name is not None:
        if not cfg.wandb_entity:
            cfg.wandb_entity = os.environ.get("WANDB_ENTITY")
        if not cfg.wandb_entity:
            raise ValueError("Specify --wandb_entity or WANDB_ENTITY when looking up a W&B run.")
        from temporal_diffusion_policy.utils import get_wandb_run_from_name
        run = get_wandb_run_from_name(cfg.wandb_run_name, cfg.wandb_entity, cfg.wandb_project)
        checkpoint_dir = run.config["checkpoint_dir"]
        cfg.checkpoint_path = checkpoint_dir

    elif cfg.get_checkpoint_from_screen_tmpfile:
        from temporal_diffusion_policy.utils import read_info_from_screen_specific_tmpfile
        info = read_info_from_screen_specific_tmpfile()
        if info is None or "checkpoint_dir" not in info:
            raise ValueError("Could not read checkpoint_dir from screen-specific tmpfile.")
        checkpoint_dir = info["checkpoint_dir"]
        cfg.checkpoint_path = checkpoint_dir
        wandb_run_name = info.get("wandb_run_name", None)
        cfg.wandb_run_name = wandb_run_name

    full_checkpoint_path = cfg.checkpoint_path + "/" + cfg.checkpoint_file

    if not os.path.exists(full_checkpoint_path):
        raise FileNotFoundError(f"Dataset path {full_checkpoint_path} does not exist.")

    checkpoint = torch.load(full_checkpoint_path, weights_only=False)
    stats = checkpoint['stats']
    train_cfg = checkpoint.get('train_cfg', {})
    cfg.cond_predict_scale = bool(
        train_cfg.get('cond_predict_scale', cfg.cond_predict_scale)
    )
    cfg.num_images = int(train_cfg.get('num_images', cfg.num_images))
    cfg.use_idleness_encoding = bool(
        train_cfg.get('use_idleness_encoding', cfg.use_idleness_encoding)
    )
    cfg.use_sinusoidal_progress_encoding = bool(
        train_cfg.get(
            'use_sinusoidal_progress_encoding',
            cfg.use_sinusoidal_progress_encoding,
        )
    )
    cfg.sinusoidal_progress_encoding_dim = int(
        train_cfg.get(
            'sinusoidal_progress_encoding_dim',
            cfg.sinusoidal_progress_encoding_dim,
        )
    )
    cfg.use_saturating_progress_encoding = bool(
        train_cfg.get(
            'use_saturating_progress_encoding',
            cfg.use_saturating_progress_encoding,
        )
    )
    cfg.idleness_encoding_rest_thresh = float(
        train_cfg.get('idleness_encoding_rest_thresh', cfg.idleness_encoding_rest_thresh)
    )
    cfg.use_temporal_time_signal = bool(train_cfg.get('use_temporal_time_signal', cfg.use_temporal_time_signal))
    cfg.use_agent_velocity_conditioning = bool(train_cfg.get('use_agent_velocity_conditioning', cfg.use_agent_velocity_conditioning))
    cfg.use_direct_idleness_conditioning = bool(
        train_cfg.get('use_direct_idleness_conditioning', cfg.use_direct_idleness_conditioning)
    )
    cfg.direct_idleness_threshold = float(
        train_cfg.get('direct_idleness_threshold', cfg.direct_idleness_threshold)
    )
    if cfg.direct_idleness_threshold <= 0:
        raise ValueError("direct_idleness_threshold must be positive")
    direct_idleness_vmax = None
    if cfg.use_direct_idleness_conditioning:
        direct_idleness_vmax = float(
            train_cfg.get(
                'direct_idleness_vmax',
                velocity_vmax(np.asarray(stats['agent_velocity']['max'])),
            )
        )
    # The checkpoint is authoritative for architecture/conditioning flags, so
    # build labels only after restoring them.
    checkpoint_name = Path(cfg.checkpoint_path).stem
    checkpoint_datestamp_prefix = f"{cfg.env_name}_"
    if checkpoint_name.startswith(checkpoint_datestamp_prefix):
        checkpoint_name = checkpoint_name.removeprefix(checkpoint_datestamp_prefix)
    # Training checkpoints carry their training seed; inference already stores
    # its rollout seed separately, so keep output names in the ckpt<datestamp>
    # convention.
    checkpoint_name = re.sub(r"_seed\d+$", "", checkpoint_name)
    if cfg.is_part_of_parallel_eval:
        inference_dataset_path = f"{cfg.inference_dataset_root}/{cfg.env_name}_{checkpoint_name}_{cfg.parallel_eval_id}_partial.zarr"
    else:
        inference_dataset_path = f"{cfg.inference_dataset_root}/{cfg.env_name}_ckpt{checkpoint_name}_{generate_short_label_from_inference_config(cfg)}.zarr"
    if cfg.generated_dataset_suffix:
        inference_dataset_path = inference_dataset_path.replace(".zarr", f"_{cfg.generated_dataset_suffix}.zarr")

    replay_buffer_mode = 'w'
    if os.path.exists(inference_dataset_path):
        print(f"Dataset already exists:\n  {inference_dataset_path}")
        print("Choose an option:")
        print("  [c] Continue and append to this dataset")
        print("  [n] Create a new dataset with an extra identifier")
        print("  [o] Overwrite this dataset and start again")
        print("  [a] Abort")
        choice = input("> ").strip().lower()

        if choice in ("c", "continue"):
            replay_buffer_mode = 'a'
            print(f"Appending to existing dataset:\n  {inference_dataset_path}")
        elif choice in ("o", "overwrite"):
            replay_buffer_mode = 'w'
            print(f"Overwriting inference dataset:\n  {inference_dataset_path}")
        elif choice in ("n", "new"):
            print("Please enter an extra identifier (e.g. run2, retryA, debug):")
            suffix = input("> ").strip()
            if not suffix:
                print("Aborted (empty identifier).")
                raise SystemExit(1)

            base, ext = os.path.splitext(inference_dataset_path)
            inference_dataset_path = f"{base}_{suffix}{ext}"
            print(f"Using new dataset path:\n  {inference_dataset_path}")
        else:
            print("Aborted.")
            raise SystemExit(1)
        
    if cfg.use_deterministic_rng:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True)

    print("=============================")
    if replay_buffer_mode == 'a':
        print(f"❗Appending to inference dataset at: {inference_dataset_path}")
    else:
        print(f"❗Generating inference dataset at: {inference_dataset_path}")
    print("=============================")

    replay_buffer = ReplayBuffer.create_from_path(inference_dataset_path, mode=replay_buffer_mode)
    
    if cfg.save_video:
        video_folder = os.path.join(cfg.video_folder)
        if not os.path.exists(video_folder):
            os.makedirs(video_folder)
    
    env_settings = get_env_settings_from_env_name(cfg.env_name)

    # device transfer
    device = torch.device('cuda')
    
    resnet_type = env_settings.resnet_type
    vision_encoder = get_resnet(resnet_type) # e.g. 'resnet18'
    vision_encoder = replace_bn_with_gn(vision_encoder)

    if cfg.num_images > 1:
        vision_encoder_hand = get_resnet(resnet_type)
        vision_encoder_hand = replace_bn_with_gn(vision_encoder_hand)

    # ResNet18 has output dim of 512
    vision_feature_dim = vision_encoder.inplanes

    # parameters
    # agent_pos is 2 dimensional
    lowdim_obs_dim = env_settings.agent_pose_dim

    if cfg.use_idleness_encoding:
        lowdim_obs_dim += 1

    if cfg.use_direct_idleness_conditioning:
        lowdim_obs_dim += 1

    if cfg.use_agent_velocity_conditioning:
        lowdim_obs_dim += int(np.asarray(stats['agent_velocity']['min']).size)
    
    if cfg.use_saturating_progress_encoding:
        lowdim_obs_dim += 1
    
    if cfg.use_sinusoidal_progress_encoding:
        lowdim_obs_dim += cfg.sinusoidal_progress_encoding_dim

    if cfg.use_temporal_time_signal:
        lowdim_obs_dim += 1

    obs_dim = vision_feature_dim
    if cfg.num_images > 1:
        obs_dim += vision_encoder_hand.inplanes
    obs_dim += lowdim_obs_dim
    action_dim = env_settings.action_dim # 2

    # create network object
    noise_pred_net = ConditionalUnet1D(
        input_dim=action_dim,
        global_cond_dim=obs_dim*cfg.obs_horizon,
        cond_predict_scale=cfg.cond_predict_scale,
    )
    # Create EMA models
    ema_nets = nn.ModuleDict({
    'vision_encoder': vision_encoder,
    'noise_pred_net': noise_pred_net
    })
    if cfg.num_images > 1:
        ema_nets['vision_encoder_hand'] = vision_encoder_hand

    # Move models to GPU
    ema_nets = ema_nets.to(device)
    ema_nets.load_state_dict(checkpoint['model_state_dict'])
    
    
    if cfg.use_pause_induced_perturbation:
        if env_settings.env_type == EnvType.ROBOSUITE:
            pip_epsilon = 0.01 if cfg.pip_epsilon is None else cfg.pip_epsilon
            pip_window_size = 40 if cfg.pip_window_size is None else cfg.pip_window_size
            pip_sigma = 0.8 if cfg.pip_sigma is None else cfg.pip_sigma
            pause_induced_perturbation = PauseInducedPerturbation(epsilon=pip_epsilon, window_size=pip_window_size, sigma=pip_sigma)
            replay_buffer.root.attrs['pip_epsilon'] = pip_epsilon
            replay_buffer.root.attrs['pip_window_size'] = pip_window_size
            replay_buffer.root.attrs['pip_sigma'] = pip_sigma
        elif env_settings.env_type == EnvType.PYGAME:
            pip_epsilon = 0.03 if cfg.pip_epsilon is None else cfg.pip_epsilon
            pip_window_size = 20 if cfg.pip_window_size is None else cfg.pip_window_size
            pip_sigma = 0.8 if cfg.pip_sigma is None else cfg.pip_sigma
            pause_induced_perturbation = PauseInducedPerturbation(epsilon=pip_epsilon, window_size=pip_window_size, sigma=pip_sigma, q_min=stats[cfg.agent_pose_key]['min'], q_max=stats[cfg.agent_pose_key]['max'])
            replay_buffer.root.attrs['pip_epsilon'] = pip_epsilon
            replay_buffer.root.attrs['pip_window_size'] = pip_window_size
            replay_buffer.root.attrs['pip_sigma'] = pip_sigma
        elif env_settings.env_name == EnvName.LIFTQA_REAL:
            pip_epsilon = 0.02 if cfg.pip_epsilon is None else cfg.pip_epsilon
            pip_window_size = 30 if cfg.pip_window_size is None else cfg.pip_window_size
            pip_sigma = 0.95 if cfg.pip_sigma is None else cfg.pip_sigma
            pause_induced_perturbation = PauseInducedPerturbation(epsilon=pip_epsilon, window_size=pip_window_size, sigma=pip_sigma)
            replay_buffer.root.attrs['pip_epsilon'] = pip_epsilon
            replay_buffer.root.attrs['pip_window_size'] = pip_window_size
            replay_buffer.root.attrs['pip_sigma'] = pip_sigma

    if cfg.use_idleness_encoding:
        if cfg.idleness_encoding_alpha is not None:
            alpha = cfg.idleness_encoding_alpha
        elif checkpoint['train_cfg'].get('in_use_idleness_encoding_alpha', None) is not None:
            alpha = checkpoint['train_cfg']['in_use_idleness_encoding_alpha']
        elif checkpoint['train_cfg'].get('idleness_encoding_alpha', None) is not None:
            alpha = checkpoint['train_cfg']['idleness_encoding_alpha']
        else:
            alpha = calculate_growth_rate_parameter(max_steps=stats['step_count']['max'])
        cfg.idleness_encoding_alpha = alpha
        idleness_encoder = IdlenessEncoder(vmax=stats['agent_velocity']['max'], rest_thresh=cfg.idleness_encoding_rest_thresh, alpha=cfg.idleness_encoding_alpha)
        

    if cfg.use_saturating_progress_encoding:
        if cfg.saturating_progress_encoding_omega is not None:
            omega = cfg.saturating_progress_encoding_omega
        elif checkpoint['train_cfg'].get('in_use_saturating_progress_encoding_omega', None) is not None:
            omega = checkpoint['train_cfg']['in_use_saturating_progress_encoding_omega']
        elif checkpoint['train_cfg'].get('saturating_progress_encoding_omega', None) is not None:
            omega = checkpoint['train_cfg']['in_use_saturating_progress_encoding_omega']
        else:
            omega = calculate_growth_rate_parameter(max_steps=stats['step_count']['max'])
        cfg.saturating_progress_encoding_omega = omega
        
    for k,v in asdict(cfg).items():
        replay_buffer.root.attrs[k] = v
    
    replay_buffer.root.attrs['inference_dataset_path'] = inference_dataset_path

    wandb_run = None
    if cfg.use_wandb:
        wandb_run = wandb.init(
            entity=cfg.wandb_entity,
            project=cfg.wandb_project,
            config={**asdict(cfg), "inference_dataset_path": inference_dataset_path},
        )

    # for k,v in checkpoint['train_cfg'].items():
    #     replay_buffer.root.attrs[f'checkpoint_train_cfg.{k}'] = v

    noise_scheduler = None
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
    
    # limit enviornment interaction to 200 steps before termination
    
    if not cfg.render:
        os.environ["SDL_VIDEODRIVER"] = "dummy" # prevents rendering
    
    if cfg.use_pause_induced_perturbation:
        pip_events = 0

    manual_next_episode_event = threading.Event()
    if cfg.enable_enter_to_next_episode:
        def wait_for_enter_to_advance_episode():
            while True:
                input("Press ENTER to start next episode...\n")
                manual_next_episode_event.set()
                print("Enter pressed: ending current episode and advancing to next.")
                time.sleep(0.1)

        threading.Thread(target=wait_for_enter_to_advance_episode, daemon=True).start()

    if env_settings.env_type == EnvType.PYGAME:
        env = create_pygame_env(cfg.env_name, render_size=env_settings.img_dim)
    elif env_settings.env_type == EnvType.ROBOSUITE:
        import robosuite.utils.transform_utils as T
        from temporal_diffusion_policy.envs.robosuite.lift_qa import create_env, translate_3_dim_action_to_7d
        env = create_env(render=cfg.render, seed=cfg.seed, camera_height_width=(env_settings.img_dim, env_settings.img_dim))
        if cfg.use_pause_induced_perturbation:
            env_shadow = create_env(render=cfg.render, seed=cfg.seed, camera_height_width=(env_settings.img_dim, env_settings.img_dim))
            env_joint = create_env(render=cfg.render, seed=cfg.seed, camera_height_width=(env_settings.img_dim, env_settings.img_dim), use_joint_control=True)
            env_joint.motion_constrained = False
    elif env_settings.env_name == EnvName.LIFTQA_REAL:
        from temporal_diffusion_policy.envs.realworld.liftqa_real import LiftQARealEnvironment, LiftQARealConfig
        env_cfg = LiftQARealConfig(show_status_window=False, img_dim=env_settings.img_dim, init_robot_joints=np.array(np.deg2rad([64.27, -69.99, 70, -90, -90, -25])))
        env = LiftQARealEnvironment(env_cfg)
    elif env_settings.env_name == EnvName.PIANO:
        from temporal_diffusion_policy.envs.realworld.piano import PianoEnvironment, PianoConfig
        env_cfg = PianoConfig(enable_robot_movement=True, enable_audio=True, robot_ip_real='192.168.0.111', camera_serial='938422072678', img_dim=env_settings.img_dim, init_robot_joints=np.array(np.deg2rad([88.3272, -57.038, 109.4865, -52.311, 50.2942, -90])))
        env = PianoEnvironment(env_cfg)
        
    
    episode_id_start = replay_buffer.n_episodes if replay_buffer_mode == 'a' else 0
    if replay_buffer_mode == 'a':
        episodes_remaining = cfg.num_episodes - episode_id_start
        if episodes_remaining <= 0:
            print(
                f"Append requested, but dataset already has {episode_id_start} episodes "
                f"(target num_episodes={cfg.num_episodes}). Nothing to run."
            )
            raise SystemExit(0)
    else:
        episodes_remaining = cfg.num_episodes

    episode_metrics = []
    num_episodes_reached_step_limit = 0
    pos_enc_history = []

    final_episode_id_exclusive = episode_id_start + episodes_remaining
    for episode_id in range(episode_id_start, final_episode_id_exclusive):

        if cfg.enable_enter_to_next_episode:
            manual_next_episode_event.clear()

        if cfg.prompt_user_before_each_episode:
            input(f"Press enter to start episode {episode_id}")
        # use a seed >200 to avoid initial states seen in the training dataset
        # seed = np.random.randint(201, 25536)
        seed = cfg.seed + replay_buffer.n_episodes

        if env_settings.env_type == EnvType.PYGAME:
            env.seed(seed) 

        if cfg.use_deterministic_rng:
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            generator = torch.Generator(device=device).manual_seed(seed)
        else:
            generator = torch.Generator(device=device)
        
        # get first observation
        obs = env.reset()
        if cfg.use_pause_induced_perturbation:
            if env_settings.env_type == EnvType.ROBOSUITE:
                env_shadow.reset()
                env_joint.reset()
                pause_induced_perturbation.reset(q0=obs['joint_position'])
            elif env_settings.env_type == EnvType.PYGAME:
                pause_induced_perturbation.reset(q0=obs[cfg.agent_pose_key])
            elif env_settings.env_name == EnvName.LIFTQA_REAL:
                pause_induced_perturbation.reset(q0=obs[cfg.agent_pose_key])



        reward = 0.0
        info = {}
        if env_settings.env_type == EnvType.PYGAME:
            info = env._get_info()
        
        if cfg.use_idleness_encoding:
            idleness_encoder.reset()
            idleness = idleness_encoder.step(obs['agent_velocity'])
            obs['idleness'] = idleness

        if cfg.use_direct_idleness_conditioning:
            obs['direct_idleness'] = float(binary_idle_indicator(
                np.asarray(obs['agent_velocity'], dtype=np.float32).reshape(1, -1),
                direct_idleness_vmax,
                cfg.direct_idleness_threshold,
            )[0, 0])

        if cfg.use_saturating_progress_encoding:
            saturating_progress_encoding = progress_saturalising_encoding(t=0, omega=cfg.saturating_progress_encoding_omega)
            obs['saturating_progress_encoding'] = saturating_progress_encoding

        if cfg.use_sinusoidal_progress_encoding:
            pos_enc = progress_positional_encoding_transformer(t=0, d_model=cfg.sinusoidal_progress_encoding_dim) 
            obs['sinusoidal_progress_encoding'] = pos_enc
            pos_enc_history.append(pos_enc)

        if cfg.use_temporal_time_signal:
            obs['time_signal'] = normalize_data(
                np.array([[0]], dtype=np.float32), stats=stats['step_count']
            ).reshape(1)

        
        # keep a queue of last 2 steps of observations
        obs_deque = collections.deque([obs] * cfg.obs_horizon, maxlen=cfg.obs_horizon)
        info_deque = collections.deque([info] * cfg.obs_horizon, maxlen=cfg.obs_horizon)
        
        # save visualization and rewards
        if cfg.save_video:
            imgs = [env.render(mode='rgb_array')]

        rewards = list()
        done = False
        step_idx = 0

        episode = []

        with tqdm(total=cfg.max_steps, desc=f"Eval {cfg.env_name} Episode {episode_id+1}/{cfg.num_episodes}") as pbar:

            while not done:
                if cfg.enable_enter_to_next_episode and manual_next_episode_event.is_set():
                    print("Manual next-episode requested, ending current episode...")
                    done = True
                    break

                B = 1 

                # stack the last obs_horizon number of observations
                images = np.stack([x[cfg.image_key] for x in obs_deque])
                nimages = torch.from_numpy(images).to(device, dtype=torch.float32)

                if cfg.num_images > 1:
                    hand_images = np.stack([x[cfg.hand_image_key] for x in obs_deque])
                    nhand_images = torch.from_numpy(hand_images).to(device, dtype=torch.float32)

                agent_poses = np.stack([x[cfg.agent_pose_key] for x in obs_deque])
                nagent_poses = normalize_data(agent_poses, stats=stats['agent_pose'])
                # if env_settings.normalization_mask is not None and cfg.agent_pose_key in env_settings.normalization_mask:
                #     mask = env_settings.normalization_mask[cfg.agent_pose_key]
                #     if mask.shape[0] != nagent_poses.shape[1]:
                #         raise ValueError(f"Normalization mask for key 'agent_pose' has shape {mask.shape}, but data has shape {nagent_poses.shape}")
                #     nagent_poses = np.where(mask, nagent_poses, agent_poses)
                nagent_poses = torch.from_numpy(nagent_poses).to(device, dtype=torch.float32)

                if cfg.use_idleness_encoding:
                    idleness = np.stack([x['idleness'] for x in obs_deque])
                    nidleness = torch.from_numpy(idleness).to(device, dtype=torch.float32)
                    nidleness = nidleness.unsqueeze(1)

                if cfg.use_direct_idleness_conditioning:
                    direct_idleness = np.stack([x['direct_idleness'] for x in obs_deque])
                    ndirect_idleness = torch.from_numpy(direct_idleness).to(
                        device, dtype=torch.float32
                    ).unsqueeze(1)

                if cfg.use_saturating_progress_encoding:
                    saturating_progress_encoding = np.stack([x['saturating_progress_encoding'] for x in obs_deque])
                    nsaturating_progress_encoding = torch.from_numpy(saturating_progress_encoding).to(device, dtype=torch.float32)
                    nsaturating_progress_encoding = nsaturating_progress_encoding.unsqueeze(1)

                if cfg.use_sinusoidal_progress_encoding:
                    sinusoidal_progress_encoding = np.stack([x['sinusoidal_progress_encoding'] for x in obs_deque])
                    nsinusoidal_progress_encoding = torch.from_numpy(sinusoidal_progress_encoding).to(device, dtype=torch.float32) # shape (obs_horizon, 1, d_model)
                    nsinusoidal_progress_encoding = nsinusoidal_progress_encoding.squeeze(1) # new shape: (obs_horizon * d_model, 1)

                if cfg.use_agent_velocity_conditioning:
                    velocities = np.stack([x['agent_velocity'] for x in obs_deque]).astype(np.float32)
                    if velocities.ndim == 1:
                        velocities = velocities[:, None]
                    nvelocities = normalize_data(velocities, stats=stats['agent_velocity'])
                    nvelocities = torch.from_numpy(nvelocities).to(device, dtype=torch.float32)

                if cfg.use_temporal_time_signal:
                    time_signal = np.stack([x['time_signal'] for x in obs_deque]).astype(np.float32)
                    if time_signal.ndim == 1:
                        time_signal = time_signal[:, None]
                    ntime_signal = torch.from_numpy(time_signal).to(device, dtype=torch.float32)


                with torch.no_grad():
                    nimages = nimages.clone().detach().to(device, dtype=torch.float32)
                    if cfg.test_blank_image:
                        nimages = torch.zeros_like(nimages)
                    image_features = ema_nets['vision_encoder'](nimages)
                    if cfg.num_images > 1:
                        if cfg.test_blank_image:
                            nhand_images = torch.zeros_like(nhand_images)
                        hand_image_features = ema_nets['vision_encoder_hand'](nhand_images)
                        image_features = torch.cat([image_features, hand_image_features], dim=-1)
                    obs_features = [image_features, nagent_poses]
                    
                    if cfg.use_idleness_encoding:
                        obs_features.append(nidleness)

                    if cfg.use_direct_idleness_conditioning:
                        obs_features.append(ndirect_idleness)

                    if cfg.use_agent_velocity_conditioning:
                        obs_features.append(nvelocities)
                    
                    if cfg.use_saturating_progress_encoding:
                        obs_features.append(nsaturating_progress_encoding)

                    if cfg.use_sinusoidal_progress_encoding:
                        obs_features.append(nsinusoidal_progress_encoding)

                    if cfg.use_temporal_time_signal:
                        obs_features.append(ntime_signal)


                    # print([x.shape for x in obs_features]) # shapes of obs features
                    obs_features = torch.cat(obs_features, dim=-1)
                    obs_cond = obs_features.unsqueeze(0).flatten(start_dim=1)
                    noisy_action = torch.randn((B, cfg.pred_horizon, action_dim), device=device)
                    naction = noisy_action

                    if cfg.policy_type == "diffusion":
                        noise_scheduler.set_timesteps(cfg.num_diffusion_iters)
                        for k in noise_scheduler.timesteps:
                            noise_pred = ema_nets['noise_pred_net'](
                                sample=naction,
                                timestep=k,
                                global_cond=obs_cond
                            )
                            naction = noise_scheduler.step(
                                model_output=noise_pred,
                                timestep=k,
                                sample=naction,
                                generator=generator,
                                # temperature=temperature_value
                            ).prev_sample
                    elif cfg.policy_type == "flow_matching":
                        dt = 1.0 / cfg.num_flow_matching_steps
                        for flow_step_idx in range(cfg.num_flow_matching_steps):
                            timestep = torch.full(
                                (B,),
                                flow_step_idx * dt,
                                device=device,
                                dtype=naction.dtype,
                            )
                            velocity = ema_nets['noise_pred_net'](
                                sample=naction,
                                timestep=timestep,
                                global_cond=obs_cond,
                            )
                            naction = naction + dt * velocity

                naction = naction.detach().to('cpu').numpy()
                naction = naction[0] # just take first batch dimension in inference. [pred_horizon (16), action_dim (2 or 3)]
                
                action_pred = unnormalize_data(naction, stats=stats['action']) 
                # if env_settings.normalization_mask is not None and 'action' in env_settings.normalization_mask:
                #     mask = np.asarray(env_settings.normalization_mask['action'], dtype=bool)
                #     if mask.shape[0] != action_pred.shape[1]:
                #         raise ValueError(f"Normalization mask for key 'action' has shape {mask.shape}, but data has shape {naction.shape}")
                #     # Broadcast (D,) -> (T, D) so mask is applied to every timestep.
                #     action_pred = np.where(mask[None, :], naction, action_pred)
                start = cfg.obs_horizon - 1
                end = start + cfg.action_horizon 
                action = action_pred[start:end,:] # [8, 2]. 3 if using time prediction


                for i in range(len(action)):
                    data = {}
                    for key in obs.keys():
                        data[key] = np.float32(obs[key]) if type(obs[key]) == float else obs[key]
                        data[key] = np.int32(obs[key]) if type(obs[key]) == int else obs[key]
                    data['reward'] = np.float32(reward)
                    data['action'] = np.float32(action[i])
                    episode.append(data)
                    
                    if env_settings.env_type == EnvType.ROBOSUITE:
                        if cfg.use_pause_induced_perturbation:
                            copy_env_state(from_env=env, to_env=env_shadow)
                            copy_env_state(from_env=env, to_env=env_joint)
                            action_relative = translate_3_dim_action_to_7d(action[i])
                            (obs_shadow, rew_shadow, done_shadow, info_shadow) = env_shadow.step(action_relative)
                            joint_pos_predicted = obs_shadow['joint_position']
                            (action_joint, pip_info) = pause_induced_perturbation.step(obs['joint_position'], joint_pos_predicted)
                            if pip_info['should_perturb']:
                                action_joint = action_joint - obs['joint_position']
                                action_joint = np.concatenate([action_joint, action[i][-1:]])
                                pip_events += 1
                                (obs_j, rew_j, done_j, info_j) = env_joint.step(action_joint)
                                if (RECONSTRUCT_PIP_IN_POSE_CONTROL_MODE := True):
                                    action_pos = obs_j['robot0_eef_pos'] - obs['robot0_eef_pos']
                                    action_rot = T.quat2axisangle(obs_j['robot0_eef_quat_site']) - T.quat2axisangle(obs['robot0_eef_quat_site'])
                                    action_reconstructed = np.concatenate([action_pos, action_rot, action[i][-1:]])
                                    (obs, reward, done, info) = env.step(action_reconstructed)
                                else:
                                    copy_env_state(from_env=env_joint, to_env=env)
                                    env.sim.forward()
                                    delta_pos = obs_j['robot0_eef_pos'] - obs['robot0_eef_pos']
                                    (obs, reward, done, info) = (obs_j, rew_j, done_j, info_j)
                            else:
                                full_action = translate_3_dim_action_to_7d(action[i])
                                (obs, reward, done, info) = env.step(full_action)
                        else:
                            full_action = translate_3_dim_action_to_7d(action[i])
                            (obs, reward, done, info) = env.step(full_action)
                    elif env_settings.env_type == EnvType.PYGAME:
                        if cfg.use_pause_induced_perturbation:
                            (action[i], pip_info) = pause_induced_perturbation.step(obs[cfg.agent_pose_key], action[i])
                            if pip_info['should_perturb']:
                                pip_events += 1
                        (obs, reward, done, info) = env.step(action[i])
                    elif env_settings.env_name == EnvName.LIFTQA_REAL:
                        if cfg.use_pause_induced_perturbation:
                            (action_pip, pip_info) = pause_induced_perturbation.step(obs[cfg.agent_pose_key], action[i])
                            action[i][:3] = action_pip[:3]
                            if pip_info['should_perturb']:
                                pip_events += 1
                                print('applying PIP')
                        (obs, reward, done, info) = env.step(action[i])
                    elif env_settings.env_name == EnvName.PIANO:
                        (obs, reward, done, info) = env.step(action[i])
                        # cv2.imshow("img", obs['full_image_raw'].astype(np.uint8))
                        # cv2.waitKey(1)
                        # time.sleep(5)

                    if cfg.use_idleness_encoding:
                    #     # compute saturating time at rest from velocity
                        idleness = idleness_encoder.step(obs['agent_velocity'])
                        obs['idleness'] = idleness

                    if cfg.use_direct_idleness_conditioning:
                        obs['direct_idleness'] = float(binary_idle_indicator(
                            np.asarray(obs['agent_velocity'], dtype=np.float32).reshape(1, -1),
                            direct_idleness_vmax,
                            cfg.direct_idleness_threshold,
                        )[0, 0])

                    if cfg.use_saturating_progress_encoding:
                        saturating_progress_encoding = progress_saturalising_encoding(t=step_idx, omega=omega)
                        obs['saturating_progress_encoding'] = saturating_progress_encoding

                    if cfg.use_sinusoidal_progress_encoding:
                        pos_enc = progress_positional_encoding_transformer(t=step_idx, d_model=cfg.sinusoidal_progress_encoding_dim) 
                        obs['sinusoidal_progress_encoding'] = pos_enc
                        pos_enc_history.append(pos_enc)

                    if cfg.use_temporal_time_signal:
                        step_value = obs.get('step_count', step_idx)
                        obs['time_signal'] = normalize_data(
                            np.array([[step_value]], dtype=np.float32),
                            stats=stats['step_count'],
                        ).reshape(1)


                    obs_deque.append(obs)
                    info_deque.append(info)
                    rewards.append(reward)
                    if cfg.save_video:
                        imgs.append(env.render(mode='rgb_array'))
                    step_idx += 1
                    pbar.update(1)
                    pbar.set_postfix(reward=reward)

                    if step_idx > cfg.max_steps:
                        done = True
                    if done:
                        break
                
                # time.sleep(2.0)

            data_dict = {key: np.stack([step[key] for step in episode]) for key in episode[0].keys()}
            replay_buffer.add_episode(data_dict, compressors='disk')
            print(f"✅ Episode saved with seed {seed}")

            episode_return = float(np.sum(rewards))
            episode_max_reward = float(np.max(rewards))
            episode_reached_step_limit = step_idx > cfg.max_steps
            if episode_reached_step_limit:
                num_episodes_reached_step_limit += 1
            episode_data = {
                "ep_length": step_idx,
                "ep_return": episode_return,
                "ep_max_reward": episode_max_reward,
                "ep_reached_limit": episode_reached_step_limit,
                "cumulative_eps_reached_limit": num_episodes_reached_step_limit,
            }
            episode_metrics.append(episode_data)

            if wandb_run is not None:
                episode_data["episode_id"] = episode_id
                wandb.log(episode_data)

            if cfg.save_video:
                video_path = os.path.join(video_folder, f"{checkpoint_name}_seed{seed}_episode{episode_id + 1}_return{episode_return:.3f}.mp4")
                vwrite(video_path, imgs)
    
    t_end = time.time()
    mean_episode_return = float(np.mean([metric["ep_return"] for metric in episode_metrics]))
    completion_within_limit_rate = (
        100.0 * (len(episode_metrics) - num_episodes_reached_step_limit) / len(episode_metrics)
    )
    print(f"Episodes completed: {len(episode_metrics)}")
    print(f"Episodes that reached the step limit: {num_episodes_reached_step_limit}")
    print(f"Completion within limit rate: {completion_within_limit_rate:.2f}%")
    print(f"Mean episode return: {mean_episode_return:.4f}")
    print(f"Inference complete, generated dataset: {inference_dataset_path}")
    best_effort_notify(
        f"Inference complete, generated dataset: {inference_dataset_path}, "
        f"mean episode return: {mean_episode_return:.4f}, time taken: {t_end - t_start:.2f} seconds"
    )
    print(f"Total inference time: {t_end - t_start:.2f} seconds")

    if env_settings.env_name == EnvName.LIFTQA_REAL:
        env.close()
    
    if cfg.use_pause_induced_perturbation:
        replay_buffer.root.attrs["num_pip_events"] = pip_events
        print(f"Total Pause-Induced Perturbation events: {pip_events}")

    if wandb_run is not None:
        wandb_run.finish()

if __name__ == '__main__':
    main()
