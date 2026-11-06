from dataclasses import asdict, dataclass
from datetime import datetime
import os
from pathlib import Path
import draccus
import numpy as np
from temporal_diffusion_policy.env_mappings import create_pygame_env, get_env_settings_from_env_name
from temporal_diffusion_policy.replay_buffer import ReplayBuffer
# from skvideo.io import vwrite

import pygame


@dataclass
class Config:
    source: str = 'data/demonstration/archive/avoid_obstacles_v2.zarr'
    output: str = 'data/demonstration/avoid_obstacles_v3.zarr'
    regeneration_comment: str = f'Fix human error in waittime' # TODO: ADD COMMENT HERE
    render: bool = False
    env_name: str = 'AvoidObstacleEnv'
    render_size: int | None = None
    control_hz: int = 10
    continue_on_failure: bool = False
    seed: int = 0 # seed should be 0 if regenerating from a demonstration dataset.
    realtime: bool = False  # Throttle to control_hz only when visualising regeneration.
    replacement_source: str | None = None  # Optional source for one repaired episode.
    replacement_episode_index: int | None = None
    repeat_final_action_until_done: bool = False
    random_initial_action_replay: bool = False

@draccus.wrap()
def main(cfg: Config):
    env_settings = get_env_settings_from_env_name(cfg.env_name)
    if cfg.render_size is not None and cfg.render_size != env_settings.img_dim:
        raise ValueError(
            f"{cfg.env_name} uses img_dim={env_settings.img_dim}; "
            f"got render_size={cfg.render_size}."
        )
    cfg.render_size = env_settings.img_dim
    source_buffer = ReplayBuffer.create_from_path(cfg.source, 'r')
    source_num_episodes = source_buffer.n_episodes

    replay_buffer = ReplayBuffer.create_from_path(cfg.output, mode='w')
    replay_buffer.update_meta({ # log metadata for reproducibility
        "control_hz": np.array([cfg.control_hz], dtype=np.int64),
        "render_size": np.array([cfg.render_size], dtype=np.int64)
    })
    for k,v in asdict(cfg).items():
        replay_buffer.root.attrs[k] = v

    replay_buffer.root.attrs['comment'] = cfg.regeneration_comment
    completed_eps = 0
    
    if not cfg.render:
        os.environ["SDL_VIDEODRIVER"] = "dummy" # prevents rendering
    env = create_pygame_env(cfg.env_name, render_size=env_settings.img_dim)
    env.control_hz = cfg.control_hz
    clock = pygame.time.Clock()

    while completed_eps < source_num_episodes:
        
        source_episode = source_buffer.get_episode(completed_eps)
        
        # Optional dataset surgery: replace exactly one episode from another
        # source.  Keep this opt-in so generic regeneration is source-faithful.
        if cfg.replacement_episode_index == completed_eps:
            if cfg.replacement_source is None:
                raise ValueError(
                    "replacement_source is required when replacement_episode_index is set."
                )
            source2_buffer = ReplayBuffer.create_from_path(
                cfg.replacement_source, 'r'
            )
            source_episode = source2_buffer.get_episode(completed_eps)

        episode = []
        seed = cfg.seed + replay_buffer.n_episodes
        print(f'🚀 Starting episode with seed {seed}')
        env.seed(seed)

        obs = env.reset()
        info = env._get_info()
        reward = 0.0
        done = False
        retry = False

        step_num = 0
        source_actions = source_episode['action']
        if len(source_actions) == 0:
            raise ValueError(f"Source episode {completed_eps} has no actions.")
        initial_replay_steps = (
            int(np.random.default_rng(seed).integers(0, 21))
            if cfg.random_initial_action_replay
            else 0
        )
        if cfg.random_initial_action_replay:
            print(
                f"Replaying the first action for {initial_replay_steps} "
                "initial timestep(s)."
            )
        # imgs = [env.render(mode='rgb_array')]

        while not done:
            pygame.event.pump()
            
            if step_num < initial_replay_steps:
                act = source_actions[0]
            elif step_num - initial_replay_steps < len(source_actions):
                act = source_actions[step_num - initial_replay_steps]
            elif cfg.repeat_final_action_until_done:
                act = source_actions[-1]
            else:
                break
            step_num += 1
            
            if act is not None:
                data = {}
                for key in obs.keys():
                    data[key] = np.float32(obs[key]) if type(obs[key]) == float else obs[key]
                    data[key] = np.int32(obs[key]) if type(obs[key]) == int else obs[key]
                data['reward'] = np.float32(reward)
                data['action'] = np.float32(act)

                episode.append(data)

                obs, reward, done, info = env.step(act)

            # if info['n_contacts'] > 0:
            #     if cfg.continue_on_failure:
            #         print("⚠️ Collision detected. Continuing to next episode...")
            #         done = True
            #         break
            #     else:
            #         retry = True
            #         print("❌ Collision detected. Retrying episode...")
            #         break
            
            # imgs.append(env.render(mode='rgb_array'))
            if cfg.realtime:
                clock.tick(cfg.control_hz)

        if not retry:
            # Convert episode list into arrays
            data_dict = {key: np.stack([step[key] for step in episode]) for key in episode[0].keys()}

            # Save to buffer
            replay_buffer.add_episode(data_dict, compressors='disk')
            completed_eps += 1
            print(f"✅ Episode saved with seed {seed}")
            # vwrite(f"test_regenerate_{completed_eps}.mp4", imgs)
        else:
            print(f"🔁 Retrying seed {seed}")
            # vwrite(f"test_regenerate_{completed_eps}.mp4", imgs)

if __name__ == "__main__":

    main()
