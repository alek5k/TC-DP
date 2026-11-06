from dataclasses import asdict, dataclass
import draccus
import numpy as np
from temporal_diffusion_policy.env_mappings import create_pygame_env, get_env_settings_from_env_name
from temporal_diffusion_policy.replay_buffer import ReplayBuffer
import pygame

@dataclass
class Config:
    output: str = 'data/demonstration/wait_at_goal.zarr'
    env_name: str = 'WaitAtGoal'
    render_size: int | None = None
    control_hz: int = 10
    episodes: int = 200
    seed: int = 0

@draccus.wrap()
def main(cfg: Config):
    env_settings = get_env_settings_from_env_name(cfg.env_name)
    if cfg.render_size is not None and cfg.render_size != env_settings.img_dim:
        raise ValueError(
            f"{cfg.env_name} uses img_dim={env_settings.img_dim}; "
            f"got render_size={cfg.render_size}."
        )
    cfg.render_size = env_settings.img_dim
    replay_buffer = ReplayBuffer.create_from_path(cfg.output, mode='a')

    replay_buffer.update_meta({ # log metadata for reproducibility
        "control_hz": np.array([cfg.control_hz], dtype=np.int64),
        "render_size": np.array([cfg.render_size], dtype=np.int64)
    })

    for k,v in asdict(cfg).items():
        replay_buffer.root.attrs[k] = v

    completed_eps = 0

    env = create_pygame_env(cfg.env_name, render_size=env_settings.img_dim)

    env.control_hz = cfg.control_hz
    clock = pygame.time.Clock()
    agent = env.teleop_agent()

    while completed_eps < cfg.episodes:
        episode = []
        seed = cfg.seed + replay_buffer.n_episodes
        print(f'🚀 Starting episode with seed {seed}')
        env.seed(seed)

        obs = env.reset()
        info = env._get_info()
        reward = 0.0
        done = False
        retry = False

        while not done:
            pygame.event.pump()
            act = agent.act()

            if act is not None:
                data = {}
                for key in obs.keys():
                    data[key] = np.float32(obs[key]) if type(obs[key]) == float else obs[key]
                    data[key] = np.int32(obs[key]) if type(obs[key]) == int else obs[key]
                data['reward'] = np.float32(reward)
                data['action'] = np.float32(act)

                episode.append(data)

                obs, reward, done, info = env.step(act)

            # Match the environment control rate so mouse interaction and
            # piano tones remain perceptible during live collection.
            clock.tick(cfg.control_hz)


        if not retry:
            # Convert episode list into arrays
            data_dict = {key: np.stack([step[key] for step in episode]) for key in episode[0].keys()}

            # Save to buffer
            replay_buffer.add_episode(data_dict, compressors='disk')
            completed_eps += 1
            print(f"✅ Episode saved with seed {seed}")
        else:
            print(f"🔁 Retrying seed {seed}")

if __name__ == "__main__":
    main()
