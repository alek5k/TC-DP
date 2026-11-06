from dataclasses import asdict, dataclass, field
import time

import draccus
import numpy as np

from diffusion_policy_contrib.common.replay_buffer import ReplayBuffer
from diffusion_policy_contrib.real_world.keystroke_counter import KeystrokeCounter
from temporal_diffusion_policy.envs.realworld.liftqa_real import LiftQARealConfig, LiftQARealEnvironment
from pynput.keyboard import Key, KeyCode


@dataclass
class EnvironmentConfig:
    dataset_path: str = "data_new/demonstration/lift_qa_real.zarr"
    episodes: int = 200
    max_episode_length: int = 1000
    env: LiftQARealConfig = field(default_factory=LiftQARealConfig)
    
    keypress_delta_x: float = 0.03
    keypress_delta_y: float = 0.03
    keypress_delta_z: float = 0.03

def _to_serializable(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {k: _to_serializable(v) for k, v in value.items()}
    return value


def run_data_collection(cfg: EnvironmentConfig) -> None:
    replay_buffer = ReplayBuffer.create_from_path(cfg.dataset_path, mode="w")

    env_cfg = cfg.env
    env: LiftQARealEnvironment = LiftQARealEnvironment(cfg=env_cfg)
    env._gripper_set_position(env_cfg.robotiq_initial_position)
    gripper_position_target = float(env_cfg.robotiq_initial_position)

    for k, v in asdict(cfg).items():
        replay_buffer.root.attrs[k] = _to_serializable(v)

    completed_episodes = 0

    try:
        key_counter = KeystrokeCounter()
        key_counter.__enter__()

        for episode_id in range(cfg.episodes):
            print(f"Started episode {episode_id}")
            obs = env.reset()
            reward = 0.0
            done = False
            info = {}
            episode_data_to_save = []

            key_counter.clear()
            prev_ctrl_count = key_counter[Key.ctrl_l]

            while True:

                target_x = float(obs["current_tcp"][0])
                target_y = float(obs["current_tcp"][1])
                target_z = float(obs["current_tcp"][2])

                x_dir = int(key_counter.is_pressed(Key.right)) - int(key_counter.is_pressed(Key.left))
                y_dir = int(key_counter.is_pressed(Key.up)) - int(key_counter.is_pressed(Key.down))
                z_dir = int(key_counter.is_pressed(Key.space)) - int(key_counter.is_pressed(KeyCode(char="c")))
                x_delta = cfg.keypress_delta_x * x_dir
                y_delta = cfg.keypress_delta_y * y_dir
                z_delta = cfg.keypress_delta_z * z_dir

                curr_ctrl_count = key_counter[Key.ctrl_l]
                delta_ctrl = max(0, curr_ctrl_count - prev_ctrl_count)
                prev_ctrl_count = curr_ctrl_count
                toggle_gripper = (delta_ctrl % 2) == 1

                target_x += x_delta
                target_y += y_delta
                target_z += z_delta

                if toggle_gripper:
                    # open is 0, closed is 1
                    midpoint = 0.5 * (env_cfg.robotiq_open_position + env_cfg.robotiq_closed_position)
                    if gripper_position_target > midpoint: # ie if it was previously closed, open it.
                        gripper_position_target = env_cfg.robotiq_open_position
                    else:
                        gripper_position_target = env_cfg.robotiq_closed_position

                action = np.array([target_x, target_y, target_z, gripper_position_target], dtype=np.float32)
                
                save_dict = obs.copy()
                save_dict["action"] = action
                save_dict["reward"] = reward
                save_dict["done"] = done

                obs, reward, done, info = env.step(action)

                for key, value in list(save_dict.items()):
                    if isinstance(value, np.ndarray) and value.dtype == np.float64:
                        save_dict[key] = value.astype(np.float32)
                    elif isinstance(value, float):
                        save_dict[key] = np.float32(value)

                episode_data_to_save.append(save_dict)

                if done or len(episode_data_to_save) >= cfg.max_episode_length:
                    break

            if done and len(episode_data_to_save) > 0:
                data_dict = {key: np.stack([step[key] for step in episode_data_to_save]) for key in episode_data_to_save[0].keys()}
                replay_buffer.add_episode(data_dict, compressors="disk")
                print(f"Saved episode {completed_episodes} -> {cfg.dataset_path}")
                completed_episodes += 1
                time.sleep(1.0)

    except KeyboardInterrupt:
        print("User cancelled data collection")
    finally:
        if "key_counter" in locals() and key_counter is not None:
            key_counter.__exit__(None, None, None)
        env.close()


@draccus.wrap()
def main(cfg: EnvironmentConfig):
    print(f"Data collection active: will save to dataset at {cfg.dataset_path}")
    run_data_collection(cfg)


if __name__ == "__main__":
    main()
