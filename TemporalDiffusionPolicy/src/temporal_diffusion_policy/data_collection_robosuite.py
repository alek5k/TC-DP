from dataclasses import asdict, dataclass
import os
import time
import cv2
import draccus
import numpy as np
from temporal_diffusion_policy.envs.robosuite.spacemouse_udp_client import SpaceMouse
from temporal_diffusion_policy.replay_buffer import ReplayBuffer
import robosuite
from temporal_diffusion_policy.envs.robosuite.lift_qa import LiftQA, create_env
# pip install robosuite hidapi h5py

@dataclass
class EnvironmentConfig:
    dataset_path: str = "data_new/demonstration/lift_qa_v1.zarr"
    episodes: int = 200
    seed: int = 0
    constrain_motion: bool = True  # whether to limit motion to horizontal plane
    allow_camera_control: bool = False  # whether to allow controlling the camera with the spacemouse
    control_hz: int = 10 # control frequency. also throttles env.
    realtime: bool = True  # whether to run the env in real-time (sleep to match control timestep)
    realtime_speed: float = 2.0  # speed multiplier for real-time mode. 1.0 is real-time, 1.2 is 20% faster than real-time, etc.
    max_episode_length: int = 1000  # max episode length in timesteps
    use_hard_reset: bool = False  # whether to use hard reset (closes and reopens the mujoco window)
    render: bool = True  # whether to render the environment
    debug_render_output_img: bool = False  # whether to output image retreived from environment fr debugging

@draccus.wrap()
def main(cfg: EnvironmentConfig):

    replay_buffer = ReplayBuffer.create_from_path(cfg.dataset_path, mode='w')
    
    env: LiftQA = create_env(
        render=cfg.render,
        constrain_motion=cfg.constrain_motion,
        allow_camera_control=cfg.allow_camera_control,
        control_hz=cfg.control_hz,
        max_episode_length=cfg.max_episode_length,
        seed=cfg.seed,
    )

    server_address = "192.168.112.1" if os.environ.get("WSL_DISTRO_NAME", None) is not None else "127.0.0.1"
    device = SpaceMouse(
        env=env,
        x_sensitivity=0.0 if cfg.constrain_motion else 1.0, # 0
        y_sensitivity=1.0,
        z_sensitivity=1.0,
        roll_sensitivity=0.0 if cfg.constrain_motion else 0.2,
        pitch_sensitivity=0.0 if cfg.constrain_motion else 0.2,
        yaw_sensitivity=0.0 if cfg.constrain_motion else 0.2, 
        server_address=server_address
    )

    for k,v in asdict(cfg).items():
        replay_buffer.root.attrs[k] = v

    ###############
    ## MAIN LOOP ##
    ###############
    completed_episodes = 0

    for episode in range(cfg.episodes):
        obs = env.reset()
        reward = 0.0
        done = False
        if cfg.render:
            env.render()

        episode_data_to_save = []

        # Initialize device control
        device.start_control()
        device.single_click_and_hold = False # close gripper by default

        while True:
            action_from_device = device.input2action()
            manual_reset_requested = action_from_device is None
            
            if manual_reset_requested:
                break

            dx, dy, dz, ax, ay, az = action_from_device["right_delta"]
            gripper = action_from_device["right_gripper"][0]
            env_action = np.array([dx, dy, dz, ax, ay, az, gripper])

            # save_dict = obs.copy()
            save_dict = {}
            save_dict['step_count'] = obs['step_count']
            save_dict['cur_time'] = obs['cur_time']
            save_dict['full_image'] = obs['full_image']
            save_dict['joint_velocity'] = obs['joint_velocity']
            save_dict['gripper_velocity'] = obs['gripper_velocity']
            save_dict['agent_velocity'] = obs['agent_velocity']
            save_dict['agent_pose'] = obs['agent_pose']
            save_dict['wait_time'] = obs['wait_time']
            save_dict['wait_times_each_visit'] = obs['wait_times_each_visit']
            save_dict['action'] = np.array([dy, dz, gripper], dtype=np.float32)  # action taken at this step
            save_dict['reward'] = reward  # reward received after taking action
            save_dict['done'] = done  # whether episode ended after this step
    
            if cfg.debug_render_output_img:
                import cv2
                img_flipped = obs['agentview_image'][::-1, :, :] # flip vertical
                cv2.imshow("img", cv2.cvtColor(img_flipped, cv2.COLOR_RGB2BGR))
                cv2.waitKey(1)

            start = time.perf_counter()
            obs, reward, done, info = env.step(env_action)

            if cfg.realtime:
                dt = env.control_timestep / cfg.realtime_speed
                elapsed = time.perf_counter() - start
                if elapsed < dt:
                    time.sleep(dt - elapsed)
            if cfg.render:
                env.render()
            
            # convert all numpy float64 to float32:
            for key in save_dict.keys():
                if isinstance(save_dict[key], np.ndarray):
                    if save_dict[key].dtype == np.float64:
                        save_dict[key] = save_dict[key].astype(np.float32)
                elif isinstance(save_dict[key], float):
                    save_dict[key] = np.float32(save_dict[key])

            episode_data_to_save.append(save_dict)

            if done:
                break

        # Episodes done here
        if done:
            data_dict = {key: np.stack([step[key] for step in episode_data_to_save]) for key in episode_data_to_save[0].keys()}
            replay_buffer.add_episode(data_dict, compressors='disk')
            print(f"Saved episode {completed_episodes}")
            completed_episodes += 1

        if manual_reset_requested:
            print("Manual reset requested")



if __name__ == "__main__":
    main()
