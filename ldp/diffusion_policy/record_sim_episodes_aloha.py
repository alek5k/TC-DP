import time
import os
import numpy as np
import argparse
import matplotlib.pyplot as plt
import h5py

from diffusion_policy.env.aloha.constants import PUPPET_GRIPPER_POSITION_NORMALIZE_FN, SIM_TASK_CONFIGS
from diffusion_policy.env.aloha.ee_sim_env import make_ee_sim_env
from diffusion_policy.env.aloha.sim_env import make_sim_env, BOX_POSE
from diffusion_policy.env.aloha.scripted_policy import SingleArmPickAndPlaceOriginPolicyLargeRand,PickAndTransferPolicy,PickAndPlaceSamePosPolicyLargeRand,PickAndHistoryTransferPolicyLargeRand, PickAndTransferPolicyLargeRand, InsertionPolicy, InsertionPolicyLargeRand, PickAndPlaceHistoryTransferPolicyLargeRand,PickAndPlaceOriginPolicyLargeRand, PickAndPlaceAtTwoModesPolicyLargeRand, SingleArmPickAndPlaceTwoModesPolicyLargeRand
import wandb
from diffusion_policy.env.aloha.visualize_episodes import save_videos
from diffusion_policy.env.aloha.constants import DT
import pathlib
from diffusion_policy.common.replay_buffer import ReplayBuffer
import json
import IPython
e = IPython.embed


def main(args):
    """
    Generate demonstration data in simulation.
    First rollout the policy (defined in ee space) in ee_sim_env. Obtain the joint trajectory.
    Replace the gripper joint positions with the commanded joint position.
    Replay this joint trajectory (as action sequence) in sim_env, and record all observations.
    Save this episode of data, and continue to next episode of data collection.
    """
    task_name = args['task_name']
    dataset_dir = args['dataset_dir']
    num_episodes = args['num_episodes']
    onscreen_render = args['onscreen_render']
    rand_type = args['rand_type']

    inject_noise = args["inject_noise"]
    render_cam_name = 'angle'

    wandb_run = wandb.init(project=task_name, config=args)


    if not os.path.isdir(dataset_dir):
        os.makedirs(dataset_dir, exist_ok=True)

    episode_len = SIM_TASK_CONFIGS[task_name]['episode_len']
    if not args["from_state"]:
        camera_names = SIM_TASK_CONFIGS[task_name]['camera_names']
    else:
        camera_names = []
    action_dim =  SIM_TASK_CONFIGS[task_name]['action_dim']
    qpos_dim =  SIM_TASK_CONFIGS[task_name]['state_dim']
    
    if task_name == 'sim_transfer_cube_scripted':
        policy_cls = PickAndTransferPolicyLargeRand
    elif task_name == 'sim_pickandplace_samepos_scripted':
        policy_cls = PickAndPlaceSamePosPolicyLargeRand
    elif task_name == 'sim_pickandplace_origin_scripted':
        policy_cls = PickAndPlaceOriginPolicyLargeRand
    elif task_name == 'sim_pickandplace_twomodes_scripted':
        policy_cls = PickAndPlaceAtTwoModesPolicyLargeRand
    elif task_name == 'sim_singlearm_pickandplace_twomodes_scripted':
        policy_cls = SingleArmPickAndPlaceTwoModesPolicyLargeRand
    elif task_name == 'sim_insertion_scripted':
        policy_cls = InsertionPolicyLargeRand
    elif task_name == 'sim_singlearm_pickandplace_origin_scripted':
        policy_cls = SingleArmPickAndPlaceOriginPolicyLargeRand
    elif task_name == 'sim_history_transfer_cube_scripted':
        policy_cls = PickAndHistoryTransferPolicyLargeRand
    elif task_name == 'sim_history_pickandplace_transfer_cube_scripted':
        policy_cls = PickAndPlaceHistoryTransferPolicyLargeRand
    else:
        raise NotImplementedError


    pathlib.Path(dataset_dir).mkdir(parents=True, exist_ok=True)

    env_cfg = SIM_TASK_CONFIGS[task_name]

    zarr_path = str(pathlib.Path(dataset_dir).joinpath('replay_buffer.zarr').absolute())
    replay_buffer = ReplayBuffer.create_from_path(
        zarr_path=zarr_path, mode='a')

    dataset_file = h5py.File(pathlib.Path(dataset_dir).joinpath("demos.hdf5"), 'w')
    dataset_data_group = dataset_file.create_group('data')

    dataset_data_group.attrs['env_args'] = json.dumps(env_cfg.copy())


    total_rollouts = 0
    successful_rollouts = 0

    success = []
    episode_idx = 0
    while episode_idx < num_episodes:
        image_list = []
        print(f'{episode_idx=}')
        print('Rollout out EE space scripted policy')
        # setup the environment
        env = make_ee_sim_env(task_name, args)
        ts = env.reset()
        episode = [ts]
        policy = policy_cls(inject_noise)
        # setup plotting
        if onscreen_render:
            ax = plt.subplot()
            plt_img = ax.imshow(ts.observation['images'][render_cam_name])
            plt.ion()
        for step in range(episode_len):
            action = policy(ts)
            ts = env.step(action)
            episode.append(ts)
            obs = ts.observation

            if onscreen_render:
                plt_img.set_data(ts.observation['images'][render_cam_name])
                plt.pause(0.002)
        plt.close()

        episode_return = np.sum([ts.reward for ts in episode[1:]])
        episode_max_reward = np.max([ts.reward for ts in episode[1:]])
        if episode[-1].reward  == env.task.max_reward:
            print(f"{episode_idx=} Successful, {episode_return=}")
        else:
            print(f"{episode_idx=} Failed")

        joint_traj = [ts.observation['qpos'] for ts in episode]
        # replace gripper pose with gripper control
        gripper_ctrl_traj = [ts.observation['gripper_ctrl'] for ts in episode]
        if "singlearm" in task_name:
            for joint, ctrl in zip(joint_traj, gripper_ctrl_traj):
                right_ctrl = PUPPET_GRIPPER_POSITION_NORMALIZE_FN(ctrl[0])
                joint[6] = right_ctrl
        else:
            for joint, ctrl in zip(joint_traj, gripper_ctrl_traj):
                left_ctrl = PUPPET_GRIPPER_POSITION_NORMALIZE_FN(ctrl[0])
                right_ctrl = PUPPET_GRIPPER_POSITION_NORMALIZE_FN(ctrl[2])
                joint[6] = left_ctrl
                joint[6+7] = right_ctrl

        subtask_info = episode[0].observation['env_state'].copy() # box pose at step 0

        # clear unused variables
        del env
        del episode
        del policy

        # setup the environment
        print('Replaying joint commands')
        env = make_sim_env(task_name)
        BOX_POSE[0] = subtask_info # make sure the sim_env has the same object configurations as ee_sim_env
        ts = env.reset()

        episode_replay = [ts]

        # setup plotting
        if onscreen_render:
            ax = plt.subplot()
            plt_img = ax.imshow(ts.observation['images'][render_cam_name])
            plt.ion()
            
        for t in range(len(joint_traj)): # note: this will increase episode length by 1
            action = joint_traj[t]
            ts = env.step(action)
            episode_replay.append(ts)
            if onscreen_render:
                plt_img.set_data(ts.observation['images'][render_cam_name])
                plt.pause(0.02)

        episode_return = np.sum([ts.reward for ts in episode_replay[1:]])
        episode_max_reward = np.max([ts.reward for ts in episode_replay[1:]])
        if episode_replay[-1].reward == env.task.max_reward:
            success.append(1)
            print(f"{episode_idx=} Successful, {episode_return=}")
        else:
            success.append(0)
            print(f"{episode_idx=} Failed")

        
        plt.close()

        """
        For each timestep:
        observations
        - images
            - each_cam_name     (480, 640, 3) 'uint8'
        - qpos                  (14,)         'float64'
        - qvel                  (14,)         'float64'

        action                  (14,)         'float64'
        """
        # Only store successful episodes  # TODO
        # if success[-1] == 0:
        #     continue
        
        trajectory = {"obs": [],
        "actions": [],}

        # because the replaying, there will be eps_len + 1 actions and eps_len + 2 timesteps
        # truncate here to be consistent
        joint_traj = joint_traj[:-1]
        episode_replay = episode_replay[:-1]

        # len(joint_traj) i.e. actions: max_timesteps
        # len(episode_replay) i.e. time steps: max_timesteps + 1
        max_timesteps = len(joint_traj)
        while joint_traj:
            action = joint_traj.pop(0)
            ts = episode_replay.pop(0)
            obs_dict = ts.observation


            for key in obs_dict["images"].keys():
                obs_dict[key] = obs_dict["images"][key]

            if 'images' in ts.observation:
                image_list.append(ts.observation['images'])
            else:
                image_list.append({'main': ts.observation['image']})


            del obs_dict["images"]

            # import IPython
            # IPython.embed()
            # camera_keys = ["top", "left_wrist", "right_wrist"]
            # for key in camera_keys:
            #     obs_dict[key] =  (obs_dict[key]*255).astype(np.uint8).transpose((1,2,0))

            trajectory["obs"].append(obs_dict)
            trajectory["actions"].append(action)
        
        ep_group = dataset_data_group.create_group(f'demo_{episode_idx}')
        obs_group = ep_group.create_group('obs')




        for obs_kwrd in trajectory['obs'][0].keys():
            obs_kwrd = str(obs_kwrd)
            obs_array = np.stack([od[obs_kwrd] for od in trajectory['obs']], axis=0)
            obs_group.create_dataset(obs_kwrd, data=obs_array)
        action_array = np.stack([a for a in trajectory['actions']], axis=0)
        ep_group.create_dataset('actions', data=action_array)
        ep_group.attrs['scripted_policy_type'] = str(policy_cls)
        
        if not args["from_state"] and not args["no_save_video"] and episode_idx < 5:
            save_videos(image_list, DT, video_path=os.path.join(dataset_dir, f'video{episode_idx}.mp4'), wandb_run=wandb_run)
        episode_idx +=1
        
    
    # TODO the images need to be saved in format
    dataset_data_group.attrs['data collection'] = f"{np.sum(success)} of {len(success)} total rollouts successful"
    dataset_file.close()
    print(f'Saved to {dataset_dir}')
    print(f'Success: {np.sum(success)} / {len(success)}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--task_name', action='store', type=str, help='task_name', required=True)
    parser.add_argument('--dataset_dir', action='store', type=str, help='dataset saving dir', required=True)
    parser.add_argument('--num_episodes', action='store', type=int, help='num_episodes', required=False)
    parser.add_argument('--onscreen_render', action='store_true')
    parser.add_argument('--from_state', action='store_true')
    parser.add_argument('--inject_noise', action='store_true')
    parser.add_argument('--no_save_video', action='store_true')
    parser.add_argument('--rand_type', action='store', type=str, help='rand_type', required=False, default="")
    
    main(vars(parser.parse_args()))
