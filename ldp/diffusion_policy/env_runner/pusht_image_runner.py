from typing import Optional
import wandb
from collections import deque
import numpy as np
import os
import torch
import torchvision
import collections
import pathlib
import tqdm
import dill
import math
import wandb.sdk.data_types.video as wv
from diffusion_policy.common.noise_sampler import NoiseGenerator
from diffusion_policy.env.pusht.pusht_image_env import PushTImageEnv
from diffusion_policy.gym_util.async_vector_env import AsyncVectorEnv
# from diffusion_policy.gym_util.sync_vector_env import SyncVectorEnv
from diffusion_policy.gym_util.multistep_wrapper import MultiStepWrapper
from diffusion_policy.gym_util.video_recording_wrapper import VideoRecordingWrapper, VideoRecorder

from diffusion_policy.policy.base_image_policy import BaseImagePolicy
from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.env_runner.base_image_runner import BaseImageRunner

# TODO: backwards-facing or just in general? could frame as delayed response
def motion_blur(x,severity):
    clip=np.asarray(x)
    blur_clip=[]
    for i in range(severity,clip.shape[1]):
        blur_image=np.sum(clip[:, i-severity:i+1],axis=1,dtype=float)/(severity+1.0)
        blur_clip.append(np.array(blur_image,dtype=float))
    return np.stack(blur_clip, axis=1)

def occlude(x, severity, period, start):
    clip = np.asarray(x)
    for i in range(clip.shape[1]):
        # guarantee at least some good frames at start
        if (start + i + period - 3) % (period) < severity:
            clip[:, i] = np.zeros_like(clip[:, i])

    return clip

# higher severity is better
def frame_corruption(x,severity):
    import torchvision.transforms as T

    # Define a color jitter transformation
    color_jitter = T.ColorJitter(
        brightness=0.4,  # Adjust brightness by ±30%
        contrast=0.3,    # Adjust contrast by ±30%
        saturation=0.3,  # Adjust saturation by ±30%
        hue=0.2          # Adjust hue by ±20%
    )

    clip = np.asarray(x)
    for i in range(clip.shape[1]):
        selector = np.random.randint(severity)
        if selector == 0:
            clip[:, i] = color_jitter(torch.from_numpy(clip[:, i]).float() / 255.0).numpy()*255
    return clip

"""
Perturbation controls:

 - fps input (for base environment)
 - observation-facing:
    - "back_motion_blur": backwards-facing motion blur for [strength] timesteps
    - "fps_change": multiply fps by [strength] factor
    - "act_delay": delay actions by [strength] timesteps (e.g. shift observations backwards)
    - "random_corrupt": randomly corrupt [strength] proportion of frames
 - action-facing:
    - "act_noise": add noise to action, scaled by [strength] factor of the size of action (see BID)
    - "act_slowdown": only execute one in every [strength] actions we get 
"""

VALID_PERTURBATIONS = ["back_motion_blur", "fps_change", "act_delay", "random_corrupt", "act_noise", "act_slowdown"]
class PushTImageRunner(BaseImageRunner):
    def __init__(self,
            output_dir,
            n_train=10,
            n_train_vis=3,
            train_start_seed=0,
            n_test=22,
            n_test_vis=6,
            legacy_test=False,
            test_start_seed=10000,
            max_steps=200,
            n_obs_steps=8,
            n_action_steps=8,
            fps=10,
            crf=22,
            render_size=96,
            past_action=False,
            tqdm_interval_sec=5.0,
            n_envs=None,
            use_past_actions=False, # TODO: fix up
            perturbations: Optional[dict[str, float]]=None,
        ):
        super().__init__(output_dir)
        if n_envs is None:
            n_envs = n_train + n_test

        self.use_past_actions = use_past_actions
        # NOTE: necessary preprocessing of inner environment for any enhancements
        if perturbations:
            if "fps_change" in perturbations:
                fps = int(perturbations["fps_change"] * fps) # TODO: WARNING FPS might not work well right now
            if "back_motion_blur" in perturbations:
                n_obs_steps += int(perturbations["back_motion_blur"]) 
            if "act_delay" in perturbations:
                n_obs_steps += int(perturbations["act_delay"])
            if "act_noise" in perturbations:
                self.disruptor = NoiseGenerator(perturbations["act_noise"])
            print(perturbations)
            
        self.perturbations = perturbations
        block_move = 0
        if perturbations and "move_block" in perturbations:
            block_move = perturbations["move_block"]

        steps_per_render = max(10 // fps, 1)
        def env_fn():
            return MultiStepWrapper(
                VideoRecordingWrapper(
                    PushTImageEnv(
                        legacy=legacy_test,
                        render_size=render_size,
                        perturb=block_move,
                    ),
                    video_recoder=VideoRecorder.create_h264(
                        fps=fps,
                        codec='h264',
                        input_pix_fmt='rgb24',
                        crf=crf,
                        thread_type='FRAME',
                        thread_count=1
                    ),
                    file_path=None,
                    steps_per_render=steps_per_render
                ),
                n_obs_steps=n_obs_steps,
                n_action_steps=n_action_steps,
                max_episode_steps=max_steps
            )

        env_fns = [env_fn] * n_envs
        env_seeds = list()
        env_prefixs = list()
        env_init_fn_dills = list()
        # train
        for i in range(n_train):
            seed = train_start_seed + i
            enable_render = i < n_train_vis

            def init_fn(env, seed=seed, enable_render=enable_render):
                # setup rendering
                # video_wrapper
                assert isinstance(env.env, VideoRecordingWrapper)
                env.env.video_recoder.stop()
                env.env.file_path = None
                if enable_render:
                    filename = pathlib.Path(output_dir).joinpath(
                        'media', wv.util.generate_id() + ".mp4")
                    filename.parent.mkdir(parents=False, exist_ok=True)
                    filename = str(filename)
                    env.env.file_path = filename

                # set seed
                assert isinstance(env, MultiStepWrapper)
                env.seed(seed)
            
            env_seeds.append(seed)
            env_prefixs.append('train/')
            env_init_fn_dills.append(dill.dumps(init_fn))

        # test
        for i in range(n_test):
            seed = test_start_seed + i
            enable_render = i < n_test_vis

            def init_fn(env, seed=seed, enable_render=enable_render):
                # setup rendering
                # video_wrapper
                assert isinstance(env.env, VideoRecordingWrapper)
                env.env.video_recoder.stop()
                env.env.file_path = None
                if enable_render:
                    filename = pathlib.Path(output_dir).joinpath(
                        'media', wv.util.generate_id() + ".mp4")
                    filename.parent.mkdir(parents=False, exist_ok=True)
                    filename = str(filename)
                    env.env.file_path = filename

                # set seed
                assert isinstance(env, MultiStepWrapper)
                env.seed(seed)
            
            env_seeds.append(seed)
            env_prefixs.append('test/')
            env_init_fn_dills.append(dill.dumps(init_fn))

        env = AsyncVectorEnv(env_fns)

        # test env
        # env.reset(seed=env_seeds)
        # x = env.step(env.action_space.sample())
        # imgs = env.call('render')
        # import pdb; pdb.set_trace()

        self.env = env
        self.env_fns = env_fns
        self.env_seeds = env_seeds
        self.env_prefixs = env_prefixs
        self.env_init_fn_dills = env_init_fn_dills
        self.fps = fps
        self.crf = crf
        self.n_obs_steps = n_obs_steps
        self.n_action_steps = n_action_steps
        self.past_action = past_action
        self.max_steps = max_steps
        self.tqdm_interval_sec = tqdm_interval_sec
    
    def run(self, policy: BaseImagePolicy):
        device = policy.device
        dtype = policy.dtype
        env = self.env

        # plan for rollout
        n_envs = len(self.env_fns)
        n_inits = len(self.env_init_fn_dills)
        n_chunks = math.ceil(n_inits / n_envs)

        # allocate data
        all_video_paths = [None] * n_inits
        all_rewards = [None] * n_inits

        for chunk_idx in range(n_chunks):
            start = chunk_idx * n_envs
            end = min(n_inits, start + n_envs)
            this_global_slice = slice(start, end)
            this_n_active_envs = end - start
            this_local_slice = slice(0,this_n_active_envs)
            
            this_init_fns = self.env_init_fn_dills[this_global_slice]
            n_diff = n_envs - len(this_init_fns)
            if n_diff > 0:
                this_init_fns.extend([self.env_init_fn_dills[0]]*n_diff)
            assert len(this_init_fns) == n_envs

            # init envs
            env.call_each('run_dill_function', 
                args_list=[(x,) for x in this_init_fns])

            # start rollout
            obs = env.reset()
            past_action = None
            policy.reset()

            pbar = tqdm.tqdm(total=self.max_steps, desc=f"Eval PushtImageRunner {chunk_idx+1}/{n_chunks}", 
                leave=False, mininterval=self.tqdm_interval_sec)
            done = False
            batch_saved = 0
            curr_step = 0
            act_hist = deque(maxlen=self.n_obs_steps - 1)
            print(f"PAST HISTORY QUEUE LEN: {self.n_obs_steps - 1}")
            while not done:
                # create obs dict
                np_obs_dict = dict(obs)
                obs_len = np_obs_dict[list(np_obs_dict.keys())[0]].shape[1]
                if batch_saved == obs_len or batch_saved == obs_len + 1:
                    with open(os.path.join(self.output_dir, f"pre-processing-data-{batch_saved}.pkl"), "wb") as f:
                        dill.dump(np_obs_dict, f)

                if self.past_action and (past_action is not None):
                    # TODO: not tested
                    np_obs_dict['past_action'] = past_action[
                        :,-(self.n_obs_steps-1):].astype(np.float32)
                
                # NOTE: PERTURBATIONS needing code are "back_motion_blur", "act_delay", 
                # "random_corrupt", "act_noise", "act_slowdown"
                if self.perturbations:
                    obs_len = np_obs_dict[list(np_obs_dict.keys())[0]].shape[1]
                    if "act_delay" in self.perturbations:
                        for key in np_obs_dict:
                            np_obs_dict[key] = np_obs_dict[key][:, :obs_len - int(self.perturbations["act_delay"])]
                    
                    if "back_motion_blur" in self.perturbations:
                        severity = int(self.perturbations["back_motion_blur"])
                        for key in np_obs_dict:
                            if key == "image":
                                np_obs_dict[key] = motion_blur(np_obs_dict[key], severity)
                            else:
                                np_obs_dict[key] = np_obs_dict[key][:, severity:obs_len]

                    if "occlude" in self.perturbations:
                        severity = self.perturbations["occlude"]
                        assert "period" in self.perturbations
                        period = self.perturbations["period"]
                        if "image" in np_obs_dict:
                            np_obs_dict["image"] = occlude(np_obs_dict["image"], severity, period, curr_step)
                        else:
                            print("WARNING: occlude set but no RGB images -- shouldn't happen")

                    if "random_corrupt" in self.perturbations:
                        severity = self.perturbations["random_corrupt"]
                        if "image" in np_obs_dict:
                            np_obs_dict["image"] = frame_corruption(np_obs_dict["image"], severity)
                        else:
                            print("WARNING: random_corrupt set but no RGB images -- shouldn't happen")
                
                if batch_saved == obs_len or batch_saved == obs_len + 1:
                    with open(os.path.join(self.output_dir, f"post-processing-data-{batch_saved}.pkl"), "wb") as f:
                        dill.dump(np_obs_dict, f)
                    print("saved pickles!")
                
                batch_saved += 1
                # device transfer
                obs_dict = dict_apply(np_obs_dict, 
                    lambda x: torch.from_numpy(x).to(
                        device=device))
                
                # TODO: add a key for past rolled out actions
                if len(act_hist) == obs_len - 1 and self.use_past_actions:
                    past_action = np.concatenate(act_hist, axis=1)
                    past_action = torch.Tensor(past_action).to(device)
                else:
                    past_action=None

                # run policy
                with torch.no_grad():
                    action_dict = policy.predict_action(obs_dict, past_action)

                # device_transfer
                np_action_dict = dict_apply(action_dict,
                    lambda x: x.detach().to('cpu').numpy())

                action = np_action_dict['action']

                if self.perturbations and "act_noise" in self.perturbations:
                    noise_cum = self.disruptor.step(action)
                    action += noise_cum

                # step env
                if curr_step == 0:
                    print(action.shape)
                obs, reward, done, info = env.step(action)
                curr_step += 1
                act_hist.append(action)

                # continue for more steps but keep observation correct
                if self.perturbations and "act_slowdown" in self.perturbations:
                    for i in range(int(self.perturbations["act_slowdown"]) - 1):
                        obs, reward, done, info = env.step(action)
                        curr_step += 1
                        act_hist.append(action)

                done = np.all(done)
                past_action = action

                # update pbar
                pbar.update(action.shape[1])
            pbar.close()

            all_video_paths[this_global_slice] = env.render()[this_local_slice]
            all_rewards[this_global_slice] = env.call('get_attr', 'reward')[this_local_slice]
        # clear out video buffer
        _ = env.reset()

        # log
        max_rewards = collections.defaultdict(list)
        log_data = dict()
        # results reported in the paper are generated using the commented out line below
        # which will only report and average metrics from first n_envs initial condition and seeds
        # fortunately this won't invalidate our conclusion since
        # 1. This bug only affects the variance of metrics, not their mean
        # 2. All baseline methods are evaluated using the same code
        # to completely reproduce reported numbers, uncomment this line:
        # for i in range(len(self.env_fns)):
        # and comment out this line
        for i in range(n_inits):
            seed = self.env_seeds[i]
            prefix = self.env_prefixs[i]
            max_reward = np.max(all_rewards[i])
            max_rewards[prefix].append(max_reward)
            log_data[prefix+f'sim_max_reward_{seed}'] = max_reward

            # visualize sim
            video_path = all_video_paths[i]
            if video_path is not None:
                sim_video = wandb.Video(video_path)
                log_data[prefix+f'sim_video_{seed}'] = sim_video

        # log aggregate metrics
        for prefix, value in max_rewards.items():
            name = prefix+'mean_score'
            value = np.mean(value)
            log_data[name] = value

        return log_data
