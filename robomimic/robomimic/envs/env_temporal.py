"""Native robomimic adapter for WaitAtGoal and LiftQA.

The wrapped environments expose float CHW images, while robomimic's image
policies consume uint8 HWC observations and process them internally. This
adapter performs that representation conversion and preserves the tasks'
native success condition: final completion, rather than robosuite's generic
object-lift predicate.
"""
from copy import deepcopy

import numpy as np

import robomimic.envs.env_base as EB


class EnvTemporal(EB.EnvBase):
    """Robomimic environment wrapper for the TemporalDiffusionPolicy tasks."""

    _SUPPORTED_ENVS = {"WaitAtGoal", "LiftQA"}

    def __init__(
        self,
        env_name,
        render=False,
        render_offscreen=False,
        use_image_obs=False,
        use_depth_obs=False,
        lang=None,
        render_size=144,
        control_hz=10,
        constrain_motion=True,
        max_episode_length=1000,
        seed=0,
        **kwargs,
    ):
        if env_name not in self._SUPPORTED_ENVS:
            raise ValueError(f"Unsupported temporal environment {env_name!r}")
        if use_depth_obs:
            raise ValueError("Temporal environments do not provide depth observations")
        if kwargs:
            raise TypeError(f"Unsupported temporal environment kwargs: {sorted(kwargs)}")

        self._env_name = env_name
        self._render_size = int(render_size)
        self._control_hz = int(control_hz)
        self._constrain_motion = bool(constrain_motion)
        self._max_episode_length = int(max_episode_length)
        self._render = bool(render)
        self._render_offscreen = bool(render_offscreen)
        self._seed = int(seed)
        self._episode_index = 0
        self._init_kwargs = {
            "render_size": self._render_size,
            "control_hz": self._control_hz,
            "constrain_motion": self._constrain_motion,
            "max_episode_length": self._max_episode_length,
            "seed": self._seed,
        }
        self._current_raw_obs = None
        self._current_obs = None
        self._current_reward = 0.0
        self._current_done = False
        self.env = self._make_env()

    def _make_env(self):
        if self._env_name == "WaitAtGoal":
            try:
                from robomimic.envs.temporal.waitatgoal import WaitAtGoal
            except ImportError as error:
                raise ImportError(
                    "WaitAtGoal requires the optional temporal environment dependencies. "
                    "Install them with: pip install -e '.[temporal-envs]'"
                ) from error
            return WaitAtGoal(render_size=self._render_size)

        try:
            from robomimic.envs.temporal.liftqa import create_env
        except ImportError as error:
            raise ImportError(
                "LiftQA requires the optional temporal environment dependencies. "
                "Install them with: pip install -e '.[temporal-envs]'"
            ) from error
        return create_env(
            render=self._render,
            constrain_motion=self._constrain_motion,
            control_hz=self._control_hz,
            max_episode_length=self._max_episode_length,
            seed=self._seed,
            camera_height_width=(self._render_size, self._render_size),
        )

    @staticmethod
    def _image_to_uint8_hwc(image):
        image = np.asarray(image)
        if image.ndim != 3:
            raise ValueError(f"Expected an image with three dimensions, got {image.shape}")
        if image.shape[0] == 3:
            image = np.moveaxis(image, 0, -1)
        if image.shape[-1] != 3:
            raise ValueError(f"Expected an RGB image, got {image.shape}")
        if image.dtype != np.uint8:
            image = np.rint(np.clip(image, 0.0, 1.0) * 255.0).astype(np.uint8)
        return np.ascontiguousarray(image)

    def _convert_observation(self, raw_obs):
        # Preserve all native fields for rollout diagnostics. Policies still
        # consume only the keys listed in their saved observation config.
        observation = {
            key: np.asarray(value).copy()
            for key, value in raw_obs.items()
        }
        observation["image"] = self._image_to_uint8_hwc(raw_obs["full_image"])
        observation["agent_pose"] = np.asarray(raw_obs["agent_pose"], dtype=np.float32).copy()
        observation["agent_velocity"] = np.asarray(raw_obs["agent_velocity"], dtype=np.float32).reshape(1)
        return observation

    def _set_current_observation(self, raw_obs):
        self._current_raw_obs = raw_obs
        self._current_obs = self._convert_observation(raw_obs)
        return self._current_obs

    def step(self, action):
        action = np.asarray(action, dtype=np.float32).copy()
        if self._env_name == "LiftQA":
            if action.shape != (3,):
                raise ValueError(f"LiftQA expects a 3-D policy action, got {action.shape}")
            from robomimic.envs.temporal.liftqa import translate_3_dim_action_to_7d
            action = translate_3_dim_action_to_7d(action)
        elif action.shape != (2,):
            raise ValueError(f"WaitAtGoal expects a 2-D policy action, got {action.shape}")

        raw_obs, reward, done, info = self.env.step(action)
        self._current_reward = float(reward)
        self._current_done = bool(done)
        info = dict(info)
        info["is_success"] = self.is_success()
        return self._set_current_observation(raw_obs), self._current_reward, self._current_done, info

    def reset(self):
        # WaitAtGoal has a seeded random initial pose. Advance the seed between
        # rollout episodes so training-time evaluation is not 50 copies of one
        # reset, while retaining a deterministic sequence.
        if self._env_name == "WaitAtGoal":
            self.env.seed(self._seed + self._episode_index)
        self._episode_index += 1
        self._current_reward = 0.0
        self._current_done = False
        return self._set_current_observation(self.env.reset())

    def reset_to(self, state):
        if self._env_name == "WaitAtGoal":
            self.env.reset()
            self.env._set_state(np.asarray(state["states"], dtype=np.float32))
            self._current_reward = 0.0
            self._current_done = False
            return self._set_current_observation(self.env._get_obs())

        self.env.reset()
        self.env.sim.set_state_from_flattened(state["states"])
        self.env.sim.forward()
        raw_obs = self.env._get_observations(force_update=True)
        self._current_reward = 0.0
        self._current_done = False
        return self._set_current_observation(self.env._get_obs_custom(raw_obs))

    def render(self, mode="human", height=None, width=None, camera_name=None):
        if mode == "human":
            if self._env_name == "WaitAtGoal":
                return self.env.render(mode="human")
            return self.env.render()
        if mode == "rgb_array":
            if self._env_name == "WaitAtGoal":
                return self.env.render(mode="rgb_array")
            if self._current_raw_obs is None:
                raise RuntimeError("Call reset before rendering")
            return self._image_to_uint8_hwc(self._current_raw_obs["full_image"])
        raise NotImplementedError(f"mode={mode} is not implemented")

    def get_observation(self, obs=None):
        if obs is None:
            if self._current_obs is None:
                raise RuntimeError("Call reset before requesting an observation")
            return deepcopy(self._current_obs)
        return self._convert_observation(obs)

    def get_state(self):
        if self._env_name == "WaitAtGoal":
            return {
                "states": np.asarray(
                    [*self.env.agent.position, *self.env.goal_pos], dtype=np.float32
                )
            }
        return {"states": np.asarray(self.env.sim.get_state().flatten())}

    def get_reward(self):
        return self._current_reward

    def get_goal(self):
        raise NotImplementedError("Temporal tasks do not expose goal observations")

    def set_goal(self, **kwargs):
        raise NotImplementedError("Temporal tasks do not support externally-set goals")

    def is_done(self):
        return self._current_done

    def is_success(self):
        # Both tasks assign reward 1 only for their final task completion.
        # This avoids treating LiftQA's intermediate generic lift reward as a
        # benchmark success.
        return {"task": bool(self._current_done and self._current_reward >= 1.0)}

    @property
    def action_dimension(self):
        return 2 if self._env_name == "WaitAtGoal" else 3

    @property
    def name(self):
        return self._env_name

    @property
    def type(self):
        return EB.EnvType.TEMPORAL_TYPE

    @property
    def version(self):
        return "temporal-env-v1"

    def serialize(self):
        return {
            "env_name": self.name,
            "env_version": self.version,
            "type": self.type,
            "env_kwargs": deepcopy(self._init_kwargs),
        }

    @classmethod
    def create_for_data_processing(
        cls,
        env_name,
        camera_names=None,
        camera_height=None,
        camera_width=None,
        reward_shaping=None,
        render=None,
        render_offscreen=None,
        use_image_obs=None,
        use_depth_obs=None,
        **kwargs,
    ):
        render_size = camera_height if camera_height is not None else kwargs.pop("render_size", 144)
        return cls(
            env_name=env_name,
            render=False if render is None else render,
            render_offscreen=True if render_offscreen is None else render_offscreen,
            use_image_obs=True if use_image_obs is None else use_image_obs,
            use_depth_obs=False if use_depth_obs is None else use_depth_obs,
            render_size=render_size,
            **kwargs,
        )

    @property
    def rollout_exceptions(self):
        return ()

    @property
    def base_env(self):
        return self.env

    def close(self):
        self.env.close()
