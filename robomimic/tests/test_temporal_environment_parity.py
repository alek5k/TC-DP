"""Fixed-seed parity checks between TemporalDiffusionPolicy and copied envs.

Run with the TemporalDiffusionPolicy environment, which includes pygame,
pymunk, robosuite, and MuJoCo. The copied modules are imported from this repo.
"""
import importlib.util
import json
from pathlib import Path
import unittest

import h5py
import numpy as np


TEMPORAL_ROOT = Path("/mnt/shared/TemporalDiffusionPolicy")


def load_source_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class TestTemporalEnvironmentParity(unittest.TestCase):
    def test_native_adapter_from_converted_hdf5(self):
        from robomimic.utils import env_utils

        fixtures = (
            ("datasets/temporal/waitatgoal_image.hdf5", (2,), (144, 144, 3)),
            ("datasets/temporal/liftqa_image.hdf5", (3,), (144, 144, 3)),
        )
        for dataset_path, action_shape, image_shape in fixtures:
            with h5py.File(dataset_path, "r") as dataset:
                env_meta = json.loads(dataset["data"].attrs["env_args"])
            self.assertEqual(env_meta["type"], 4)
            env = env_utils.create_env_from_metadata(
                env_meta=env_meta,
                render=False,
                render_offscreen=True,
                use_image_obs=True,
            )
            try:
                observation = env.reset()
                self.assertEqual(observation["image"].shape, image_shape)
                self.assertEqual(observation["image"].dtype, np.uint8)
                self.assertEqual(observation["agent_pose"].shape, action_shape)
                self.assertEqual(observation["agent_velocity"].shape, (1,))
                next_observation, _, _, info = env.step(np.zeros(action_shape, dtype=np.float32))
                self.assertEqual(next_observation["image"].shape, image_shape)
                self.assertEqual(next_observation["agent_velocity"].shape, (1,))
                self.assertIn("is_success", info)
            finally:
                env.close()

    def test_waitatgoal_fixed_seed_rollout(self):
        source_module = load_source_module(
            "temporal_source_waitatgoal", TEMPORAL_ROOT / "envs/pygame/waitatgoal_env.py"
        )
        from robomimic.envs.temporal.waitatgoal import WaitAtGoal

        actions = [np.array([100., 100.], dtype=np.float32), np.array([200., 250.], dtype=np.float32)]
        trajectories = []
        for env_class in (source_module.WaitAtGoal, WaitAtGoal):
            env = env_class(render_size=144)
            try:
                env.seed(123)
                obs = env.reset()
                trajectory = [(obs["full_image"].copy(), obs["agent_pose"].copy(), 0.0, False)]
                for action in actions:
                    obs, reward, done, _ = env.step(action.copy())
                    trajectory.append((obs["full_image"].copy(), obs["agent_pose"].copy(), reward, done))
                trajectories.append(trajectory)
            finally:
                env.close()

        for source_step, copied_step in zip(*trajectories):
            np.testing.assert_array_equal(source_step[0], copied_step[0])
            np.testing.assert_array_equal(source_step[1], copied_step[1])
            self.assertEqual(source_step[2:], copied_step[2:])

    def test_liftqa_fixed_seed_rollout(self):
        source_module = load_source_module(
            "temporal_source_liftqa", TEMPORAL_ROOT / "envs/robosuite/lift_qa.py"
        )
        from robomimic.envs.temporal import liftqa as copied_module

        actions = [np.array([0.02, 0.01, -1.0]), np.array([0.01, -0.01, 1.0])]
        trajectories = []
        for module in (source_module, copied_module):
            env = module.create_env(
                render=False, constrain_motion=True, control_hz=10,
                max_episode_length=400, seed=123, camera_height_width=(144, 144),
            )
            try:
                obs = env.reset()
                trajectory = [(obs["full_image"].copy(), obs["agent_pose"].copy(), 0.0, False)]
                for action in actions:
                    obs, reward, done, _ = env.step(module.translate_3_dim_action_to_7d(action))
                    trajectory.append((obs["full_image"].copy(), obs["agent_pose"].copy(), reward, done))
                trajectories.append(trajectory)
            finally:
                env.close()

        for source_step, copied_step in zip(*trajectories):
            np.testing.assert_allclose(source_step[0], copied_step[0], rtol=0.0, atol=0.0)
            np.testing.assert_allclose(source_step[1], copied_step[1], rtol=0.0, atol=0.0)
            self.assertEqual(source_step[2:], copied_step[2:])


if __name__ == "__main__":
    unittest.main()
