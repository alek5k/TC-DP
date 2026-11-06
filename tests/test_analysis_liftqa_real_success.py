"""Check the real-world success rule independently of simulation rewards."""

import numpy as np
import pytest

from temporal_diffusion_policy.analysis.waittime import extract_episode_metrics
from temporal_diffusion_policy.replay_buffer import ReplayBuffer


def _rollout(length, wait, *, environment="LiftQA-Real", max_steps=5, reward=0.0, goal_state=False):
    replay = ReplayBuffer.create_empty_zarr()
    replay.root.attrs.update({"env_name": environment, "max_steps": max_steps})
    episode = {
        "wait_time": np.full(length, wait, dtype=np.float32),
        "step_count": np.arange(length, dtype=np.int32),
        "wait_times_each_visit": np.full((length, 1), wait, dtype=np.float32),
        "agent_pose": np.zeros((length, 3), dtype=np.float32),
        "reward": np.full(length, reward, dtype=np.float32),
    }
    if goal_state:
        episode["goal_state"] = np.full(length, 2, dtype=np.int32)
    replay.add_episode(episode)
    return replay


@pytest.mark.parametrize("goal_state", [False, True])
@pytest.mark.parametrize(
    "length,wait,expected",
    [(4, 2.1, 100.0), (5, 2.1, 0.0), (6, 2.1, 0.0), (4, 2.0, 0.0), (4, 1.9, 0.0)],
)
def test_real_success_requires_strict_wait_and_early_termination(length, wait, expected, goal_state):
    # Neither a high reward nor a saved goal state may bypass the real rule.
    replay = _rollout(length, wait, reward=1.0, goal_state=goal_state)
    assert extract_episode_metrics("run.zarr", replay)[-1] == expected


def test_real_success_uses_recorded_step_limit():
    replay = _rollout(4, 2.1, max_steps=3)
    assert extract_episode_metrics("run.zarr", replay)[-1] == 0.0


def test_real_success_supports_legacy_metadata():
    replay = _rollout(4, 2.1)
    del replay.root.attrs["env_name"]
    del replay.root.attrs["max_steps"]
    assert extract_episode_metrics("LiftQA-Real_run.zarr", replay)[-1] == 100.0


@pytest.mark.parametrize("length,reward,expected", [(4, 0.6, 100.0), (400, 0.6, 0.0), (400, 1.0, 100.0)])
def test_simulation_success_still_uses_reward_and_termination(length, reward, expected):
    replay = _rollout(length, 0.0, environment="LiftQA", reward=reward)
    assert extract_episode_metrics("run.zarr", replay)[-1] == expected


def test_environment_metadata_takes_precedence_over_path():
    replay = _rollout(4, 0.0, environment="LiftQA", reward=0.6)
    assert extract_episode_metrics("LiftQA-Real_old_name.zarr", replay)[-1] == 100.0
