"""Tests for automatic dataset resolution in wait-time analysis."""

import json

from temporal_diffusion_policy.analysis.waittime import (
    DatasetType,
    discover_environment_locators,
)


def _make_dataset(root, relative_path: str, env_name: str | None = None) -> None:
    dataset = root / relative_path
    (dataset / "meta" / "episode_ends").mkdir(parents=True)
    if env_name is not None:
        (dataset / ".zattrs").write_text(
            json.dumps({"env_name": env_name}), encoding="utf-8"
        )


def test_auto_resolution_finds_matching_complete_demo_and_inference(tmp_path) -> None:
    _make_dataset(tmp_path, "data/demonstration/wait_at_goal_dataset_v4.zarr")
    _make_dataset(
        tmp_path,
        "data/inference/custom_name.zarr",
        env_name="WaitAtGoal",
    )
    _make_dataset(tmp_path, "data/inference/LiftQA-Real_run.zarr", env_name="LiftQA-Real")
    (tmp_path / "data/inference/WaitAtGoal_incomplete.zarr").mkdir(parents=True)

    locators = discover_environment_locators("WaitAtGoal", repo_root=tmp_path)

    assert [(locator.path, locator.dataset_type) for locator in locators] == [
        ("data/demonstration/wait_at_goal_dataset_v4.zarr", DatasetType.DEMO),
        ("data/inference/custom_name.zarr", DatasetType.INFERENCE),
    ]
