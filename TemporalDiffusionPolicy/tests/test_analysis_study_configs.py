"""Tests for the JSON-backed manual study configuration editor."""

import json

import pytest

from temporal_diffusion_policy.analysis import study_configs


def test_add_and_remove_study_run(tmp_path, monkeypatch) -> None:
    config_path = tmp_path / "study_wait_at_goal_tc_dp_paper.json"
    config_path.write_text(
        json.dumps(
            {"environment": "WaitAtGoal", "study": "tc_dp_paper", "runs": []}
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(study_configs, "STUDY_CONFIG_DIRECTORY", tmp_path)
    run = {
        "path": "data/inference/example.zarr",
        "dataset_type": "Inference",
    }

    study_configs.add_study_run("WaitAtGoal", "tc_dp_paper", run)
    assert study_configs.load_study_configs()["WaitAtGoal"]["tc_dp_paper"] == [run]

    removed = study_configs.remove_study_run("WaitAtGoal", "tc_dp_paper", 0)
    assert removed == run
    assert study_configs.load_study_configs()["WaitAtGoal"]["tc_dp_paper"] == []

    with pytest.raises(ValueError, match="No run 1"):
        study_configs.remove_study_run("WaitAtGoal", "tc_dp_paper", 0)


def test_load_study_config_allows_jsonc_comments(tmp_path, monkeypatch) -> None:
    (tmp_path / "study_wait_at_goal_tc_dp_paper.json").write_text(
        """// A line comment is allowed.
{
  "environment": "WaitAtGoal", /* A block comment is also allowed. */
  "study": "tc_dp_paper",
  "runs": [
    {"path": "data/inference/https://example.zarr", "dataset_type": "Inference"}
  ]
}
""",
        encoding="utf-8",
    )
    monkeypatch.setattr(study_configs, "STUDY_CONFIG_DIRECTORY", tmp_path)

    loaded = study_configs.load_study_configs()

    assert loaded["WaitAtGoal"]["tc_dp_paper"][0]["path"].endswith(
        "https://example.zarr"
    )


def test_editing_one_study_does_not_rewrite_other_study_files(tmp_path, monkeypatch) -> None:
    target_path = tmp_path / "study_wait_at_goal_tc_dp_paper.json"
    target_path.write_text(
        json.dumps(
            {"environment": "WaitAtGoal", "study": "tc_dp_paper", "runs": []}
        ),
        encoding="utf-8",
    )
    untouched_path = tmp_path / "study_wait_at_goal_pip_sweep.json"
    untouched_source = "// Keep this comment.\n" + json.dumps(
        {"environment": "WaitAtGoal", "study": "pip_sweep", "runs": []}
    )
    untouched_path.write_text(untouched_source, encoding="utf-8")
    monkeypatch.setattr(study_configs, "STUDY_CONFIG_DIRECTORY", tmp_path)

    study_configs.add_study_run(
        "WaitAtGoal",
        "tc_dp_paper",
        {"path": "data/inference/example.zarr", "dataset_type": "Inference"},
    )

    assert untouched_path.read_text(encoding="utf-8") == untouched_source
