"""Regression tests for the experiment CLI's progress displays."""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

from temporal_diffusion_policy import experiment_cli
from temporal_diffusion_policy.scripts import check_progress
from temporal_diffusion_policy.scripts.check_progress import collect_checkpoint_progress, print_progress


def test_training_progress_prints_active_runs_in_final_section(
    tmp_path: Path, capsys
) -> None:
    checkpoint_root = tmp_path / "checkpoints"
    active = checkpoint_root / "WaitAtGoal_20260814_120000_seed1"
    complete = checkpoint_root / "WaitAtGoal_20260814_120001_seed2"
    active.mkdir(parents=True)
    complete.mkdir(parents=True)
    (active / "config.txt").write_text(
        "dataset_path: data/active.zarr\n", encoding="utf-8"
    )
    (active / "latest.pth").write_text("in progress", encoding="utf-8")
    (complete / "config.txt").write_text(
        "dataset_path: data/complete.zarr\n", encoding="utf-8"
    )
    (complete / "final.pth").write_text("finished", encoding="utf-8")

    print_progress(collect_checkpoint_progress(checkpoint_root), checkpoint_root)

    output = capsys.readouterr().out
    inactive_section, active_section = output.split("Active training runs:")
    assert "Completed or inactive training runs:" in inactive_section
    assert complete.name in inactive_section
    assert active.name not in inactive_section
    assert active.name in active_section
    assert "dataset" not in output
    assert "active.zarr" not in output
    assert "complete.zarr" not in output


def test_training_progress_uses_running_wandb_run_for_stale_checkpoint(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    checkpoint_root = tmp_path / "checkpoints"
    checkpoint = checkpoint_root / "WaitAtGoal_20260814_120000_seed1"
    checkpoint.mkdir(parents=True)
    (checkpoint / "config.txt").write_text(
        "\n".join((
            "dataset_path: data/example.zarr",
            "wandb_entity: example-entity",
            "wandb_project: example-project",
            "wandb_run_id: abc123",
            "num_epochs: 200",
        )),
        encoding="utf-8",
    )
    stale_time = time.time() - check_progress.ACTIVE_WINDOW_SECONDS - 1
    os.utime(checkpoint / "config.txt", (stale_time, stale_time))

    class FakeApi:
        def run(self, path: str):
            assert path == "example-entity/example-project/abc123"
            return type("Run", (), {"state": "running", "summary": {"epoch": 49}})()

    monkeypatch.setattr(check_progress, "WANDB_API_TIMEOUT_SECONDS", 1)
    monkeypatch.setattr(check_progress, "wandb_run_state_and_epoch", lambda item, api: ("running", "50/200"))
    monkeypatch.setattr("wandb.Api", lambda timeout: FakeApi())

    print_progress(collect_checkpoint_progress(checkpoint_root), checkpoint_root)

    output = capsys.readouterr().out
    _, active_section = output.split("Active training runs:")
    assert checkpoint.name in active_section
    assert "50/200" in active_section


def test_evaluation_progress_prints_active_runs_in_final_section(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    inference_root = tmp_path / "data" / "inference"
    active = inference_root / "active.zarr"
    complete = inference_root / "complete.zarr"
    for dataset, num_episodes, completed_episodes in (
        (active, 5, 2),
        (complete, 3, 3),
    ):
        (dataset / "meta" / "episode_ends").mkdir(parents=True)
        (dataset / ".zattrs").write_text(
            json.dumps({"num_episodes": num_episodes}), encoding="utf-8"
        )
        (dataset / "meta" / "episode_ends" / ".zarray").write_text(
            json.dumps({"shape": [completed_episodes]}), encoding="utf-8"
        )
    os.utime(active / "meta" / "episode_ends" / ".zarray", (time.time(), time.time()))
    monkeypatch.setattr(experiment_cli, "REPO_ROOT", tmp_path)

    experiment_cli.run_check_evaluation_progress()

    output = capsys.readouterr().out
    inactive_section, active_section = output.split("Active evaluation runs:")
    assert "Completed or inactive evaluation runs:" in inactive_section
    assert complete.name in inactive_section
    assert active.name not in inactive_section
    assert active.name in active_section


def test_checkpoint_prompt_displays_only_checkpoint_names(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    checkpoint = tmp_path / "data" / "checkpoints" / "WaitAtGoal_20260814_120000"
    checkpoint.mkdir(parents=True)
    monkeypatch.setattr("builtins.input", lambda _: "1")

    assert experiment_cli._prompt_checkpoint([checkpoint]) == checkpoint

    output = capsys.readouterr().out
    assert checkpoint.name in output
    assert str(checkpoint.parent) not in output


def test_main_returns_to_workflow_menu_after_a_command(monkeypatch) -> None:
    selections = iter(["Check train progress", "Exit"])
    monkeypatch.setattr(experiment_cli, "_prompt_menu", lambda *_: next(selections))
    calls: list[str] = []
    monkeypatch.setattr(
        experiment_cli, "run_check_progress", lambda: calls.append("progress")
    )

    experiment_cli.main()

    assert calls == ["progress"]
