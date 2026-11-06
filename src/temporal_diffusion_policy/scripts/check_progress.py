#!/usr/bin/env python3
"""Report checkpoint completion and temporal-conditioning configuration.

The tool is read-only. It reads ``config.txt`` when available, otherwise a
``final.pth.readonly_dump.txt`` produced for archived checkpoints. Completed
runs report elapsed training time from ``config.txt`` to ``final.pth``.
"""

from __future__ import annotations

import argparse
import ast
from dataclasses import dataclass
from pathlib import Path
import re
import time
from typing import Any


DEFAULT_CHECKPOINT_ROOT = Path.cwd() / "data" / "checkpoints"
DEFAULT_INFERENCE_ROOT = Path.cwd() / "data" / "inference"
GREEN = "\033[32m"
RESET = "\033[0m"
ACTIVE_WINDOW_SECONDS = 3 * 60 * 60
WANDB_API_TIMEOUT_SECONDS = 5


@dataclass(frozen=True)
class CheckpointProgress:
    path: Path
    dataset_path: str
    final_exists: bool
    has_evaluation: bool
    config_source: str
    label: str
    wandb_entity: str | None
    wandb_project: str | None
    wandb_run_id: str | None
    wandb_run_name: str | None
    num_epochs: int | None
    training_duration_seconds: float | None


def parse_scalar(value: str) -> Any:
    """Parse config values while retaining unknown values as strings."""
    value = value.strip()
    if value in {"True", "False"}:
        return value == "True"
    if value == "None":
        return None
    try:
        return ast.literal_eval(value)
    except (SyntaxError, ValueError):
        return value


def read_config(path: Path) -> dict[str, Any]:
    """Read the repository's simple ``key: value`` configuration format."""
    config: dict[str, Any] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip() or ":" not in line:
            continue
        key, value = line.split(":", 1)
        config[key.strip()] = parse_scalar(value)
    return config


def load_checkpoint_config(checkpoint_dir: Path) -> tuple[dict[str, Any], str]:
    """Prefer live config.txt and otherwise fall back to an archived dump."""
    for filename, source in (
        ("config.txt", "config.txt"),
        ("final.pth.readonly_dump.txt", "readonly dump"),
    ):
        path = checkpoint_dir / filename
        if path.is_file():
            return read_config(path), source
    return {}, "missing"


def training_duration_seconds(checkpoint_dir: Path) -> float | None:
    """Return elapsed training time from saved config to final checkpoint."""
    config_path = checkpoint_dir / "config.txt"
    final_path = checkpoint_dir / "final.pth"
    if not config_path.is_file() or not final_path.is_file():
        return None
    try:
        return max(0.0, final_path.stat().st_mtime - config_path.stat().st_mtime)
    except OSError:
        return None


def format_duration(seconds: float | None) -> str:
    """Format a duration compactly for the terminal table."""
    if seconds is None:
        return "-"
    total_seconds = round(seconds)
    hours, remainder = divmod(total_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    if hours:
        return f"{hours}h {minutes}m {seconds}s"
    if minutes:
        return f"{minutes}m {seconds}s"
    return f"{seconds}s"


def generate_short_label(config: dict[str, Any]) -> str:
    """Mirror inference.generate_short_label_from_inference_config."""
    label_parts = []
    if config.get("use_temporal_time_signal", False):
        label_parts.append("+T")
    if config.get("use_agent_velocity_conditioning", False):
        label_parts.append("+V")
    if config.get("use_direct_idleness_conditioning", False):
        label_parts.append("+DI")

    if config.get("use_idleness_encoding", False):
        label_parts.append("+ID")
    if config.get("use_saturating_progress_encoding", False):
        label_parts.append("+SaP")
    if config.get("use_sinusoidal_progress_encoding", False):
        label_parts.append(f"+SiP{config.get('sinusoidal_progress_encoding_dim', 128)}")
    if config.get("use_pause_induced_perturbation", False):
        label_parts.append("PIP")
    return "_".join(label_parts) or "-"


def _normalise_datestamp(value: str) -> str:
    """Treat YYYYMMDDHHMMSS and YYYYMMDD_HHMMSS as the same run timestamp."""
    return value.replace("_", "")


def _checkpoint_key(checkpoint_name: str) -> tuple[str, str] | None:
    """Extract ``(environment, datestamp)`` from an environment-prefixed checkpoint."""
    match = re.match(
        r"^(?P<env>.+)_(?P<date>\d{8}_\d{6})(?:_seed\d+)?$", checkpoint_name
    )
    if match is None:
        return None
    return match.group("env"), _normalise_datestamp(match.group("date"))


def _inference_keys(inference_root: Path) -> set[tuple[str, str]]:
    """Return checkpoint keys represented by inference datasets in one directory."""
    if not inference_root.is_dir():
        return set()

    keys = set()
    for dataset_path in inference_root.iterdir():
        if not dataset_path.is_dir() or dataset_path.suffix != ".zarr":
            continue
        name = dataset_path.name.removesuffix(".zarr")
        if "_ckpt" not in name:
            continue
        prefix, checkpoint_part = name.split("_ckpt", maxsplit=1)
        # Older inference paths place a six-character host hash before _ckpt.
        if re.search(r"_[0-9a-f]{6}$", prefix):
            prefix = prefix[:-7]
        checkpoint_part = checkpoint_part.removeprefix("interactive_")
        match = re.match(r"(?P<date>\d{8}_?\d{6})(?:_|$)", checkpoint_part)
        if match is not None:
            keys.add((prefix, _normalise_datestamp(match.group("date"))))
    return keys


def collect_checkpoint_progress(
    checkpoint_root: Path, inference_root: Path = DEFAULT_INFERENCE_ROOT
) -> list[CheckpointProgress]:
    if not checkpoint_root.is_dir():
        raise FileNotFoundError(f"Checkpoint root does not exist: {checkpoint_root}")

    inference_keys = _inference_keys(inference_root)
    progress = []
    for checkpoint_dir in checkpoint_root.iterdir():
        if not checkpoint_dir.is_dir():
            continue
        config, source = load_checkpoint_config(checkpoint_dir)
        progress.append(CheckpointProgress(
            path=checkpoint_dir,
            dataset_path=str(config.get("dataset_path", "<unknown dataset>")),
            final_exists=(checkpoint_dir / "final.pth").is_file(),
            has_evaluation=_checkpoint_key(checkpoint_dir.name) in inference_keys,
            config_source=source,
            label=generate_short_label(config),
            wandb_entity=config.get("wandb_entity"),
            wandb_project=config.get("wandb_project"),
            wandb_run_id=config.get("wandb_run_id"),
            wandb_run_name=config.get("wandb_run_name"),
            num_epochs=config.get("num_epochs"),
            training_duration_seconds=training_duration_seconds(checkpoint_dir),
        ))
    return sorted(progress, key=lambda item: (item.dataset_path, item.path.name))


def checkpoint_status(checkpoint_dir: Path, now: float | None = None) -> str:
    """Return the current state of a training directory.

    A final checkpoint is definitive completion.  Until then, consider the
    run active when any regular file inside the checkpoint directory was
    updated in the previous three hours.
    """
    if (checkpoint_dir / "final.pth").is_file():
        return "complete"

    current_time = time.time() if now is None else now
    for path in checkpoint_dir.rglob("*"):
        try:
            if (
                path.is_file()
                and current_time - path.stat().st_mtime < ACTIVE_WINDOW_SECONDS
            ):
                return "active"
        except OSError:
            continue
    return "incomplete"


def wandb_run_state_and_epoch(
    item: CheckpointProgress, api: Any
) -> tuple[str | None, str]:
    """Return a W&B run's state and latest epoch using its configured identity."""
    try:
        project_path = f"{item.wandb_entity}/{item.wandb_project}"
        if item.wandb_run_id is not None:
            run = api.run(f"{project_path}/{item.wandb_run_id}")
        elif item.wandb_run_name is not None:
            runs = api.runs(project_path, filters={"display_name": item.wandb_run_name})
            run = next(iter(runs), None)
        if run is None:
            return None, "-"
        epoch = run.summary.get("epoch")
        total = int(item.num_epochs) if item.num_epochs is not None else None
        epoch_progress = "-"
        if epoch is not None:
            completed = int(epoch) + 1
            epoch_progress = f"{completed}/{total}" if total is not None else str(completed)
        return str(run.state).lower(), epoch_progress
    except Exception:
        # Progress reporting is read-only and should still work offline.
        return None, "-"


def print_progress(progress: list[CheckpointProgress], checkpoint_root: Path) -> None:
    wandb_api: Any | None = None
    wandb_api_initialized = False

    def get_wandb_api() -> Any | None:
        nonlocal wandb_api, wandb_api_initialized
        if wandb_api_initialized:
            return wandb_api
        wandb_api_initialized = True
        try:
            import wandb

            wandb_api = wandb.Api(timeout=WANDB_API_TIMEOUT_SECONDS)
        except Exception:
            wandb_api = None
        return wandb_api

    rows = []
    for item in progress:
        status = checkpoint_status(item.path)
        epoch_progress = "-"
        has_wandb_identity = (
            item.wandb_entity is not None
            and item.wandb_project is not None
            and (item.wandb_run_id is not None or item.wandb_run_name is not None)
        )
        if status != "complete" and has_wandb_identity:
            api = get_wandb_api()
            if api is not None:
                wandb_state, epoch_progress = wandb_run_state_and_epoch(item, api)
                if status == "incomplete" and wandb_state == "running":
                    status = "active"
        rows.append((
            item.path.name,
            status,
            epoch_progress if status == "active" else "-",
            "yes" if item.has_evaluation else "no",
            item.label,
            format_duration(item.training_duration_seconds),
        ))
    headers = (
        "checkpoint", "status", "epoch", "eval", "conditioning", "training time"
    )
    widths = [
        max(len(header), *(len(row[index]) for row in rows)) if rows else len(header)
        for index, header in enumerate(headers)
    ]

    def print_rows(rows_to_print: list[tuple[str, str, str, str, str, str]]) -> None:
        for row in rows_to_print:
            formatted_row = []
            for index, value in enumerate(row):
                padded_value = value.ljust(widths[index])
                if (headers[index] == "status" and value == "complete") or (
                    headers[index] == "eval" and value == "yes"
                ):
                    padded_value = f"{GREEN}{padded_value}{RESET}"
                elif headers[index] == "status" and value == "active":
                    padded_value = f"\033[33m{padded_value}{RESET}"
                formatted_row.append(padded_value)
            print("  ".join(formatted_row))

    def print_section(
        title: str, rows_to_print: list[tuple[str, str, str, str, str, str]]
    ) -> None:
        print(f"\n{title}")
        if not rows_to_print:
            print("  none")
            return
        print("  ".join(header.ljust(widths[index]) for index, header in enumerate(headers)))
        print("  ".join("-" * width for width in widths))
        print_rows(rows_to_print)

    active_rows = [row for row in rows if row[1] == "active"]
    inactive_rows = [row for row in rows if row[1] != "active"]
    print(f"Checkpoint root: {checkpoint_root}")
    print_section("Completed or inactive training runs:", inactive_rows)
    print_section("Active training runs:", active_rows)
    print(f"\nTotal: {len(progress)} | complete: {sum(item.final_exists for item in progress)}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint-root",
        type=Path,
        default=DEFAULT_CHECKPOINT_ROOT,
        help=f"Checkpoint directory to scan (default: {DEFAULT_CHECKPOINT_ROOT})",
    )
    parser.add_argument(
        "--inference-root",
        type=Path,
        default=DEFAULT_INFERENCE_ROOT,
        help=f"Inference datasets to scan (default: {DEFAULT_INFERENCE_ROOT})",
    )
    args = parser.parse_args()
    progress = collect_checkpoint_progress(args.checkpoint_root, args.inference_root)
    print_progress(progress, args.checkpoint_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
