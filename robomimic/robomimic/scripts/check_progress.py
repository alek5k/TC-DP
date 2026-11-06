#!/usr/bin/env python3
"""Show training progress for all trained model log files."""

import argparse
import json
import re
import time
from dataclasses import dataclass
from pathlib import Path
from read_tensorboard_data import get_training_summary
from datetime import datetime

TRAIN_EPOCH_RE = re.compile(r"\bTrain Epoch\s+(\d+)\b")
SUCCESS_RE = re.compile(r"_success_([0-9]+(?:\.[0-9]+)?)\.pth$")
TASK_ORDER = ["can", "square", "lift", "tool_hang", "transport"]
ALGO_ORDER = ["bc", "bc_mod_nocrop", "bc_rnn", "bc_rnn_mod_nocrop", "hbc", "diffusion_policy", "diffusion_policy_mod", "diffusion_policy_mod_nocrop", "tc_diffusion_policy", "tc_diffusion_policy_mod", "tc_diffusion_policy_mod_nocrop"]
DATASET_TYPE_ORDER = ["ph", "mh", "waitatgoal", "liftqa"]
ANSI_GREEN = "\033[32m"
ANSI_BLUE = "\033[34m"
ANSI_RED = "\033[31m"
ANSI_RESET = "\033[0m"
RUN_ID_RE = re.compile(r"^(\d{14})_")


@dataclass
class RunProgress:
    run_dir: Path
    log_path: Path
    config_path: Path
    current_epoch: int | None
    total_epochs: int | None
    experiment_name: str
    algo: str
    task: str
    dataset_type: str
    last_updated_time: float | None
    started_time: float | None
    last_pth_mtime: float | None
    best_success_rate: float | None
    evaluation_complete: bool = False


def find_log_paths(root: Path) -> list[Path]:
    return sorted(root.glob("**/log.txt"))


def latest_train_epoch(log_path: Path) -> int | None:
    latest_epoch = None
    with log_path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            match = TRAIN_EPOCH_RE.search(line)
            if match:
                latest_epoch = int(match.group(1))
    return latest_epoch


def load_config(config_path: Path) -> dict:
    with config_path.open("r", encoding="utf-8") as handle:
        config = json.load(handle)
    return config if isinstance(config, dict) else {}


def total_epochs_from_config(config: dict) -> int | None:
    train_cfg = config.get("train", {})
    if isinstance(train_cfg, dict) and train_cfg.get("num_epochs") is not None:
        return int(train_cfg["num_epochs"])

    # Saved configs may also include optimizer metadata populated by train.py.
    algo_cfg = config.get("algo", {})
    if isinstance(algo_cfg, dict):
        optim_params = algo_cfg.get("optim_params", {})
        if isinstance(optim_params, dict):
            for optim_cfg in optim_params.values():
                if isinstance(optim_cfg, dict) and optim_cfg.get("num_epochs") is not None:
                    return int(optim_cfg["num_epochs"])

    return None


def experiment_name_from_config(config: dict, run_dir: Path) -> str:
    experiment_cfg = config.get("experiment", {})
    if isinstance(experiment_cfg, dict) and experiment_cfg.get("name"):
        return str(experiment_cfg["name"])
    return run_dir.parent.name


def run_parts_from_path(run_dir: Path) -> tuple[str, str, str]:
    parts = run_dir.parts
    try:
        index = parts.index("temporaldp")
        return parts[index + 1], parts[index + 2], parts[index + 3]
    except (ValueError, IndexError):
        return "?", "?", "?"


def training_summary_from_run(run_dir: Path, log_path: Path) -> dict:
    """Read a usable training event file, or fall back to the text log."""
    event_paths = sorted(run_dir.glob("**/*tfevents.*"), reverse=True)
    for event_path in event_paths:
        try:
            return get_training_summary(str(event_path.absolute()))
        except (KeyError, OSError, ValueError):
            # A run can contain an incomplete event file, or a non-training
            # writer such as a profiler. Keep looking rather than aborting the
            # report for every other run.
            continue

    try:
        log_mtime = log_path.stat().st_mtime
    except OSError:
        log_mtime = None
    return {
        "max_steps": latest_train_epoch(log_path),
        "start_time": log_mtime,
        "last_updated_time": log_mtime,
    }


def collect_progress(root: Path) -> list[RunProgress]:
    runs = []
    for log_path in find_log_paths(root):
        run_dir = log_path.parent.parent
        config_path = run_dir / "config.json"
        last_pth_path = run_dir / "last.pth"
        models_dir = run_dir / "models"
        config = load_config(config_path) if config_path.exists() else {}
        algo, task, dataset_type = run_parts_from_path(run_dir)

        summary = training_summary_from_run(run_dir, log_path)


        try:
            last_pth_mtime = last_pth_path.stat().st_mtime
        except OSError:
            last_pth_mtime = None
        runs.append(
            RunProgress(
                run_dir=run_dir,
                log_path=log_path,
                config_path=config_path,
                current_epoch=summary['max_steps'],# latest_train_epoch(log_path),
                total_epochs=total_epochs_from_config(config),
                experiment_name=experiment_name_from_config(config, run_dir),
                algo=algo,
                task=task,
                dataset_type=dataset_type,
                started_time=summary['start_time'],
                last_updated_time=summary['last_updated_time'],
                last_pth_mtime=last_pth_mtime,
                best_success_rate=best_success_rate_from_models_dir(models_dir),
            )
        )
    return runs


def best_success_rate_from_models_dir(models_dir: Path) -> float | None:
    if not models_dir.exists():
        return None

    best = None
    for path in models_dir.glob("model_epoch_*.pth"):
        match = SUCCESS_RE.search(path.name)
        if not match:
            continue
        success = float(match.group(1))
        if best is None or success > best:
            best = success
    return best


def completed_evaluation_run_ids(rollouts_root: Path, now: float, active_within_seconds: int) -> set[str]:
    """Return run ids with a rollout HDF that is no longer actively changing.

    This follows ``check_progress_rollout.py``: a rollout is active while its
    HDF was modified within ``active_within_seconds`` and complete otherwise.
    """
    if not rollouts_root.exists():
        return set()

    completed = set()
    rollout_paths = list(rollouts_root.glob("**/*.hdf"))
    rollout_paths.extend(rollouts_root.glob("**/*.hdf5"))
    for rollout_path in rollout_paths:
        match = RUN_ID_RE.match(rollout_path.name)
        if match is None:
            continue
        try:
            age = max(0.0, now - rollout_path.stat().st_mtime)
        except OSError:
            continue
        if age > active_within_seconds:
            completed.add(match.group(1))
    return completed


def progress_bar(current: int | None, total: int | None, width: int) -> str:
    if current is None or total is None or total <= 0:
        return "[" + ("?" * width) + "]"

    ratio = min(max(current / total, 0.0), 1.0)
    filled = int(round(ratio * width))
    return "[" + ("#" * filled) + ("-" * (width - filled)) + "]"


def is_complete(run: RunProgress) -> bool:
    return (
        run.current_epoch is not None
        and run.total_epochs is not None
        and run.total_epochs > 0
        and run.current_epoch >= run.total_epochs
    )


def age_seconds(run: RunProgress, now: float) -> float | None:
    if run.last_updated_time is None:
        return None
    return max(0.0, now - run.last_updated_time)


def format_age(seconds: float | None) -> str:
    if seconds is None:
        return "?"
    if seconds < 60:
        return "{:.0f}s ago".format(seconds)
    if seconds < 3600:
        return "{:.0f}m ago".format(seconds / 60)
    if seconds < 86400:
        return "{:.1f}h ago".format(seconds / 3600)
    return "{:.1f}d ago".format(seconds / 86400)


def format_duration(seconds: float | None) -> str:
    if seconds is None or seconds < 0:
        return "?"
    if seconds < 60:
        return "{:.0f}s".format(seconds)
    if seconds < 3600:
        return "{:.0f}m".format(seconds / 60)
    if seconds < 86400:
        return "{:.1f}h".format(seconds / 3600)
    return "{:.1f}d".format(seconds / 86400)


def run_status(run: RunProgress, now: float, active_within_seconds: int) -> str:
    if is_complete(run):
        return "done"
    age = age_seconds(run, now)
    if age is not None and age <= active_within_seconds:
        return "active"
    return "stale"


def colorize(text: str, status: str, color: bool) -> str:
    if not color:
        return text
    if status == "done":
        return "{}{}{}".format(ANSI_GREEN, text, ANSI_RESET)
    if status == "active":
        return "{}{}{}".format(ANSI_BLUE, text, ANSI_RESET)
    if status == "no":
        return "{}{}{}".format(ANSI_RED, text, ANSI_RESET)
    if status == "yes":
        return "{}{}{}".format(ANSI_GREEN, text, ANSI_RESET)
    return text


def format_status(run: RunProgress, now: float, active_within_seconds: int, color: bool, width: int = 18) -> str:
    status = run_status(run, now, active_within_seconds)
    if status == "done":
        text = status
    else:
        text = "{} ({})".format(status, format_age(age_seconds(run, now)))
    return colorize(text.ljust(width), status=status, color=color)


def progress_cells(
    run: RunProgress,
    now: float,
    active_within_seconds: int,
    show_progress_bar: bool,
    width: int,
) -> list[str]:
    current = run.current_epoch
    total = run.total_epochs
    elapsed = format_duration(run.last_updated_time - run.started_time)

    if current is None or total is None or total <= 0:
        pct = "??.?%"
        epoch_text = "{}/{}".format(current if current is not None else "?", total if total is not None else "?")
    else:
        pct = "{:.1f}%".format(100.0 * min(current / total, 1.0))
        epoch_text = "{}/{}".format(current, total)

    success_text = (
        "{:.2f}".format(run.best_success_rate)
        if run.best_success_rate is not None
        else "?"
    )

    cells = [
        run.dataset_type,
        run.task,
        run.algo,
        run.run_dir.name,
        epoch_text,
        pct,
    ]
    if show_progress_bar:
        cells.append(progress_bar(current, total, width))
    cells.extend(
        [
            elapsed,
            run_status(run, now, active_within_seconds),
            "yes" if run.evaluation_complete else "no",
            success_text,
        ]
    )
    return cells


def format_progress(
    run: RunProgress,
    width: int,
    now: float,
    active_within_seconds: int,
    show_progress_bar: bool = False,
) -> str:
    """Compact single-line format used by the run-deletion menu."""
    return "  ".join(progress_cells(run, now, active_within_seconds, show_progress_bar, width))


def print_progress_table(
    runs: list[RunProgress],
    now: float,
    active_within_seconds: int,
    show_progress_bar: bool,
    width: int,
    color: bool,
) -> None:
    headers = ["Dataset", "Task", "Algo", "Run", "Epoch", "Progress"]
    if show_progress_bar:
        headers.append("Bar")
    headers.extend(["Elapsed", "Train", "Eval", "Best success"])
    rows = [
        progress_cells(run, now, active_within_seconds, show_progress_bar, width)
        for run in sorted(runs, key=sort_key)
    ]
    column_widths = [
        max(len(header), *(len(row[index]) for row in rows))
        for index, header in enumerate(headers)
    ]

    def separator() -> str:
        return "+-" + "-+-".join("-" * column_width for column_width in column_widths) + "-+"

    def render_row(cells: list[str], run: RunProgress | None = None) -> str:
        rendered = []
        for index, cell in enumerate(cells):
            padded = cell.ljust(column_widths[index])
            if run is not None and headers[index] == "Train":
                padded = colorize(padded, run_status(run, now, active_within_seconds), color)
            elif run is not None and headers[index] == "Eval":
                padded = colorize(padded, "yes" if cell == "yes" else "no", color)
            rendered.append(padded)
        return "| " + " | ".join(rendered) + " |"

    print(separator())
    print(render_row(headers))
    print(separator())
    for run, row in zip(sorted(runs, key=sort_key), rows):
        print(render_row(row, run))
    print(separator())


def sort_key(run: RunProgress):
    task_index = TASK_ORDER.index(run.task) if run.task in TASK_ORDER else len(TASK_ORDER)
    algo_index = ALGO_ORDER.index(run.algo) if run.algo in ALGO_ORDER else len(ALGO_ORDER)
    dataset_type_index = (
        DATASET_TYPE_ORDER.index(run.dataset_type)
        if run.dataset_type in DATASET_TYPE_ORDER
        else len(DATASET_TYPE_ORDER)
    )
    return (dataset_type_index, run.dataset_type, task_index, algo_index, run.run_dir.name)


def main():
    parser = argparse.ArgumentParser(description="Show train epoch progress for trained_models/**/log.txt")
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("trained_models"),
        help="root directory to scan for log.txt files",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=16,
        help="progress bar width in characters when --show-progress-bar is used",
    )
    parser.add_argument(
        "--show-progress-bar",
        action="store_true",
        help="include an ASCII progress-bar column",
    )
    parser.add_argument(
        "--active-within",
        type=int,
        default=300,
        help="consider an incomplete run active if log.txt was modified within this many seconds",
    )
    parser.add_argument(
        "--rollouts-root",
        type=Path,
        default=Path("rollouts"),
        help="root directory containing evaluation rollout HDF files",
    )
    parser.add_argument(
        "--rollout-active-within",
        type=int,
        default=60,
        help="use check_progress_rollout's completion rule: a newer HDF is still active",
    )
    parser.add_argument(
        "--no-color",
        action="store_true",
        help="disable ANSI status colors",
    )
    args = parser.parse_args()

    root = args.root.expanduser()
    if not root.exists():
        raise FileNotFoundError("Root path not found: {}".format(root))

    runs = collect_progress(root)
    if not runs:
        print("No log.txt files found under {}".format(root))
        return

    print("Found {} run(s) under {}".format(len(runs), root))
    print("")
    now = time.time()
    completed_run_ids = completed_evaluation_run_ids(
        args.rollouts_root.expanduser(),
        now=now,
        active_within_seconds=args.rollout_active_within,
    )
    for run in runs:
        run.evaluation_complete = run.run_dir.name in completed_run_ids
    print_progress_table(
        runs,
        now=now,
        active_within_seconds=args.active_within,
        show_progress_bar=args.show_progress_bar,
        width=args.width,
        color=not args.no_color,
    )


if __name__ == "__main__":
    main()
