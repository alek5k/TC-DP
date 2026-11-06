#!/usr/bin/env python3
"""Show rollout activity based on HDF file modification times."""

import argparse
import time
from dataclasses import dataclass
from pathlib import Path


TASK_ORDER = ["can", "square", "lift", "tool_hang", "transport"]
ALGO_ORDER = ["bc", "bc_rnn", "hbc", "diffusion_policy", "tc_diffusion_policy", "diffusion_policy_mod", "tc_diffusion_policy_mod"]
DATASET_TYPE_ORDER = ["ph", "mh"]
ANSI_GREEN = "\033[32m"
ANSI_BLUE = "\033[34m"
ANSI_RESET = "\033[0m"
STATUS_ACTIVE = "active"
STATUS_COMPLETE = "complete"


@dataclass
class RolloutProgress:
    file_path: Path
    algo: str
    task: str
    dataset_type: str
    info_path: Path
    info_mtime: float | None
    mtime: float | None


def find_rollout_paths(root: Path) -> list[Path]:
    paths = list(root.glob("**/*.hdf"))
    paths.extend(root.glob("**/*.hdf5"))
    return sorted(paths)


def rollout_parts_from_path(file_path: Path, root: Path) -> tuple[str, str, str]:
    try:
        rel_parts = file_path.relative_to(root).parts
    except ValueError:
        return "?", "?", "?"
    if len(rel_parts) < 4:
        return "?", "?", "?"
    return rel_parts[0], rel_parts[1], rel_parts[2]


def collect_progress(root: Path) -> list[RolloutProgress]:
    runs = []
    for file_path in find_rollout_paths(root):
        task, dataset_type, algo = rollout_parts_from_path(file_path, root)
        info_path = file_path.with_name(file_path.stem + "_info.txt")
        try:
            mtime = file_path.stat().st_mtime
        except OSError:
            mtime = None
        try:
            info_mtime = info_path.stat().st_mtime
        except OSError:
            info_mtime = None
        runs.append(
            RolloutProgress(
                file_path=file_path,
                algo=algo,
                task=task,
                dataset_type=dataset_type,
                info_path=info_path,
                info_mtime=info_mtime,
                mtime=mtime,
            )
        )
    return runs


def age_seconds(run: RolloutProgress, now: float) -> float | None:
    if run.mtime is None:
        return None
    return max(0.0, now - run.mtime)


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


def rollout_duration_seconds(run: RolloutProgress) -> float | None:
    if run.info_mtime is None or run.mtime is None:
        return None
    return max(0.0, run.mtime - run.info_mtime)


def format_duration(seconds: float | None) -> str:
    if seconds is None:
        return "?"
    if seconds < 60:
        return "{:.0f}s".format(seconds)
    if seconds < 3600:
        return "{:.0f}m".format(seconds / 60)
    if seconds < 86400:
        return "{:.1f}h".format(seconds / 3600)
    return "{:.1f}d".format(seconds / 86400)


def rollout_status(run: RolloutProgress, now: float, active_within_seconds: int) -> str:
    age = age_seconds(run, now)
    if age is not None and age <= active_within_seconds:
        return STATUS_ACTIVE
    return STATUS_COMPLETE


def colorize(text: str, status: str, color: bool) -> str:
    if color and status == STATUS_ACTIVE:
        return "{}{}{}".format(ANSI_BLUE, text, ANSI_RESET)
    if color and status == STATUS_COMPLETE:
        return "{}{}{}".format(ANSI_GREEN, text, ANSI_RESET)
    return text


def format_status(run: RolloutProgress, now: float, active_within_seconds: int, color: bool, width: int = 24) -> str:
    status = rollout_status(run, now, active_within_seconds)
    text = "{} ({})".format(status, format_age(age_seconds(run, now)))
    return colorize(text.ljust(width), status=status, color=color)


def format_progress(run: RolloutProgress, now: float, active_within_seconds: int) -> str:
    return "{}  {:>7}  {:<25}  {:<10}  ({:<2})  {}".format(
        format_status(run, now, active_within_seconds, color=True),
        format_duration(rollout_duration_seconds(run)),
        run.algo,
        run.task,
        run.dataset_type,
        run.file_path.name,
    )


def sort_key(run: RolloutProgress):
    task_index = TASK_ORDER.index(run.task) if run.task in TASK_ORDER else len(TASK_ORDER)
    algo_index = ALGO_ORDER.index(run.algo) if run.algo in ALGO_ORDER else len(ALGO_ORDER)
    dataset_type_index = (
        DATASET_TYPE_ORDER.index(run.dataset_type)
        if run.dataset_type in DATASET_TYPE_ORDER
        else len(DATASET_TYPE_ORDER)
    )
    return (dataset_type_index, run.dataset_type, task_index, algo_index, run.file_path.name)


def main():
    parser = argparse.ArgumentParser(description="Show rollout progress for rollouts/**/*.hdf*")
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("rollouts"),
        help="root directory to scan for rollout HDF files",
    )
    parser.add_argument(
        "--active-within",
        type=int,
        default=60,
        help="consider a rollout active if its file was modified within this many seconds",
    )
    args = parser.parse_args()

    root = args.root.expanduser()
    if not root.exists():
        raise FileNotFoundError("Root path not found: {}".format(root))

    runs = collect_progress(root)
    if not runs:
        print("No rollout HDF files found under {}".format(root))
        return

    print("Found {} rollout file(s) under {}".format(len(runs), root))
    print("")
    now = time.time()
    last_dataset_type = None
    for run in sorted(runs, key=sort_key):
        if last_dataset_type is not None and run.dataset_type != last_dataset_type:
            print("")
        print(format_progress(run, now=now, active_within_seconds=args.active_within))
        last_dataset_type = run.dataset_type


if __name__ == "__main__":
    main()
