#!/usr/bin/env python3
"""Extract per-epoch metric dictionaries from robomimic log files."""

import json
import math
import os
import re
import sys
from typing import Dict, Iterable, List, Sequence

ROOT_DIR = "/home/sydney1/Repos/robomimic/trained_models/temporaldp"
DEFAULT_OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "metrics_plots")

ANSI_ESCAPE_RE = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")
VALIDATION_EPOCH_RE = re.compile(r"\bValidation Epoch\s+(\d+)\b")
TRAIN_EPOCH_RE = re.compile(r"\bTrain Epoch\s+(\d+)\b")
ROLLOUT_EPOCH_RE = re.compile(r"\bEpoch\s+(\d+)\s+Rollouts\s+took\s+([0-9]*\.?[0-9]+)s\s+\(avg\)\s+with\s+results:")
ROLLOUT_ENV_RE = re.compile(r"\bEnv:\s*(\S+)")

def iter_log_paths(root: str) -> Iterable[str]:
    if os.path.isdir(root):
        for dirpath, _dirs, files in os.walk(root):
            for name in files:
                if name == "log.txt":
                    yield os.path.join(dirpath, name)
    else:
        yield root

def parse_metrics(log_path: str) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    config = load_run_config(log_path)
    current_epoch: int | None = None
    current_split: str | None = None
    current_rollout_env: str | None = None
    current_rollout_time: float | None = None
    in_dict = False
    brace_count = 0
    buffer: List[str] = []

    with open(log_path, "r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            clean = ANSI_ESCAPE_RE.sub("", line)

            if in_dict:
                buffer.append(clean)
                brace_count += clean.count("{") - clean.count("}")
                if brace_count == 0:
                    block = "".join(buffer)
                    buffer.clear()
                    in_dict = False

                    if current_epoch is not None and current_split is not None:
                        try:
                            metrics = json.loads(block)
                        except json.JSONDecodeError:
                            metrics = None

                        if isinstance(metrics, dict):
                            if current_split == "Rollout":
                                if current_rollout_env is not None:
                                    metrics.setdefault("Env", current_rollout_env)
                                if current_rollout_time is not None:
                                    metrics.setdefault("Rollout_Time_Avg", current_rollout_time)
                            row = {
                                "log_path": log_path,
                                "config": config,
                                "split": current_split,
                                "epoch": current_epoch,
                                "metrics": metrics,
                            }
                            rows.append(row)

                    current_epoch = None
                    current_split = None
                    current_rollout_env = None
                    current_rollout_time = None
                continue

            match = VALIDATION_EPOCH_RE.search(clean)
            if match:
                current_epoch = int(match.group(1))
                current_split = "Validation"
                continue

            match = TRAIN_EPOCH_RE.search(clean)
            if match:
                current_epoch = int(match.group(1))
                current_split = "Train"
                current_rollout_env = None
                current_rollout_time = None
                continue

            match = ROLLOUT_EPOCH_RE.search(clean)
            if match:
                current_epoch = int(match.group(1))
                current_split = "Rollout"
                current_rollout_time = float(match.group(2))
                current_rollout_env = None
                continue

            if current_split == "Rollout":
                match = ROLLOUT_ENV_RE.search(clean)
                if match:
                    current_rollout_env = match.group(1)
                    continue

            if current_epoch is not None and "{" in clean:
                in_dict = True
                buffer = [clean]
                brace_count = clean.count("{") - clean.count("}")
                if brace_count == 0:
                    block = "".join(buffer)
                    buffer.clear()
                    in_dict = False
                    if current_split is not None:
                        try:
                            metrics = json.loads(block)
                        except json.JSONDecodeError:
                            metrics = None
                        if isinstance(metrics, dict):
                            if current_split == "Rollout":
                                if current_rollout_env is not None:
                                    metrics.setdefault("Env", current_rollout_env)
                                if current_rollout_time is not None:
                                    metrics.setdefault("Rollout_Time_Avg", current_rollout_time)
                            row = {
                                "log_path": log_path,
                                "config": config,
                                "split": current_split,
                                "epoch": current_epoch,
                                "metrics": metrics,
                            }
                            rows.append(row)
                    current_epoch = None
                    current_split = None
                    current_rollout_env = None
                    current_rollout_time = None
                continue

    rows.sort(key=lambda item: (item.get("epoch", 0), item.get("split", "")))
    return rows


def build_nested(rows: Iterable[Dict[str, object]]) -> Dict[str, Dict[str, Dict[str, Dict[str, object]]]]:
    nested: Dict[str, Dict[str, Dict[str, Dict[str, object]]]] = {}
    for row in rows:
        log_path = str(row["log_path"])
        run_key = derive_run_key(log_path, row.get("config"))
        split = str(row["split"]).lower()
        epoch = str(row["epoch"])
        metrics = row.get("metrics")
        if not isinstance(metrics, dict):
            continue
        nested.setdefault(run_key, {}).setdefault(split, {})[epoch] = metrics
    return nested


def derive_run_key(log_path: str, config: object | None = None) -> str:
    normalized = os.path.normpath(log_path)
    parts = normalized.split(os.sep)
    try:
        logs_index = parts.index("logs")
        run_name = parts[logs_index - 2]
        timestamp = parts[logs_index - 1]
        if run_name and timestamp:
            seed_suffix = _seed_suffix(config if isinstance(config, dict) else None, log_path)
            return f"{run_name}/{timestamp}{seed_suffix}"
    except (ValueError, IndexError):
        pass
    return log_path


def load_run_config(log_path: str) -> Dict[str, object] | None:
    logs_dir = os.path.dirname(log_path)
    run_dir = os.path.dirname(logs_dir)
    config_path = os.path.join(run_dir, "config.json")
    try:
        with open(config_path, "r", encoding="utf-8") as handle:
            config = json.load(handle)
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return None
    return config if isinstance(config, dict) else None


def _seed_suffix(config: Dict[str, object] | None, log_path: str) -> str:
    if config is None:
        config = load_run_config(log_path)

    seed = None
    if isinstance(config, dict):
        train_cfg = config.get("train")
        if isinstance(train_cfg, dict):
            seed = train_cfg.get("seed")

    if seed is None:
        return ""

    return f"_seed{seed}"


def extract_data():
    rows: List[Dict[str, object]] = []
    if not os.path.exists(ROOT_DIR):
        raise NameError(f"Root path not found: {ROOT_DIR}")

    for path in iter_log_paths(ROOT_DIR):
        metrics = parse_metrics(path)
        rows.extend(metrics)

    if not rows:
        raise ValueError("No metrics found.")

    nested = build_nested(rows)
    return nested


def generate_plots(
    data: Dict[str, Dict[str, Dict[str, Dict[str, object]]]],
    output_dir: str = DEFAULT_OUTPUT_DIR,
    show: bool = False,
) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError("matplotlib is required for plotting") from exc

    os.makedirs(output_dir, exist_ok=True)

    grouped_runs: Dict[str, Dict[str, Dict[str, Dict[str, Dict[str, object]]]]] = {}
    for run_key, splits in data.items():
        run_name = run_key.split("/", 1)[0]
        task_name, algo_name = _derive_task_and_algo(run_name)
        grouped_runs.setdefault(task_name, {}).setdefault(algo_name, {})[run_key] = splits

    for task_name, algo_runs in sorted(grouped_runs.items()):
        for algo_name, runs in sorted(algo_runs.items()):
            for split_name in ["train", "validation", "rollout"]:
                metric_names: List[str] = []
                metric_set = set()
                for splits in runs.values():
                    for metrics in splits.get(split_name, {}).values():
                        if isinstance(metrics, dict):
                            for key, value in metrics.items():
                                if _coerce_float(value) is not None and key not in metric_set:
                                    metric_set.add(key)
                                    metric_names.append(key)

                if not metric_names:
                    continue

                cols = 2 if len(metric_names) > 1 else 1
                rows = math.ceil(len(metric_names) / cols)
                fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 3.5 * rows), squeeze=False)
                fig.suptitle(f"{task_name} - {algo_name} - {split_name}", fontsize=14)

                for idx, metric_name in enumerate(metric_names):
                    axis = axes[idx // cols][idx % cols]
                    for run_key, splits in sorted(runs.items()):
                        epochs = []
                        values = []
                        for epoch_str, metrics in splits.get(split_name, {}).items():
                            if not isinstance(metrics, dict):
                                continue
                            if metric_name not in metrics:
                                continue
                            value = _coerce_float(metrics[metric_name])
                            if value is None:
                                continue
                            try:
                                epoch = int(epoch_str)
                            except ValueError:
                                continue
                            epochs.append(epoch)
                            values.append(value)

                        if epochs:
                            ordered = sorted(zip(epochs, values), key=lambda pair: pair[0])
                            epochs_sorted, values_sorted = zip(*ordered)
                            label = run_key.split("/", 1)[1] if "/" in run_key else run_key
                            axis.plot(epochs_sorted, values_sorted, marker="o", markersize=2, label=label)

                    axis.set_title(metric_name)
                    axis.set_xlabel("Epoch")
                    axis.set_ylabel(metric_name)
                    axis.grid(True, alpha=0.3)
                    axis.legend(fontsize=8, loc="best")

                for extra in range(len(metric_names), rows * cols):
                    axes[extra // cols][extra % cols].axis("off")

                fig.tight_layout(rect=[0, 0, 1, 0.95])
                split_dir = os.path.join(output_dir, split_name, _sanitize_filename(task_name))
                os.makedirs(split_dir, exist_ok=True)
                filename = f"{_sanitize_filename(algo_name)}.png"
                fig_path = os.path.join(split_dir, filename)
                fig.savefig(fig_path, dpi=150)
                if show:
                    plt.show()
                plt.close(fig)


def _derive_task_and_algo(run_name: str) -> tuple[str, str]:
    prefix = "temporaldp_"
    suffix = "_mh_image"
    if run_name.startswith(prefix) and run_name.endswith(suffix):
        stem = run_name[len(prefix):-len(suffix)]
        parts = stem.split("_")
        if len(parts) >= 2:
            task_name = parts[-1]
            algo_name = "_".join(parts[:-1])
            if task_name and algo_name:
                return task_name, algo_name
    return run_name, run_name


def _coerce_float(value: object) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return None
    return None


def _sanitize_filename(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value)


if __name__ == "__main__":
    data = extract_data()
    generate_plots(data, output_dir=DEFAULT_OUTPUT_DIR, show=False)
    # print(data)
