#!/usr/bin/env python3
"""Interactively launch TemporalDP training and checkpoint evaluations.

Run from the repository root. The launcher selects ``robomimic2`` for standard
Robomimic experiments and ``robomimic2_temporalenvs_mujoco350`` for
WaitAtGoal and LiftQA experiments:

    python robomimic/scripts/experiment_cli.py
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import itertools
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import time
from typing import Sequence, TypeVar


REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_ROOT = REPO_ROOT / "robomimic" / "exps" / "temporaldp"
PACKAGE_ROOT = REPO_ROOT / "robomimic"
BASE_CONDA_ENV = "robomimic2"
# LiftQA demonstrations were collected with MuJoCo 3.5.0. Use the isolated
# matching environment for all temporal training and evaluation rollouts.
TEMPORAL_CONDA_ENV = "robomimic2_temporalenvs_mujoco350"
T = TypeVar("T")
SCREEN_SESSION_SEQUENCE = itertools.count()


@dataclass(frozen=True)
class Experiment:
    environment: str
    algorithm: str
    config_path: Path
    split: str | None = None

    @property
    def is_temporal(self) -> bool:
        return self.environment in {"waitatgoal", "liftqa"}

    @property
    def label(self) -> str:
        split = f" / {self.split}" if self.split else ""
        return f"{self.algorithm}{split}{self.conditioning_label}"

    @property
    def conditioning_label(self) -> str:
        """Compact temporal-conditioning label read from the experiment config."""
        try:
            with self.config_path.open(encoding="utf-8") as file:
                temporal = json.load(file).get("observation", {}).get("temporal_encodings", {})
        except (FileNotFoundError, json.JSONDecodeError, OSError):
            return ""

        features = []
        if temporal.get("use_idleness_encoding", False):
            features.append("ID")
        if temporal.get("use_sinusoidal_progress_encoding", False):
            features.append("SiP")
        if temporal.get("use_saturating_progress_encoding", False):
            features.append("SaP")
        return f" [{', '.join(features)}]" if features else ""


@dataclass(frozen=True)
class Checkpoint:
    path: Path
    run_id: str
    success: float | None
    epoch: int | None

    @property
    def label(self) -> str:
        details = []
        if self.success is not None:
            details.append(f"best success={self.success:.2f}")
        if self.epoch is not None:
            details.append(f"epoch={self.epoch}")
        suffix = f" ({', '.join(details)})" if details else ""
        return f"{self.run_id}{suffix} / {self.path.name}"


@dataclass(frozen=True)
class ExecutionMode:
    """How a command should be started by the launcher."""

    use_screen: bool = False


def prompt_execution_mode() -> ExecutionMode | None:
    mode = prompt_choice(
        "Launch mode",
        ["Foreground", "Detached screen session"],
        str,
    )
    if mode is None:
        return None
    return ExecutionMode(use_screen=mode == "Detached screen session")


def prompt_choice(title: str, options: Sequence[T], label) -> T | None:
    if not options:
        print(f"No {title.lower()} available.")
        return None

    print(f"\n{title}")
    for index, option in enumerate(options, start=1):
        print(f"  {index:>2}. {label(option)}")
    print("   0. Cancel")

    while True:
        answer = input("Select an option: ").strip()
        if answer == "0":
            return None
        try:
            selected = int(answer)
        except ValueError:
            print("Enter an option number.")
            continue
        if 1 <= selected <= len(options):
            return options[selected - 1]
        print(f"Choose a number from 0 to {len(options)}.")


def prompt_multiple_choices(title: str, options: Sequence[T], label) -> list[T] | None:
    """Select one or more options, retaining the order entered by the user."""
    if not options:
        print(f"No {title.lower()} available.")
        return None

    print(f"\n{title}")
    for index, option in enumerate(options, start=1):
        print(f"  {index:>2}. {label(option)}")
    print("   0. Cancel")

    while True:
        answer = input("Select one or more options, comma-separated: ").strip()
        if answer == "0":
            return None
        try:
            indices = [int(index.strip()) for index in answer.split(",")]
        except ValueError:
            print("Enter one or more comma-separated option numbers.")
            continue
        if not indices or len(set(indices)) != len(indices):
            print("Choose one or more unique option numbers, or 0 to cancel.")
            continue
        if any(not 1 <= index <= len(options) for index in indices):
            print(f"Choose numbers from 1 to {len(options)}, or 0 to cancel.")
            continue
        return [options[index - 1] for index in indices]


def prompt_text(label: str, default: str | None = None, validator=None) -> str:
    suffix = f" [{default}]" if default is not None else ""
    while True:
        answer = input(f"{label}{suffix}: ").strip()
        value = answer or default
        if value is None:
            print("A value is required.")
            continue
        if validator is not None and not validator(value):
            print("That value is not valid.")
            continue
        return value


def print_gpu_memory_summary() -> None:
    """Print a best-effort GPU memory snapshot before asking for a device."""
    nvidia_smi = shutil.which("nvidia-smi")
    if nvidia_smi is None:
        print("GPU memory: nvidia-smi not found.")
        return
    command = [
        nvidia_smi,
        "--query-gpu=index,memory.used,memory.total",
        "--format=csv,noheader,nounits",
    ]
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    if result.returncode:
        print("GPU memory: unavailable.")
        return
    summary = []
    for line in result.stdout.splitlines():
        try:
            index, used, total = (value.strip() for value in line.split(","))
            used_mib, total_mib = int(used), int(total)
            percent = 100.0 * used_mib / total_mib if total_mib else 0.0
            summary.append(f"GPU {index}: {used_mib}/{total_mib} MiB ({percent:.0f}%)")
        except ValueError:
            continue
    print("GPU memory: " + (" | ".join(summary) if summary else "unavailable."))


def prompt_gpu() -> str:
    print_gpu_memory_summary()
    return prompt_text("GPU", default="0", validator=lambda value: bool(value))


def prompt_yes_no(question: str, default: bool = False) -> bool:
    marker = "Y/n" if default else "y/N"
    while True:
        answer = input(f"{question} [{marker}]: ").strip().lower()
        if not answer:
            return default
        if answer in {"y", "yes"}:
            return True
        if answer in {"n", "no"}:
            return False
        print("Please answer y or n.")


def parse_seeds(value: str) -> list[str]:
    seeds = [seed.strip() for seed in value.split(",")]
    if not seeds or any(not seed.isdigit() for seed in seeds):
        raise ValueError("Seeds must be comma-separated non-negative integers.")
    return seeds


def prompt_seeds(label: str, default: str) -> list[str]:
    while True:
        value = prompt_text(label, default=default)
        try:
            return parse_seeds(value)
        except ValueError as exc:
            print(exc)


def discover_experiments() -> list[Experiment]:
    experiments: list[Experiment] = []
    for config_path in CONFIG_ROOT.rglob("*.json"):
        relative = config_path.relative_to(CONFIG_ROOT)
        parts = relative.parts
        if len(parts) == 4 and parts[2] == "image":
            environment, split, _, filename = parts
            experiments.append(Experiment(environment, config_path.stem, config_path, split))
        elif len(parts) == 3 and parts[0] == "temporal":
            _, environment, filename = parts
            experiments.append(Experiment(environment, config_path.stem, config_path))
    return sorted(experiments, key=lambda item: (item.environment, item.split or "", item.algorithm))


def select_experiment(experiments: Sequence[Experiment]) -> Experiment | None:
    environments = sorted({experiment.environment for experiment in experiments})
    environment = prompt_choice("Environment", environments, str)
    if environment is None:
        return None
    candidates = [experiment for experiment in experiments if experiment.environment == environment]
    candidates.sort(key=lambda experiment: experiment.algorithm)
    return prompt_choice("Algorithm and dataset split", candidates, lambda experiment: experiment.label)


def load_config(experiment: Experiment) -> dict:
    with experiment.config_path.open() as file:
        return json.load(file)


def checkpoint_action_normalization(checkpoint: Checkpoint) -> str | None:
    """Read the immutable run config without unpickling the checkpoint."""
    config_path = checkpoint.path.parent.parent / "config.json"
    try:
        with config_path.open() as file:
            return json.load(file)["train"]["action_config"]["actions"].get("normalization")
    except (FileNotFoundError, KeyError, TypeError, json.JSONDecodeError):
        return None


def checkpoint_root(experiment: Experiment) -> Path:
    config = load_config(experiment)
    output_dir = Path(config["train"]["output_dir"]).expanduser()
    if not output_dir.is_absolute():
        output_dir = PACKAGE_ROOT / output_dir
    return output_dir.resolve() / config["experiment"]["name"]


def discover_checkpoints(experiment: Experiment) -> list[Checkpoint]:
    root = checkpoint_root(experiment)
    checkpoints_by_run: dict[str, list[Checkpoint]] = {}
    for path in root.glob("*/models/*.pth"):
        success_match = re.search(r"success_([0-9]+(?:\.[0-9]+)?)", path.name)
        epoch_match = re.search(r"epoch_(\d+)", path.name)
        checkpoint = Checkpoint(
            path=path,
            run_id=path.parent.parent.name,
            success=float(success_match.group(1)) if success_match else None,
            epoch=int(epoch_match.group(1)) if epoch_match else None,
        )
        checkpoints_by_run.setdefault(checkpoint.run_id, []).append(checkpoint)

    def checkpoint_rank(checkpoint: Checkpoint) -> tuple[bool, float, int, float]:
        return (
            checkpoint.success is not None,
            checkpoint.success if checkpoint.success is not None else float("-inf"),
            checkpoint.epoch if checkpoint.epoch is not None else -1,
            checkpoint.path.stat().st_mtime,
        )

    best_checkpoints = [
        max(run_checkpoints, key=checkpoint_rank)
        for run_checkpoints in checkpoints_by_run.values()
    ]
    return sorted(best_checkpoints, key=lambda item: item.run_id, reverse=True)


def conda_executable() -> str:
    candidates = [
        os.environ.get("CONDA_EXE"),
        shutil.which("conda"),
        str(Path.home() / "miniconda3" / "bin" / "conda"),
    ]
    for candidate in candidates:
        if candidate and Path(candidate).is_file():
            return candidate
    raise FileNotFoundError("Could not find conda. Set CONDA_EXE or add conda to PATH.")


def make_screen_session(label: str) -> str:
    """Return a screen-safe, human-readable session name."""
    safe_label = re.sub(r"[^A-Za-z0-9_.-]+", "-", label).strip("-_") or "job"
    # A batch launch shares one Python process and often creates every screen
    # within the same second. The sequence prevents ambiguous duplicate names.
    suffix = f"-{time.strftime('%Y%m%d-%H%M%S')}-{os.getpid()}-{next(SCREEN_SESSION_SEQUENCE)}"
    # GNU screen limits session names. Keep enough room for the fixed prefix
    # and unique suffix instead of allowing long conditioning-variant labels
    # to make detached launches fail.
    max_length = 64
    label_budget = max_length - len("robomimic-") - len(suffix)
    return f"robomimic-{safe_label[:label_budget]}{suffix}"


def start_screen_session(session: str, command: str, dry_run: bool) -> int:
    """Start ``command`` in a detached, login-interactive screen shell."""
    screen = shutil.which("screen")
    if screen is None:
        print("screen was not found on PATH; cannot launch a detached session.")
        return 127

    script_path = Path("/tmp") / f"{session}.sh"
    print(f"\nScreen session: {session}")
    print(f"Attach with: screen -r {session}")
    print(f"Command: {command}")
    if dry_run:
        print("Dry run: screen session not launched.")
        return 0

    script_path.write_text(
        "#!/usr/bin/env bash\n"
        "shopt -s expand_aliases\n"
        "source ~/.bashrc 2>/dev/null || true\n"
        f"{command}\n"
        "exit_code=$?\n"
        "if [ \"$exit_code\" -eq 0 ]; then\n"
        "    exit 0\n"
        "fi\n"
        "echo \"Command failed with status $exit_code; leaving this screen session open.\"\n",
        encoding="utf-8",
    )
    script_path.chmod(0o755)

    screen_cmd = [
        screen,
        "-T", "screen-256color",
        "-S", session,
        "-dm",
        "bash",
        "-l",
        "-i",
    ]
    exit_code = subprocess.run(screen_cmd, cwd=REPO_ROOT, check=False).returncode
    if exit_code:
        print(f"Could not create screen session {session} (status {exit_code}).")
        return exit_code

    payload = f"source {script_path}\r"
    stuff_cmd = [
        screen,
        "-S", session,
        "-p", "0",
        "-X", "stuff",
        payload,
    ]
    exit_code = subprocess.run(stuff_cmd, cwd=REPO_ROOT, check=False).returncode
    if exit_code:
        print(f"Could not send the command to screen session {session} (status {exit_code}).")
    return exit_code


def command_with_runtime(
    command: list[str],
    gpu: str,
    temporal: bool,
) -> tuple[list[str], dict[str, str], str]:
    environment = os.environ.copy()
    environment["CUDA_VISIBLE_DEVICES"] = gpu
    environment["MUJOCO_GL"] = "egl"
    # The packaged Robosuite installation enables Numba's file cache, but its
    # source location is not cacheable on the shared runtime filesystem.
    environment["NUMBA_DISABLE_JIT"] = "1"
    if temporal:
        environment["SDL_VIDEODRIVER"] = "dummy"
    conda_env = TEMPORAL_CONDA_ENV if temporal else BASE_CONDA_ENV
    runtime_command = [conda_executable(), "run", "--no-capture-output", "-n", conda_env, *command]

    env_prefix = " ".join(
        f"{name}={shlex.quote(environment[name])}"
        for name in ("CUDA_VISIBLE_DEVICES", "MUJOCO_GL", "SDL_VIDEODRIVER", "NUMBA_DISABLE_JIT")
        if name in environment
    )
    return runtime_command, environment, f"{env_prefix} {shlex.join(runtime_command)}"


def launch(
    command: list[str],
    gpu: str,
    temporal: bool,
    dry_run: bool,
    confirm: bool = True,
    execution: ExecutionMode = ExecutionMode(),
    label: str = "job",
) -> int:
    runtime_command, environment, printable_command = command_with_runtime(command, gpu, temporal)
    print("\nCommand:")
    print(f"  {printable_command}")
    if dry_run:
        print("Dry run: command not launched.")
        return 0
    if confirm and not prompt_yes_no("Launch this command?", default=False):
        print("Skipped.")
        return 0
    if execution.use_screen:
        screen_command = f"cd {shlex.quote(str(REPO_ROOT))} && {printable_command}"
        return start_screen_session(make_screen_session(label), screen_command, dry_run=False)
    return subprocess.run(runtime_command, cwd=REPO_ROOT, env=environment, check=False).returncode


def run_progress_script(script_name: str) -> None:
    command = [
        conda_executable(),
        "run",
        "--no-capture-output",
        "-n",
        BASE_CONDA_ENV,
        "python",
        f"robomimic/scripts/{script_name}",
    ]
    print(f"\nRunning: {shlex.join(command)}")
    exit_code = subprocess.run(command, cwd=REPO_ROOT, check=False).returncode
    if exit_code:
        print(f"{script_name} exited with status {exit_code}.")


def run_analysis(dry_run: bool) -> None:
    """Run analysis for one standard PH task or one temporal dataset."""
    environment = prompt_choice(
        "Analysis environment",
        ["lift", "can", "square", "temporal"],
        str,
    )
    if environment is None:
        return

    temporal_split = None
    if environment == "temporal":
        temporal_split = prompt_choice(
            "Temporal dataset",
            ["waitatgoal", "liftqa"],
            str,
        )
        if temporal_split is None:
            return

    command = [
        "python",
        "robomimic/scripts/analysis_temporal.py",
        "--task_names",
        f"[{environment}]",
        "--dataset_split",
        "ph",
    ]
    if temporal_split is not None:
        command.extend(["--temporal_dataset_split", temporal_split])
    execution = prompt_execution_mode()
    if execution is None:
        return
    exit_code = launch(
        command,
        gpu="0",
        temporal=temporal_split is not None,
        dry_run=dry_run,
        confirm=True,
        execution=execution,
        label=f"analysis-{temporal_split or environment}",
    )
    if exit_code:
        print(f"analysis_temporal.py exited with status {exit_code}.")


def run_training(experiments: Sequence[Experiment], dry_run: bool) -> None:
    experiment = select_experiment(experiments)
    if experiment is None:
        return
    gpu = prompt_gpu()
    seeds = prompt_seeds("Training seed(s), comma-separated", default="1")
    execution = prompt_execution_mode()
    if execution is None:
        return
    if len(seeds) > 1 and not dry_run:
        if not prompt_yes_no(f"Launch {len(seeds)} training jobs sequentially?", default=False):
            print("Skipped.")
            return

    # A detached screen returns as soon as it starts, so launching each seed
    # separately would run them concurrently. Put all commands in one screen
    # script instead: each ``conda run`` blocks before the next seed begins.
    if execution.use_screen and len(seeds) > 1:
        session = make_screen_session(f"train-{experiment.environment}-{experiment.algorithm}-batch")
        batch_lines = [
            "source ~/.bashrc 2>/dev/null || true",
            "shopt -s expand_aliases",
        ]
        for seed in seeds:
            command = [
                "python",
                "robomimic/scripts/train.py",
                "--config",
                str(experiment.config_path.relative_to(REPO_ROOT)),
                "--seed",
                seed,
            ]
            _, _, printable_command = command_with_runtime(command, gpu, experiment.is_temporal)
            print("\nCommand:")
            print(f"  {printable_command}")
            alert_prefix = (
                "RoboMimic training failed: "
                f"environment={experiment.environment}, algorithm={experiment.algorithm}, "
                f"seed={seed}, gpu={gpu}"
            )
            # Do not proceed to another seed after a failure. ``eval`` parses
            # the alias after .bashrc has been sourced, so it can invoke the
            # user's ``vla_alert`` alias from this non-interactive shell.
            batch_lines.extend(
                [
                    printable_command,
                    "status=$?",
                    'if [ "$status" -ne 0 ]; then',
                    f"  alert_message={shlex.quote(alert_prefix)}",
                    '  alert_message="${alert_message}; exit status ${status}"',
                    "  eval 'vla_alert \"$alert_message\"' || true",
                    '  exit "$status"',
                    "fi",
                ]
            )
        batch_script = "\n".join(batch_lines)
        screen_command = (
            f"cd {shlex.quote(str(REPO_ROOT))} && "
            f"bash -c {shlex.quote(batch_script)}"
        )
        exit_code = start_screen_session(session, screen_command, dry_run=dry_run)
        if exit_code:
            print(f"Training batch exited with status {exit_code}.")
        return

    for seed in seeds:
        command = [
            "python",
            "robomimic/scripts/train.py",
            "--config",
            str(experiment.config_path.relative_to(REPO_ROOT)),
            "--seed",
            seed,
        ]
        exit_code = launch(
            command,
            gpu,
            experiment.is_temporal,
            dry_run,
            confirm=len(seeds) == 1,
            execution=execution,
            label=f"train-{experiment.environment}-{experiment.algorithm}-seed{seed}",
        )
        if exit_code:
            print(f"Training seed {seed} exited with status {exit_code}.")


def rollout_output_path(experiment: Experiment, checkpoint: Checkpoint, seed: str, multi_seed: bool) -> Path:
    if experiment.is_temporal:
        parent = REPO_ROOT / "rollouts" / "temporal" / experiment.environment / experiment.algorithm
    else:
        assert experiment.split is not None
        parent = REPO_ROOT / "rollouts" / experiment.environment / experiment.split / experiment.algorithm
    seed_suffix = f"_seed{seed}" if multi_seed else ""
    return parent / f"{checkpoint.run_id}_{checkpoint.path.stem}{seed_suffix}.hdf5"


def write_eval_info(
    info_path: Path,
    experiment: Experiment,
    checkpoint: Checkpoint,
    dataset_path: Path,
    gpu: str,
    seed: str,
    n_rollouts: str,
) -> None:
    info_path.parent.mkdir(parents=True, exist_ok=True)
    fields = {
        "RUN_ID": checkpoint.run_id,
        "SEED": seed,
        "N_ROLLOUTS": n_rollouts,
        "ENV": experiment.environment,
        "ALGO": experiment.algorithm,
        "GPU": gpu,
        "SPLIT": experiment.split or "temporal",
        "AGENT": str(checkpoint.path),
        "DATASET": str(dataset_path),
    }
    info_path.write_text("".join(f"{key}={value}\n" for key, value in fields.items()))


def archive_evaluation_rollout(
    dataset_path: Path,
    gpu: str,
    temporal: bool,
    dry_run: bool,
) -> int:
    """Compress a completed rollout and reclaim its HDF5 image storage."""
    command = [
        "python",
        "robomimic/scripts/rollout_hdf_to_video.py",
        str(dataset_path),
        "--overwrite",
        "--delete-images",
        "--delete-lang-emb",
    ]
    return launch(
        command,
        gpu,
        temporal,
        dry_run,
        confirm=False,
        label=f"archive-{dataset_path.stem}",
    )


def run_evaluation(experiments: Sequence[Experiment], dry_run: bool) -> None:
    experiment = select_experiment(experiments)
    if experiment is None:
        return
    checkpoints = prompt_multiple_choices(
        "Timestamped runs (best checkpoint selected automatically)",
        discover_checkpoints(experiment),
        lambda item: item.label,
    )
    if checkpoints is None:
        return
    if experiment.is_temporal and any(
        checkpoint_action_normalization(checkpoint) is None
        for checkpoint in checkpoints
    ):
        print(
            "\nWarning: this temporal configuration uses unnormalised coordinate actions. "
            "Existing checkpoints trained from it commonly saturate near [1, 1] and "
            "collide after a few steps. This evaluation will still run and be written, "
            "but retrain with the updated min_max configuration for meaningful rollouts."
        )
    gpu = prompt_gpu()
    default_seed = "1000"
    seeds = prompt_seeds("Evaluation seed(s), comma-separated", default=default_seed)
    default_n_rollouts = "200" if experiment.is_temporal else "100"
    n_rollouts = prompt_text(
        "Number of rollouts",
        default=default_n_rollouts,
        validator=lambda value: value.isdigit() and int(value) > 0,
    )
    execution = prompt_execution_mode()
    if execution is None:
        return
    if not dry_run and not prompt_yes_no(
        f"Write metadata and launch {len(checkpoints) * len(seeds)} evaluation job(s) sequentially?", default=False
    ):
        print("Skipped.")
        return

    plan = {
        "experiment": experiment_to_plan(experiment),
        "checkpoint_paths": [str(checkpoint.path) for checkpoint in checkpoints],
        "gpu": gpu,
        "evaluation_seeds": seeds,
        "n_rollouts": n_rollouts,
        "archive_rollout_images": True,
    }
    if not execution.use_screen:
        run_evaluation_plan(plan, dry_run=dry_run)
        return

    session = make_screen_session(f"eval-{experiment.environment}-{experiment.algorithm}")
    plan_path = Path("/tmp") / f"{session}.json"
    if not dry_run:
        plan_path.write_text(json.dumps(plan, indent=2) + "\n", encoding="utf-8")
    coordinator_command = [
        conda_executable(),
        "run",
        "--no-capture-output",
        "-n",
        BASE_CONDA_ENV,
        "python",
        "robomimic/scripts/experiment_cli.py",
        "--run-evaluation-plan",
        str(plan_path),
    ]
    command = f"cd {shlex.quote(str(REPO_ROOT))} && {shlex.join(coordinator_command)}"
    start_screen_session(session, command, dry_run=dry_run)


def experiment_to_plan(experiment: Experiment) -> dict:
    return {
        "environment": experiment.environment,
        "algorithm": experiment.algorithm,
        "config_path": str(experiment.config_path.relative_to(REPO_ROOT)),
        "split": experiment.split,
    }


def experiment_from_plan(plan: dict) -> Experiment:
    experiment_data = plan["experiment"]
    config_path = Path(experiment_data["config_path"])
    if not config_path.is_absolute():
        config_path = REPO_ROOT / config_path
    config_path = config_path.resolve()
    try:
        config_path.relative_to(CONFIG_ROOT)
    except ValueError as exc:
        raise ValueError(f"Evaluation plan config is outside {CONFIG_ROOT}: {config_path}") from exc
    return Experiment(
        environment=experiment_data["environment"],
        algorithm=experiment_data["algorithm"],
        config_path=config_path,
        split=experiment_data.get("split"),
    )


def run_evaluation_plan(plan: dict, dry_run: bool) -> None:
    """Evaluate selected checkpoints in their plan order, one job at a time."""
    experiment = experiment_from_plan(plan)
    available_checkpoints = {
        str(checkpoint.path.resolve()): checkpoint
        for checkpoint in discover_checkpoints(experiment)
    }
    checkpoint_paths = [str(Path(path).resolve()) for path in plan["checkpoint_paths"]]
    checkpoints = []
    for checkpoint_path in checkpoint_paths:
        checkpoint = available_checkpoints.get(checkpoint_path)
        if checkpoint is None:
            raise FileNotFoundError(
                "Selected checkpoint is no longer available as the best checkpoint for this run: "
                f"{checkpoint_path}"
            )
        checkpoints.append(checkpoint)

    gpu = str(plan["gpu"])
    seeds = [str(seed) for seed in plan["evaluation_seeds"]]
    n_rollouts = str(plan["n_rollouts"])
    archive_rollout_images = bool(plan.get("archive_rollout_images", False))
    if not checkpoints or not seeds:
        raise ValueError("Evaluation plan must contain at least one checkpoint and one seed.")

    print(
        f"\nEvaluation workflow: {len(checkpoints)} selected run(s), "
        f"{len(seeds)} seed(s) each, executed sequentially."
    )
    for checkpoint in checkpoints:
        for seed in seeds:
            dataset_path = rollout_output_path(experiment, checkpoint, seed, multi_seed=len(seeds) > 1)
            info_path = dataset_path.with_name(f"{dataset_path.stem}_info.txt")
            command = [
                "python",
                "robomimic/scripts/run_trained_agent.py",
                "--agent",
                str(checkpoint.path),
                "--dataset_path",
                str(dataset_path),
                "--n_rollouts",
                n_rollouts,
                "--seed",
                seed,
                "--dataset_obs",
            ]
            if not dry_run:
                dataset_path.parent.mkdir(parents=True, exist_ok=True)
                write_eval_info(info_path, experiment, checkpoint, dataset_path, gpu, seed, n_rollouts)
            exit_code = launch(
                command,
                gpu,
                experiment.is_temporal,
                dry_run,
                confirm=False,
                label=f"eval-{experiment.environment}-{experiment.algorithm}-{checkpoint.run_id}-seed{seed}",
            )
            if exit_code:
                print(f"Evaluation for {checkpoint.run_id}, seed {seed} exited with status {exit_code}.")
                continue
            if archive_rollout_images:
                archive_exit_code = archive_evaluation_rollout(
                    dataset_path,
                    gpu,
                    experiment.is_temporal,
                    dry_run,
                )
                if archive_exit_code:
                    print(
                        f"Image archival for {checkpoint.run_id}, seed {seed} exited with status "
                        f"{archive_exit_code}; rollout images were retained."
                    )


def select_progress_runs_for_deletion():
    # Reuse the progress report's own collection, sorting, and status logic so
    # the deletion menu reports the same state as ``check_progress.py``.
    from check_progress import collect_progress, format_progress, run_status, sort_key

    root = REPO_ROOT / "trained_models"
    runs = sorted(collect_progress(root), key=sort_key)
    if not runs:
        print(f"No timestamped training runs found under {root}.")
        return None

    now = time.time()
    active_within_seconds = 300
    print("\nTimestamped training runs (active runs cannot be deleted)")
    for index, run in enumerate(runs, start=1):
        print(f"  {index:>2}. {format_progress(run, width=16, now=now, active_within_seconds=active_within_seconds)}")
    print("   0. Cancel")

    while True:
        answer = input("Select run(s) to delete, comma-separated: ").strip()
        if answer == "0":
            return None
        try:
            indices = [int(index.strip()) for index in answer.split(",")]
        except ValueError:
            print("Enter one or more comma-separated option numbers.")
            continue
        if not indices or len(set(indices)) != len(indices) or any(not 1 <= index <= len(runs) for index in indices):
            print(f"Choose unique numbers from 1 to {len(runs)}, or 0 to cancel.")
            continue
        selected_runs = [runs[index - 1] for index in indices]
        active_runs = [
            run for run in selected_runs
            if run_status(run, now=now, active_within_seconds=active_within_seconds) == "active"
        ]
        if active_runs:
            print(
                "Active run(s) cannot be deleted: "
                + ", ".join(run.run_dir.name for run in active_runs)
            )
            return None
        return selected_runs


def delete_run_directories(dry_run: bool) -> None:
    runs = select_progress_runs_for_deletion()
    if runs is None:
        return
    root = (REPO_ROOT / "trained_models").resolve()
    targets = [run.run_dir.resolve() for run in runs]
    for target in targets:
        try:
            target.relative_to(root)
        except ValueError as exc:
            raise ValueError(f"Refusing to delete a path outside trained_models: {target}") from exc
        if not target.is_dir() or not target.name.isdigit() or len(target.name) != 14:
            raise ValueError(f"Refusing to delete a non-timestamped run directory: {target}")

    timestamps = ",".join(target.name for target in targets)
    confirmation = prompt_text(
        f"Type {timestamps} to permanently delete the selected runs",
        default=None,
    )
    if confirmation != timestamps:
        print("Timestamp did not match; nothing was deleted.")
        return
    if dry_run:
        print("Dry run: would delete " + ", ".join(str(target) for target in targets))
        return
    for target in targets:
        shutil.rmtree(target)
        print(f"Deleted {target}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true", help="Print commands without launching them or writing metadata.")
    parser.add_argument(
        "--run-evaluation-plan",
        type=Path,
        help=argparse.SUPPRESS,
    )
    args = parser.parse_args()

    if args.run_evaluation_plan is not None:
        with args.run_evaluation_plan.open(encoding="utf-8") as file:
            run_evaluation_plan(json.load(file), dry_run=args.dry_run)
        return

    experiments = discover_experiments()
    if not experiments:
        raise FileNotFoundError(f"No TemporalDP configs found under {CONFIG_ROOT}")

    try:
        while True:
            action = prompt_choice(
                "Action",
                [
                    "Train",
                    "Evaluate",
                    "Run analysis",
                    "Check training progress",
                    "Check rollout progress",
                    "Delete timestamped training run",
                ],
                str,
            )
            if action is None:
                return
            if action == "Train":
                run_training(experiments, args.dry_run)
            elif action == "Evaluate":
                run_evaluation(experiments, args.dry_run)
            elif action == "Run analysis":
                run_analysis(args.dry_run)
            elif action == "Check training progress":
                run_progress_script("check_progress.py")
            elif action == "Check rollout progress":
                run_progress_script("check_progress_rollout.py")
            else:
                delete_run_directories(args.dry_run)
    except (EOFError, KeyboardInterrupt):
        print("\nExited.")


if __name__ == "__main__":
    main()
