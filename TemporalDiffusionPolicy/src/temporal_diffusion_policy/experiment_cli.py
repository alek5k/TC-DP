#!/usr/bin/env python3
"""Interactively train a policy and evaluate it with matching conditioning."""

from __future__ import annotations

import ast
import importlib.util
import subprocess
import sys
import time
import os
import json
import re
import shlex
import shutil
from datetime import datetime, timedelta
from pathlib import Path


# Allow ``python src/temporal_diffusion_policy/experiment_cli.py`` during
# development, while normal installations use ``tdp`` or ``python -m``.
if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def _repository_root() -> Path:
    """Locate the checkout containing this module, if one is available.

    This makes ``tdp`` reliable when invoked from another working directory
    after an editable install.  A regular wheel has no checkout metadata, so
    it instead uses the current directory as that installation's workspace.
    """
    for candidate in Path(__file__).resolve().parents:
        if (candidate / "pyproject.toml").is_file():
            return candidate
    return Path.cwd()


REPO_ROOT = _repository_root()
DEFAULT_MAJOR_TRAINING_CONFIG = {
    "batch_size": 64,
    "num_epochs": 200,
    "lr": 1e-4,
}




def _demonstration_datasets() -> list[Path]:
    """Return legacy zarr demonstrations supported by TemporalConditioningDataset."""
    datasets: list[Path] = []
    for relative_root in ("data/demonstrations", "data/demonstration"):
        root = REPO_ROOT / relative_root
        if not root.is_dir():
            continue
        datasets.extend(
            path for path in root.iterdir()
            if path.is_dir() and path.suffix == ".zarr" and (path / "meta" / "episode_ends").exists()
        )
    return sorted(set(datasets))


def _prompt_choice(prompt: str, choices: list[Path]) -> Path:
    while True:
        print("\nAvailable demonstration datasets:")
        for index, dataset in enumerate(choices, start=1):
            print(f"  {index}. {dataset.relative_to(REPO_ROOT)}")
        answer = input(prompt).strip()
        try:
            return choices[int(answer) - 1]
        except (ValueError, IndexError):
            print(f"Enter a number from 1 to {len(choices)}.")


def _prompt_menu(prompt: str, choices: list[str]) -> str:
    while True:
        print()
        for index, choice in enumerate(choices, start=1):
            print(f"  {index}. {choice}")
        answer = input(prompt).strip()
        try:
            return choices[int(answer) - 1]
        except (ValueError, IndexError):
            print(f"Enter a number from 1 to {len(choices)}.")


def _prompt_int(prompt: str, default: int, minimum: int = 1) -> int:
    while True:
        answer = input(f"{prompt} [{default}]: ").strip()
        if not answer:
            return default
        try:
            value = int(answer)
            if value < minimum:
                raise ValueError
            return value
        except ValueError:
            print(f"Enter an integer greater than or equal to {minimum}.")


def _prompt_gpu() -> str:
    """Return a CUDA_VISIBLE_DEVICES value without requiring CUDA at CLI import."""
    while True:
        answer = input("GPU index (CUDA_VISIBLE_DEVICES) [0]: ").strip()
        if not answer:
            return "0"
        # CUDA_VISIBLE_DEVICES supports a comma-separated list; validate the
        # common numeric form while preserving the selected physical GPU IDs.
        gpu_ids = answer.split(",")
        if all(gpu_id.strip().isdigit() for gpu_id in gpu_ids):
            return ",".join(gpu_id.strip() for gpu_id in gpu_ids)
        print("Enter one GPU index (for example 0 or 1), or a comma-separated list.")


def print_gpu_memory_usage(gpu_devices: str) -> None:
    """Print used GPU memory as a percentage without blocking experiment launch."""
    command = [
        "nvidia-smi",
        "--query-gpu=index,memory.used,memory.total",
        "--format=csv,noheader,nounits",
    ]
    try:
        result = subprocess.run(command, capture_output=True, text=True, check=False)
    except FileNotFoundError:
        print("GPU memory usage unavailable: nvidia-smi was not found.")
        return
    if result.returncode:
        detail = result.stderr.strip() or f"status {result.returncode}"
        print(f"GPU memory usage unavailable: nvidia-smi returned {detail}.")
        return

    requested_devices = set(gpu_devices.split(","))
    usage_rows = []
    for line in result.stdout.splitlines():
        values = [value.strip() for value in line.split(",")]
        if len(values) != 3 or values[0] not in requested_devices:
            continue
        try:
            index, used, total = int(values[0]), float(values[1]), float(values[2])
        except ValueError:
            continue
        percent = 100.0 * used / total if total else 0.0
        usage_rows.append(f"GPU {index}: {percent:.1f}% ({used:.0f}/{total:.0f} MiB)")
    if usage_rows:
        print("GPU memory before launch: " + " | ".join(usage_rows))
    else:
        print(f"GPU memory usage unavailable for selected GPU(s): {gpu_devices}.")


def run_nvidia_smi() -> None:
    """Show the standard NVIDIA GPU status report in this terminal."""
    try:
        subprocess.run(["nvidia-smi"], cwd=REPO_ROOT, check=False)
    except FileNotFoundError:
        print("nvidia-smi was not found.")


def _prompt_float(prompt: str, default: float) -> float:
    while True:
        answer = input(f"{prompt} [{default:g}]: ").strip()
        if not answer:
            return default
        try:
            value = float(answer)
            if value <= 0:
                raise ValueError
            return value
        except ValueError:
            print("Enter a positive number.")


def _prompt_bool(prompt: str, default: bool = False) -> bool:
    default_label = "Y/n" if default else "y/N"
    while True:
        answer = input(f"{prompt} [{default_label}]: ").strip().lower()
        if not answer:
            return default
        if answer in {"y", "yes"}:
            return True
        if answer in {"n", "no"}:
            return False
        print("Enter y or n.")


def _as_draccus_bool(value: bool) -> str:
    return "True" if value else "False"


def _vla_alert_command(message: str) -> str:
    """Return a best-effort vla_alert command that cannot fail a run."""
    return f"(vla_alert {shlex.quote(message)} >/dev/null 2>&1 || true)"


def _completed_checkpoints() -> list[Path]:
    """Return checkpoint directories that can be evaluated immediately."""
    checkpoint_root = REPO_ROOT / "data" / "checkpoints"
    if not checkpoint_root.is_dir():
        return []
    return sorted(
        (
            checkpoint_dir for checkpoint_dir in checkpoint_root.iterdir()
            if checkpoint_dir.is_dir() and (checkpoint_dir / "final.pth").is_file()
        ),
        key=lambda checkpoint_dir: checkpoint_dir.name,
    )


def _prompt_checkpoint(choices: list[Path]) -> Path:
    while True:
        print("\nAvailable completed checkpoints:")
        for index, checkpoint in enumerate(choices, start=1):
            print(f"  {index}. {checkpoint.name}")
        answer = input("Select a checkpoint: ").strip()
        try:
            return choices[int(answer) - 1]
        except (ValueError, IndexError):
            print(f"Enter a number from 1 to {len(choices)}.")


def _infer_env_name_from_checkpoint(checkpoint_dir: Path) -> str | None:
    """Infer the environment from the current environment-prefixed path format."""
    known_env_names = ("LiftQA-Real", "WaitAtGoal", "LiftQA")
    for env_name in known_env_names:
        if checkpoint_dir.name.startswith(f"{env_name}_"):
            return env_name
    return None


def _prompt_env_name(checkpoint_dir: Path) -> str:
    inferred = _infer_env_name_from_checkpoint(checkpoint_dir)
    prompt = "Inference environment"
    if inferred is not None:
        prompt += f" [{inferred}]"
    while True:
        answer = input(f"{prompt}: ").strip()
        if not answer and inferred is not None:
            return inferred
        if answer:
            return answer
        print("Enter an environment name, or accept the inferred default.")


def _make_screen_session(label: str) -> str:
    """Return a screen-safe, human-readable detached-session name."""
    safe_label = re.sub(r"[^A-Za-z0-9_.-]+", "-", label).strip("-_") or "job"
    return f"temporaldp-{safe_label}-{time.strftime('%Y%m%d-%H%M%S')}-{os.getpid()}"


def _start_screen_session(session: str, command: str) -> int:
    """Start a detached login-interactive screen using the Robomimic pattern."""
    screen = shutil.which("screen")
    if screen is None:
        print("screen was not found on PATH; cannot launch the experiment.")
        return 127

    script_path = Path("/tmp") / f"{session}.sh"
    print(f"\nScreen session: {session}")
    print(f"Attach with: screen -r {session}")
    print(f"Command: {command}")
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

    screen_command = [screen, "-T", "screen-256color", "-S", session, "-dm", "bash", "-l", "-i"]
    exit_code = subprocess.run(screen_command, cwd=REPO_ROOT, check=False).returncode
    if exit_code:
        print(f"Could not create screen session {session} (status {exit_code}).")
        return exit_code

    stuff_command = [screen, "-S", session, "-p", "0", "-X", "stuff", f"source {script_path}\r"]
    exit_code = subprocess.run(stuff_command, cwd=REPO_ROOT, check=False).returncode
    if exit_code:
        print(f"Could not send the command to screen session {session} (status {exit_code}).")
    return exit_code


def run_train_and_evaluate() -> None:
    datasets = _demonstration_datasets()
    if not datasets:
        raise RuntimeError(
            "No supported zarr demonstrations found in data/demonstrations or "
            "data/demonstration."
        )

    dataset_path = _prompt_choice("Select a dataset: ", datasets)
    while True:
        num_images = _prompt_int("Number of cameras for DP (1 or 2)", default=1)
        if 1 <= num_images <= 2:
            break
        print("This DP implementation supports one or two cameras.")
    gpu_devices = _prompt_gpu()
    batch_size = _prompt_int("Batch size", default=64)
    learning_rate = _prompt_float("Learning rate", default=1e-4)
    num_epochs = _prompt_int("Training epochs", default=200)
    num_runs = _prompt_int("Sequential train/evaluate runs", default=1)
    training_seeds = [
        _prompt_int(f"Training random seed (run {index})", default=index, minimum=0)
        for index in range(1, num_runs + 1)
    ]
    inference_seed = _prompt_int("Inference random seed", default=1000, minimum=0)
    save_ema_weights = _prompt_bool("Save EMA weights", default=True)
    cond_predict_scale = _prompt_bool("Use conditional prediction scale", default=True)
    use_ddim_sampling = _prompt_bool("Use DDIM sampling for inference", default=False)
    temporal_mode = _prompt_menu(
        "Select temporal conditioning: ",
        [
            "None (standard Diffusion Policy)",
            "Handcrafted temporal features",
        ],
    )

    use_idleness = False
    use_direct_idleness = False
    use_sinusoidal = False
    sinusoidal_dim = 128
    use_saturating = False
    use_time_signal = False
    use_velocity = False
    direct_idleness_threshold = 0.03
    if temporal_mode.startswith("Handcrafted"):
        use_idleness = _prompt_bool("Use idleness encoding")
        use_direct_idleness = _prompt_bool(
            "Use direct binary idleness conditioning"
        )
        if use_direct_idleness:
            direct_idleness_threshold = _prompt_float(
                "Direct idleness threshold", default=0.03
            )
        use_sinusoidal = _prompt_bool("Use sinusoidal progress encoding")
        sinusoidal_dim = (
            _prompt_int("Sinusoidal encoding dimension", default=128)
            if use_sinusoidal else 128
        )
        use_saturating = _prompt_bool("Use saturating progress encoding")
        use_time_signal = _prompt_bool("Use temporal time signal")
        use_velocity = _prompt_bool("Use agent velocity conditioning")
    cache_images_on_gpu = _prompt_bool("Cache raw images on GPU", default=False)

    from temporal_diffusion_policy.env_mappings import get_env_settings_from_dataset_name

    env_name = get_env_settings_from_dataset_name(str(dataset_path)).env_name
    checkpoint_root = REPO_ROOT / "data" / "checkpoints"
    next_second = 0
    checkpoint_dirs = []
    for training_seed in training_seeds:
        while True:
            run_stamp = (datetime.now() + timedelta(seconds=next_second)).strftime(
                "%Y%m%d_%H%M%S"
            )
            next_second += 1
            checkpoint_dir = checkpoint_root / f"{env_name}_{run_stamp}_seed{training_seed}"
            if not checkpoint_dir.exists():
                checkpoint_dirs.append(checkpoint_dir)
                break
    dataset_arg = str(dataset_path.relative_to(REPO_ROOT))

    shared_policy_args = [
        "--num_images", str(num_images),
        "--cond_predict_scale", _as_draccus_bool(cond_predict_scale),
        "--use_ddim_sampling", _as_draccus_bool(use_ddim_sampling),
        "--use_idleness_encoding", _as_draccus_bool(use_idleness),
        "--use_direct_idleness_conditioning", _as_draccus_bool(use_direct_idleness),
        "--direct_idleness_threshold", str(direct_idleness_threshold),
        "--use_sinusoidal_progress_encoding", _as_draccus_bool(use_sinusoidal),
        "--sinusoidal_progress_encoding_dim", str(sinusoidal_dim),
        "--use_saturating_progress_encoding", _as_draccus_bool(use_saturating),
        "--use_temporal_time_signal", _as_draccus_bool(use_time_signal),
        "--use_agent_velocity_conditioning", _as_draccus_bool(use_velocity),
        "--cache_images_on_gpu", _as_draccus_bool(cache_images_on_gpu),
    ]
    run_commands = []
    for training_seed, checkpoint_dir in zip(training_seeds, checkpoint_dirs):
        training_command = [
            sys.executable, "-m", "temporal_diffusion_policy.training",
            "--dataset_path", dataset_arg,
            "--batch_size", str(batch_size),
            "--lr", str(learning_rate),
            "--num_epochs", str(num_epochs),
            "--seed", str(training_seed),
            "--save_ema_weights", _as_draccus_bool(save_ema_weights),
            "--override_checkpoint_name", checkpoint_dir.name,
            "--use_wandb", "True",
            *shared_policy_args,
        ]
        inference_command = [
            sys.executable, "-m", "temporal_diffusion_policy.inference",
            "--checkpoint_path", str(checkpoint_dir.relative_to(REPO_ROOT)),
            "--env_name", str(env_name),
            "--seed", str(inference_seed),
            *shared_policy_args,
        ]
        training_shell_command = (
            f"CUDA_VISIBLE_DEVICES={shlex.quote(gpu_devices)} "
            f"{shlex.join(training_command)}"
        )
        inference_shell_command = (
            f"CUDA_VISIBLE_DEVICES={shlex.quote(gpu_devices)} "
            f"{shlex.join(inference_command)}"
        )
        run_commands.append(" && ".join((
            training_shell_command,
            _vla_alert_command(f"Training finished: {checkpoint_dir.name}"),
            inference_shell_command,
            _vla_alert_command(
                f"Evaluation finished: {checkpoint_dir.name} ({env_name})"
            ),
        )))
    command = f"cd {shlex.quote(str(REPO_ROOT))} && " + " && ".join(run_commands)
    print_gpu_memory_usage(gpu_devices)
    exit_code = _start_screen_session(
        _make_screen_session(f"train-eval-{env_name}-x{num_runs}"), command
    )
    if exit_code:
        raise RuntimeError(f"Could not launch screen session (status {exit_code}).")


def run_evaluate_checkpoint() -> None:
    """Run inference for one existing checkpoint, optionally with PIP."""
    checkpoints = _completed_checkpoints()
    if not checkpoints:
        raise RuntimeError(
            "No completed checkpoints found under data/checkpoints (expected final.pth)."
        )

    checkpoint_dir = _prompt_checkpoint(checkpoints)
    env_name = _prompt_env_name(checkpoint_dir)
    gpu_devices = _prompt_gpu()
    inference_seed = _prompt_int("Inference random seed", default=1000, minimum=0)
    use_ddim_sampling = _prompt_bool("Use DDIM sampling for inference", default=False)
    use_pip = _prompt_bool("Enable Pause-Induced Perturbation (PIP)", default=False)

    pip_args: list[str] = [
        "--use_pause_induced_perturbation", _as_draccus_bool(use_pip),
    ]
    if use_pip:
        pip_epsilon = _prompt_float("PIP epsilon", default=0.03)
        pip_window_size = _prompt_int("PIP window size", default=20)
        pip_sigma = _prompt_float("PIP sigma", default=0.8)
        pip_args.extend([
            "--pip_epsilon", str(pip_epsilon),
            "--pip_window_size", str(pip_window_size),
            "--pip_sigma", str(pip_sigma),
        ])

    inference_command = [
        sys.executable,
        "-m",
        "temporal_diffusion_policy.inference",
        "--checkpoint_path",
        str(checkpoint_dir.relative_to(REPO_ROOT)),
        "--env_name",
        env_name,
        "--seed",
        str(inference_seed),
        "--use_ddim_sampling",
        _as_draccus_bool(use_ddim_sampling),
        *pip_args,
    ]
    command = (
        f"cd {shlex.quote(str(REPO_ROOT))} && "
        f"CUDA_VISIBLE_DEVICES={shlex.quote(gpu_devices)} {shlex.join(inference_command)} && "
        f"{_vla_alert_command(f'Evaluation finished: {checkpoint_dir.name} ({env_name})')}"
    )
    exit_code = _start_screen_session(
        _make_screen_session(f"eval-{checkpoint_dir.name}"), command
    )
    if exit_code:
        raise RuntimeError(f"Could not launch screen session (status {exit_code}).")


def run_analysis() -> None:
    analysis_action = _prompt_menu(
        "Select an analysis action: ",
        [
            "Run wait-time analysis",
            "Edit study configuration",
            "Back",
        ],
    )
    if analysis_action == "Back":
        return
    if analysis_action == "Edit study configuration":
        run_analysis_study_config_editor()
        return

    env_name = _prompt_menu(
        "Select an analysis environment: ",
        [
            "None (all environments)",
            "WaitAtGoal",
            "LiftQA",
            "LiftQA-Real",
        ],
    )
    if env_name.startswith("None"):
        env_name = "None"
    study_selection = _prompt_menu(
        "Select study collection: ",
        [
            "TC-DP paper",
            "PIP sweep",
            "SiP sweep",
            "All studies",
        ],
    )
    study_values = {
        "TC-DP paper": ["tc_dp_paper"],
        "PIP sweep": ["pip_sweep"],
        "SiP sweep": ["sip_sweep"],
        "All studies": [],
    }[study_selection]
    include_training_parameters = _prompt_bool(
        "Include training-parameter graph column", default=False
    )
    auto_resolve_datasets = _prompt_bool(
        "Automatically resolve available datasets", default=False
    )
    analysis_command = [
        sys.executable,
        "-m",
        "temporal_diffusion_policy.analysis.waittime",
        "--env_name",
        env_name,
    ]
    if study_values:
        analysis_command.extend(["--studies", f"[{','.join(study_values)}]"])
    if include_training_parameters:
        analysis_command.extend(["--include_training_parameters", "True"])
    if not auto_resolve_datasets:
        analysis_command.extend(["--auto_resolve_datasets", "False"])
    # Prefer the checked-out source over an older non-editable package install.
    # This keeps analysis launched from the CLI aligned with the current repo.
    analysis_env = os.environ.copy()
    source_root = str(REPO_ROOT / "src")
    existing_pythonpath = analysis_env.get("PYTHONPATH")
    analysis_env["PYTHONPATH"] = (
        source_root
        if not existing_pythonpath
        else source_root + os.pathsep + existing_pythonpath
    )
    subprocess.run(analysis_command, cwd=REPO_ROOT, env=analysis_env, check=True)


def run_analysis_study_config_editor() -> None:
    """Interactively add or remove manually curated analysis-study runs."""
    from temporal_diffusion_policy.analysis.study_configs import (
        STUDY_CONFIG_DIRECTORY,
        add_study_run,
        load_study_configs,
        remove_study_run,
    )
    from temporal_diffusion_policy.analysis.waittime import discover_environment_locators

    def select_environment_and_study(configs: dict) -> tuple[str, str]:
        environments = sorted(configs)
        environment = _prompt_menu("Select analysis environment: ", environments)
        studies = sorted(configs[environment])
        study = _prompt_menu("Select study: ", studies)
        return environment, study

    print(f"Manual study configuration directory: {STUDY_CONFIG_DIRECTORY}")
    while True:
        action = _prompt_menu(
            "Select an edit: ", ["List runs", "Add run", "Remove run", "Back"]
        )
        if action == "Back":
            return

        configs = load_study_configs()
        environment, study = select_environment_and_study(configs)
        runs = configs[environment][study]
        if action == "List runs":
            if not runs:
                print("No runs configured.")
            for index, run in enumerate(runs, start=1):
                print(f"  {index}. {run['path']}")
            continue

        if action == "Add run":
            available_runs = discover_environment_locators(environment)
            if not available_runs:
                print(f"No compatible local {environment} datasets were found.")
                continue
            print("Available datasets:")
            for index, locator in enumerate(available_runs, start=1):
                print(f"  {index}. {locator.path}")
            dataset_index = _prompt_int(
                "Dataset to add", default=1, minimum=1
            ) - 1
            if dataset_index >= len(available_runs):
                print(f"Enter a number from 1 to {len(available_runs)}.")
                continue
            selected_run = available_runs[dataset_index]
            path = selected_run.path
            label = input("Display label (leave blank for automatic): ").strip()
            aggregate_key = input("Aggregate key (leave blank for none): ").strip()
            run = {"path": path, "dataset_type": selected_run.dataset_type.value}
            if label:
                run["label"] = label
            if aggregate_key:
                run["aggregate_key"] = aggregate_key
            try:
                add_study_run(environment, study, run)
            except ValueError as error:
                print(error)
            else:
                print(f"Added {path} to {environment}/{study}.")
            continue

        if not runs:
            print("No runs configured.")
            continue
        for index, run in enumerate(runs, start=1):
            print(f"  {index}. {run['path']}")
        index = _prompt_int("Run to remove", default=1, minimum=1) - 1
        try:
            removed = remove_study_run(environment, study, index)
        except ValueError as error:
            print(error)
        else:
            print(f"Removed {removed['path']} from {environment}/{study}.")


def run_check_progress() -> None:
    """Print the current training-checkpoint report in this terminal."""
    command = [
        sys.executable,
        "-m",
        "temporal_diffusion_policy.scripts.check_progress",
    ]
    subprocess.run(command, cwd=REPO_ROOT, check=True)


def run_check_evaluation_progress() -> None:
    """Report whether every generated inference dataset has all episodes."""
    inference_root = REPO_ROOT / "data" / "inference"
    if not inference_root.is_dir():
        print(f"Inference root does not exist: {inference_root}")
        return

    rows: list[tuple[str, str, str]] = []
    for dataset_dir in sorted(inference_root.glob("*.zarr"), key=lambda path: path.name):
        attrs_path = dataset_dir / ".zattrs"
        episode_ends_path = dataset_dir / "meta" / "episode_ends" / ".zarray"
        episodes = "-/-"
        status = "invalid"
        try:
            attrs = json.loads(attrs_path.read_text(encoding="utf-8"))
            expected_count = int(attrs["num_episodes"])
            metadata = json.loads(episode_ends_path.read_text(encoding="utf-8"))
            completed_count = int(metadata["shape"][0])
            episodes = f"{completed_count}/{expected_count}"
            if completed_count == expected_count:
                status = "complete"
            elif time.time() - episode_ends_path.stat().st_mtime < 60:
                status = "active"
            else:
                status = "incomplete"
        except (FileNotFoundError, KeyError, TypeError, ValueError, json.JSONDecodeError):
            pass
        rows.append((dataset_dir.name, episodes, status))

    headers = ("inference dataset", "episodes", "status")
    widths = [
        max(len(header), *(len(row[index]) for row in rows)) if rows else len(header)
        for index, header in enumerate(headers)
    ]

    def print_rows(rows_to_print: list[tuple[str, str, str]]) -> None:
        for row in rows_to_print:
            formatted = []
            for index, value in enumerate(row):
                padded = value.ljust(widths[index])
                if headers[index] == "status" and value == "complete":
                    padded = f"\033[32m{padded}\033[0m"
                elif headers[index] == "status" and value == "active":
                    padded = f"\033[33m{padded}\033[0m"
                formatted.append(padded)
            print("  ".join(formatted))

    def print_section(
        title: str, rows_to_print: list[tuple[str, str, str]]
    ) -> None:
        print(f"\n{title}")
        if not rows_to_print:
            print("  none")
            return
        print("  ".join(header.ljust(widths[index]) for index, header in enumerate(headers)))
        print("  ".join("-" * width for width in widths))
        print_rows(rows_to_print)

    active_rows = [row for row in rows if row[2] == "active"]
    inactive_rows = [row for row in rows if row[2] != "active"]
    print(f"Inference root: {inference_root}")
    print_section("Completed or inactive evaluation runs:", inactive_rows)
    print_section("Active evaluation runs:", active_rows)


def run_checkpoint_cleanup() -> None:
    """Interactively prune stale checkpoints in this terminal."""
    command = [
        sys.executable,
        "-m",
        "temporal_diffusion_policy.scripts.clean_checkpoints",
    ]
    subprocess.run(command, cwd=REPO_ROOT, check=True)


def _training_runs() -> list[Path]:
    checkpoint_root = REPO_ROOT / "data" / "checkpoints"
    if not checkpoint_root.is_dir():
        return []
    return sorted(
        (
            checkpoint_dir for checkpoint_dir in checkpoint_root.iterdir()
            if checkpoint_dir.is_dir() and (checkpoint_dir / "config.txt").is_file()
        ),
        key=lambda path: path.name,
        reverse=True,
    )


def _evaluation_runs() -> list[Path]:
    inference_root = REPO_ROOT / "data" / "inference"
    if not inference_root.is_dir():
        return []
    return sorted(
        (
            dataset for dataset in inference_root.glob("*.zarr")
            if dataset.is_dir() and (dataset / ".zattrs").is_file()
        ),
        key=lambda path: path.name,
        reverse=True,
    )


def _prompt_run(run_kind: str, choices: list[Path]) -> Path:
    while True:
        print(f"\nAvailable {run_kind} runs:")
        for index, run in enumerate(choices, start=1):
            print(f"  {index}. {run.name}")
        answer = input(f"Select a {run_kind} run: ").strip()
        try:
            return choices[int(answer) - 1]
        except (ValueError, IndexError):
            print(f"Enter a number from 1 to {len(choices)}.")


def _read_training_parameters(checkpoint_dir: Path) -> dict[str, object]:
    """Read the saved draccus-style training config without importing training.py."""
    parameters: dict[str, object] = {}
    for line in (checkpoint_dir / "config.txt").read_text(encoding="utf-8").splitlines():
        key, separator, value = line.partition(":")
        if not separator:
            continue
        value = value.strip()
        try:
            parameters[key] = ast.literal_eval(value)
        except (ValueError, SyntaxError):
            parameters[key] = value
    return parameters


def _print_parameters(parameters: dict[str, object], indent: int = 2) -> None:
    """Print nested saved settings one parameter per line."""
    prefix = " " * indent
    for key in sorted(parameters):
        value = parameters[key]
        if isinstance(value, dict):
            print(f"{prefix}{key}:")
            _print_parameters(value, indent + 2)
        else:
            print(f"{prefix}{key}: {value}")


def _resolve_checkpoint_path(evaluation_parameters: dict[str, object]) -> Path | None:
    checkpoint_path = evaluation_parameters.get("checkpoint_path")
    if not checkpoint_path:
        return None
    path = Path(str(checkpoint_path))
    return path if path.is_absolute() else REPO_ROOT / path


def run_parameter_explorer() -> None:
    """Inspect a selected saved training or evaluation run configuration."""
    run_kind = _prompt_menu(
        "Explore parameters for: ",
        ["Training run", "Evaluation run", "Back"],
    )
    if run_kind == "Back":
        return

    if run_kind == "Training run":
        runs = _training_runs()
        if not runs:
            print("No training runs with config.txt were found.")
            return
        checkpoint_dir = _prompt_run("training", runs)
        print(f"\nTraining run: {checkpoint_dir.name}")
        _print_parameters(_read_training_parameters(checkpoint_dir))
        return

    runs = _evaluation_runs()
    if not runs:
        print("No evaluation datasets with .zattrs were found.")
        return
    dataset_dir = _prompt_run("evaluation", runs)
    evaluation_parameters = json.loads(
        (dataset_dir / ".zattrs").read_text(encoding="utf-8")
    )
    print(f"\nEvaluation run: {dataset_dir.name}")
    _print_parameters(evaluation_parameters)

    checkpoint_dir = _resolve_checkpoint_path(evaluation_parameters)
    if checkpoint_dir is None:
        print("\nReferenced training checkpoint: not recorded")
    elif not (checkpoint_dir / "config.txt").is_file():
        print(f"\nReferenced training checkpoint: {checkpoint_dir} (config.txt not found)")
    else:
        print(f"\nReferenced training checkpoint: {checkpoint_dir.name}")
        _print_parameters(_read_training_parameters(checkpoint_dir))






def _format_training_time(start_to_finish_seconds: float) -> str:
    """Format elapsed checkpoint training time as Xh Ym."""
    total_minutes = int(max(0.0, start_to_finish_seconds)) // 60
    hours, minutes = divmod(total_minutes, 60)
    return f"{hours}h {minutes}m"


def _checkpoint_timestamp_seconds(checkpoint_dir: Path) -> float | None:
    """Parse the timestamp embedded in a legacy or environment-prefixed run name."""
    match = re.search(r"(?<!\d)(\d{8}_\d{6})(?!\d)", checkpoint_dir.name)
    if match is None:
        return None
    try:
        return datetime.strptime(match.group(1), "%Y%m%d_%H%M%S").timestamp()
    except ValueError:
        return None


def _training_time_from_inference_dataset(dataset_path: Path) -> str:
    """Read config.txt-to-final.pth elapsed time for one inference dataset."""
    try:
        attrs = json.loads((dataset_path / ".zattrs").read_text(encoding="utf-8"))
        checkpoint_path = attrs.get("checkpoint_path")
        if not checkpoint_path:
            return "-"
        checkpoint_dir = Path(str(checkpoint_path))
        if not checkpoint_dir.is_absolute():
            checkpoint_dir = REPO_ROOT / checkpoint_dir
        config_path = checkpoint_dir / "config.txt"
        final_path = checkpoint_dir / "final.pth"
        if not final_path.is_file():
            return "-"
        final_time = final_path.stat().st_mtime
        config_time = config_path.stat().st_mtime if config_path.is_file() else None
        start_time = (
            config_time
            if config_time is not None and config_time <= final_time
            else _checkpoint_timestamp_seconds(checkpoint_dir)
        )
        if start_time is None or start_time > final_time:
            return "-"
        return _format_training_time(final_time - start_time)
    except (OSError, TypeError, ValueError, json.JSONDecodeError):
        return "-"


def _major_training_config_diff(dataset_path: Path) -> str:
    """Summarise core training settings that differ from source defaults."""
    try:
        attrs = json.loads((dataset_path / ".zattrs").read_text(encoding="utf-8"))
        checkpoint_path = attrs.get("checkpoint_path")
        if not checkpoint_path:
            return "-"
        checkpoint_dir = Path(str(checkpoint_path))
        if not checkpoint_dir.is_absolute():
            checkpoint_dir = REPO_ROOT / checkpoint_dir
        lines = (checkpoint_dir / "config.txt").read_text(encoding="utf-8").splitlines()
    except (OSError, TypeError, ValueError, json.JSONDecodeError):
        return "-"

    recorded_values: dict[str, object] = {}
    for line in lines:
        name, separator, value = line.partition(":")
        if not separator or name not in DEFAULT_MAJOR_TRAINING_CONFIG:
            continue
        try:
            recorded_values[name] = ast.literal_eval(value.strip())
        except (SyntaxError, ValueError):
            recorded_values[name] = value.strip()

    labels = {"batch_size": "bs", "num_epochs": "epochs", "lr": "lr"}
    differences = []
    for name, default_value in DEFAULT_MAJOR_TRAINING_CONFIG.items():
        value = recorded_values.get(name)
        if value is not None and value != default_value:
            value_label = f"{value:g}" if isinstance(value, float) else str(value)
            differences.append(f"{labels[name]}={value_label}")
    return ", ".join(differences) or "-"






def run_tools() -> None:
    """Show the repository tools that can be run from the interactive CLI."""
    tool_choices = [
        "Explore parameters",
        "Show GPU status (nvidia-smi)",
        "Back",
    ]
    while True:
        tool = _prompt_menu("Select a tool: ", tool_choices)
        if tool == "Back":
            return
        if tool == 'Explore parameters':
            run_parameter_explorer()
        elif tool == 'Show GPU status (nvidia-smi)':
            run_nvidia_smi()


def main() -> None:
    workflows = [
        "Train and evaluate",
        "Evaluate existing checkpoint",
        "Run analysis",
        "Check train progress",
        "Check eval progress",
        "Clean checkpoints",
        "Run tools",
        "Exit",
    ]
    handlers = {
        "Train and evaluate": run_train_and_evaluate,
        "Evaluate existing checkpoint": run_evaluate_checkpoint,
        "Run analysis": run_analysis,
        "Check train progress": run_check_progress,
        "Check eval progress": run_check_evaluation_progress,
        "Clean checkpoints": run_checkpoint_cleanup,
        "Run tools": run_tools,
    }

    while True:
        workflow = _prompt_menu("Select a workflow: ", workflows)
        if workflow == "Exit":
            return
        handlers[workflow]()


if __name__ == "__main__":
    main()
