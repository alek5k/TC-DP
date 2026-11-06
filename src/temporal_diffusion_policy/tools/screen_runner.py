"""
Composable screen session runner for training/inference jobs.

Usage:
    python screen_runner_new.py                # start sessions
    python screen_runner_new.py --stop         # stop sessions
    python screen_runner_new.py --dryrun       # preview commands
"""

import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional
import draccus
from .screen_utils import start_sessions, stop_sessions

# ════════════════════════════════════════════════════════════════════════════════
# Configuration
# ════════════════════════════════════════════════════════════════════════════════
@dataclass
class ScreenRunnerConfig:
    startup_stagger_interval_seconds: int = 10
    stop: bool = False
    dryrun: bool = False


@dataclass
class EnvDataset:
    """Environment and dataset pairing."""
    env_name: str
    dataset_path: str


@dataclass
class RunConfig:
    """Configuration for a single training run."""
    suffix: str = ""
    kwargs: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        assert "." not in self.suffix, f"RunConfig.suffix must not contain '.', got: {self.suffix!r}"
        assert not any(c.isspace() for c in self.suffix), f"RunConfig.suffix must not contain whitespace, got: {self.suffix!r}"


# ════════════════════════════════════════════════════════════════════════════════
# Command builder
# ════════════════════════════════════════════════════════════════════════════════
@dataclass
class CommandBuilder:
    """Fluent builder for shell commands with optional prefixes and chaining."""

    parts: List[str] = field(default_factory=list)
    conda_env: Optional[str] = None
    sleep_seconds: Optional[int] = None

    def with_conda(self, env_name: str) -> "CommandBuilder":
        self.conda_env = env_name
        return self

    def with_sleep(self, seconds: Optional[int]) -> "CommandBuilder":
        self.sleep_seconds = seconds
        return self

    def add(self, cmd: str) -> "CommandBuilder":
        self.parts.append(cmd)
        return self

    def run_python(self, script: str, cuda_device: Optional[int] = None, **kwargs: Any) -> "CommandBuilder":
        args = format_cli_args(**kwargs)
        cuda_prefix = f"CUDA_VISIBLE_DEVICES={cuda_device} " if cuda_device is not None else ""
        self.parts.append(f"{cuda_prefix}python {script} {args}".strip())
        return self

    def run_module(self, module: str, cuda_device: Optional[int] = None, **kwargs: Any) -> "CommandBuilder":
        args = format_cli_args(**kwargs)
        cuda_prefix = f"CUDA_VISIBLE_DEVICES={cuda_device} " if cuda_device is not None else ""
        self.parts.append(f"{cuda_prefix}python -m {module} {args}".strip())
        return self

    def wait_for_session(self, session_name: str) -> "CommandBuilder":
        wait_script = Path(__file__).parent / "screen_wait.py"
        assert wait_script.exists(), "screen_wait.py not found in the same directory as screen_runner.py"
        self.parts.append(f"python {wait_script} --pattern={session_name}")
        return self

    def exit_on_finish(self) -> "CommandBuilder":
        self.parts.append("exit")
        return self

    def extend(self, other: "CommandBuilder") -> "CommandBuilder":
        """Extend this builder with parts and env vars from another builder."""
        self.parts.extend(other.parts)
        return self

    def build(self) -> str:
        prefixes: List[str] = []
        if self.conda_env:
            prefixes.append(self.conda_env)
        if self.sleep_seconds:
            prefixes.append(f"sleep {self.sleep_seconds}")

        return " && ".join(prefixes + self.parts)


def format_cli_args(**kwargs: Any) -> str:
    """Convert kwargs to CLI-style arguments."""
    return " ".join(f"--{key}={value}" for key, value in kwargs.items() if value is not None)


# ════════════════════════════════════════════════════════════════════════════════
# High-level command factories
# ════════════════════════════════════════════════════════════════════════════════
def training_command(cuda_device: Optional[int] = None, batch_size: int = 512, num_epochs: int = 200, **kwargs: Any) -> CommandBuilder:
    from ..training import ExperimentConfig
    training_keys = set(asdict(ExperimentConfig()).keys())
    train_kwargs = {k: v for k, v in kwargs.items() if k in training_keys}
    return CommandBuilder().run_module("temporal_diffusion_policy.training", cuda_device=cuda_device, batch_size=batch_size, num_epochs=num_epochs, **train_kwargs)

def inference_command(cuda_device: Optional[int] = None, **kwargs: Any) -> CommandBuilder:
    from ..inference import InferenceConfig
    inference_keys = set(asdict(InferenceConfig()).keys())
    infer_kwargs = {k: v for k, v in kwargs.items() if k in inference_keys}
    return CommandBuilder().run_module("temporal_diffusion_policy.inference", cuda_device=cuda_device, **infer_kwargs)

def regenerate_dataset_command(source_path: str, output_path: str, env_name: str, seed: int = 0) -> CommandBuilder:
    return (
        CommandBuilder()
        .with_conda("godiffusion")
        .run_python(
            "data_collection_regenerate.py",
            source=source_path,
            output=output_path,
            env_name=env_name,
            continue_on_failure=True,
            seed=seed,
        )
        .add(f"cp -f {source_path}/.zattrs {output_path}/.zattrs")
        .run_python("data_regeneration_validation.py", source=source_path, output=output_path)
        .exit_on_finish()
    )


# ════════════════════════════════════════════════════════════════════════════════
# Sweep generators
# ════════════════════════════════════════════════════════════════════════════════

def regenerate_inference_datasets() -> Dict[str, str]:
    commands: Dict[str, str] = {}
    base = Path("data/inference")

    for d in base.iterdir():
        if not d.is_dir() or "regenerated" in d.name or "archive" in d.name:
            continue
        env = d.name.split("_")[0]
        dataset_name = d.name.replace(".zarr", "")
        new_path = f"data/inference/{dataset_name}_regenerated.zarr"
        cmd = regenerate_dataset_command(
            source_path=f"data/inference/{dataset_name}.zarr",
            output_path=new_path,
            env_name=env,
            seed=200,
        )
        commands[f"regenerate-{dataset_name}"] = cmd.build()
    return commands


def parallel_inference_commands(cuda_device: Optional[int] = None, num_episodes: int = 200, workers: int = 20, seed: int = 200, **kwargs: Any) -> Dict[str, str]:
    commands: Dict[str, str] = {}
    run_id = uuid.uuid4().hex[:8]

    def episode_range(worker_id: int) -> tuple[int, int]:
        base_count = num_episodes // workers
        remainder = num_episodes % workers
        start = worker_id * base_count + min(worker_id, remainder)
        end = start + base_count + (1 if worker_id < remainder else 0)
        return start, end

    server_cmd = (
        CommandBuilder()
        .with_conda("godiffusion")
        .run_python(
            "inference_server.py",
            parallel_eval_id=run_id,
            seed=seed,
            num_episodes=num_episodes,
            **kwargs,
        )
    )
    commands["parallel-eval-server"] = server_cmd.build()

    for i in range(workers):
        start, end = episode_range(i)
        worker_cmd = (
            CommandBuilder()
            .with_conda("godiffusion")
            .with_sleep(15)
            .run_python(
                "inference_client.py",
                is_part_of_parallel_eval=True,
                parallel_eval_id=run_id,
                seed=seed + start,
                num_episodes=end - start,
                **kwargs,
            )
            .exit_on_finish()
        )
        commands[f"parallel-eval-worker-{i}"] = worker_cmd.build()

    return commands


# ────────────────────────────────────────────────────────────────────────────────
# Common environments
# ────────────────────────────────────────────────────────────────────────────────

DEFAULT_ENVIRONMENTS = [
    EnvDataset("WaitAtGoal", "data/demonstration/wait_at_goal_dataset_v4.zarr"),
    # EnvDataset("AvoidObstacleEnv", "data/demonstration/avoid_obstacles_v3.zarr"),
    EnvDataset("LiftQA", "data/demonstration/lift_qa_v2.zarr"),
]



def build_sweep(name: str, runs: List[RunConfig], environments: List[EnvDataset] = DEFAULT_ENVIRONMENTS, num_gpus: int = 2, batch_size: int = 512, num_epochs: int = 200, num_infer_eps: int = 200) -> Dict[str, str]:
    """
    Build a sweep of training commands with GPU scheduling and dependencies.
    
    Args:
        name: Base name for the sweep (env_name will be auto-prefixed)
        runs: List of RunConfigs
        environments: List of EnvDataset to run sweep over
        num_gpus: Number of GPUs to distribute runs across (round-robin assignment)
        batch_size: Batch size for training
        num_epochs: Number of epochs for training
        num_infer_eps: Number of inference episodes
    """
    commands: Dict[str, str] = {}
    last_on_gpu: Dict[int, Optional[str]] = {}
    run_counter = 0

    for env in environments:
        base_name = f"{env.env_name}-{name}"

        for run in runs:
            run_name = f"{base_name}{run.suffix}"
            gpu = run_counter % num_gpus
            run_counter += 1
            
            builder = CommandBuilder().with_conda("godiffusion")

            # Wait for GPU availability
            if last_on_gpu.get(gpu):
                builder.wait_for_session(last_on_gpu[gpu])
            
            builder.extend(
                training_command(
                    cuda_device=gpu,
                    dataset_path=env.dataset_path,
                    env_name=env.env_name,
                    batch_size=batch_size,
                    num_epochs=num_epochs,
                    **run.kwargs,
                )
            ).extend(
                inference_command(
                    cuda_device=gpu,
                    env_name=env.env_name,
                    get_checkpoint_from_screen_tmpfile=True,
                    generated_dataset_suffix=name,
                    num_episodes=num_infer_eps,
                    **run.kwargs,
                )
            ).exit_on_finish()

            commands[run_name] = builder.build()
            last_on_gpu[gpu] = run_name

    return commands


def generate_full_sweep(include_id_sip_ablations = True, include_gp_ablations = True) -> Dict[str, str]:
    runs = [
        RunConfig(),
        RunConfig(
            suffix="_+ID",
            kwargs={"use_idleness_encoding": True},
        ),
        RunConfig(
            suffix="_+SaP",
            kwargs={"use_saturating_progress_encoding": True},
        ),
        RunConfig(
            suffix="_+SiP=128",
            kwargs={
                "use_sinusoidal_progress_encoding": True,
                "sinusoidal_progress_encoding_dim": 128,
            },
        ),
        RunConfig(
            suffix="_+ID+SaP",
            kwargs={
                "use_idleness_encoding": True,
                "use_saturating_progress_encoding": True,
            },
        ),
        RunConfig(
            suffix="_+ID+SiP=128",
            kwargs={
                "use_idleness_encoding": True,
                "use_sinusoidal_progress_encoding": True,
                "sinusoidal_progress_encoding_dim": 128,
            },
        ),
    ]

    if include_id_sip_ablations:
        runs_id_sip = [
            RunConfig(
                suffix=f"_+ID+SiP={dim}",
                kwargs={
                    "use_idleness_encoding": True,
                    "use_sinusoidal_progress_encoding": True,
                    "sinusoidal_progress_encoding_dim": dim,
                },
            )
            for dim in [4, 8, 32, 64, 256]
        ]
        runs.extend(runs_id_sip)

    if include_gp_ablations:
        runs_gp = [
            RunConfig(
                suffix=f"_+SiP={dim}",
                kwargs={
                    "use_sinusoidal_progress_encoding": True,
                    "sinusoidal_progress_encoding_dim": dim,
                },
            )
            for dim in [4, 8, 32, 64, 256]
        ]
        runs.extend(runs_gp)

    return build_sweep("", runs)




def generate_revalidation_sweep() -> Dict[str, str]:
    runs = [
        RunConfig(),
        RunConfig(
            suffix="_revalidate_+ID",
            kwargs={"use_idleness_encoding": True},
        ),
        RunConfig(
            suffix="_revalidate_+ID+SaP",
            kwargs={
                "use_idleness_encoding": True,
                "use_saturating_progress_encoding": True,
            },
        ),
        RunConfig(
            suffix="_revalidate_+ID+SiP=128",
            kwargs={
                "use_idleness_encoding": True,
                "use_sinusoidal_progress_encoding": True,
                "sinusoidal_progress_encoding_dim": 128,
            },
        ),
    ]

    return build_sweep("revalidation", runs)



def generate_id_sip_ablation_sweep() -> Dict[str, str]:
    runs = [
        RunConfig(suffix=f"_+ID+SiP={dim}", kwargs={
            "use_idleness_encoding": True,
            "use_sinusoidal_progress_encoding": True,
            "sinusoidal_progress_encoding_dim": dim,
        })
        for dim in [4, 8, 32, 64, 128, 256]
    ]
    runs2 = [
        RunConfig(suffix=f"_+SiP={dim}", kwargs={
            "use_sinusoidal_progress_encoding": True,
            "sinusoidal_progress_encoding_dim": dim,
        })
        for dim in [4, 8, 32, 64, 128, 256]
    ]
    runs.extend(runs2)
    return build_sweep("gp-ablation", runs)

def fmt_float(x: float) -> str:
    s = f"{x:g}"   # compact float formatting
    return s.replace(".", "p")


def generate_idleness_sensitivity_sweep() -> Dict[str, str]:
    runs = []
    for idleness_thresh_theta in [0.03, 0.06, 0.1, 0.5]: # 0.03 original
        for idleness_growth_rate_alpha in [0.02, 0.05, 0.1, 0.5]: # 0.047 original
            cfg = RunConfig(suffix=f"_theta_{fmt_float(idleness_thresh_theta)}_alpha_{fmt_float(idleness_growth_rate_alpha)}_+ID+SiP=256", kwargs={
                    "use_idleness_encoding": True,
                    "use_sinusoidal_progress_encoding": True,
                    "sinusoidal_progress_encoding_dim": 256,
                    "idleness_encoding_rest_thresh": idleness_thresh_theta,
                    "idleness_encoding_alpha": idleness_growth_rate_alpha,
                })
            runs.append(cfg)
    return build_sweep("idleness-sensitivity-sw", runs)

def generate_gp_ablation_sweep() -> Dict[str, str]:
    runs = [
        RunConfig(suffix=f"_+SiP={dim}", kwargs={
            "use_sinusoidal_progress_encoding": True,
            "sinusoidal_progress_encoding_dim": dim,
        })
        for dim in [4, 8, 32, 64, 128, 256]
    ]
    return build_sweep("gp-ablation", runs)


def generate_pip_sweep(num_gpus: int = 2, max_per_gpu: int = 7) -> Dict[str, str]:
    runs = [
        RunConfig(
            suffix=f"_+PIP_eps={eps}_win={win}_sigma={sigma}", # suffix for the screen session
            kwargs={
                "use_pause_induced_perturbation": True,
                "pip_epsilon": eps,
                "pip_window_size": win,
                "pip_sigma": sigma,
                "generated_dataset_suffix": f"_eps{eps}_win{win}_sig{sigma}", 
            },
        )
        for eps in [0.001, 0.01, 0.03, 0.06]
        for win in [20, 30, 40]
        for sigma in [0.6, 0.8, 0.9, 0.99]
    ]

    commands: Dict[str, str] = {}

    # NEW: per-GPU, keep K independent slot tails
    last_on_gpu: dict[int, list[Optional[str]]] = {
        gpu: [None] * max_per_gpu for gpu in range(num_gpus)
    }

    run_counter = 0

    for env in DEFAULT_ENVIRONMENTS:
        base_name = f"{env.env_name}"

        if env.env_name == "LiftQA":
            wandb_run_name = "wild-shape-165"
        elif env.env_name == "WaitAtGoal":
            wandb_run_name = "azure-blaze-100"
        else:
            raise ValueError(f"Unknown env name: {env.env_name}")

        for run in runs:
            run_name = f"{base_name}{run.suffix}"

            # Assign gpu + slot
            gpu = run_counter % num_gpus
            slot = (run_counter // num_gpus) % max_per_gpu
            run_counter += 1

            builder = CommandBuilder().with_conda("godiffusion")

            # Wait only for the previous run in this (gpu, slot)
            prev = last_on_gpu[gpu][slot]
            if prev:
                builder.wait_for_session(prev)

            builder.extend(
                inference_command(
                    cuda_device=gpu,
                    env_name=env.env_name,
                    wandb_run_name=wandb_run_name,
                    **run.kwargs,
                )
            ).exit_on_finish()

            commands[run_name] = builder.build()

            # Update slot tail
            last_on_gpu[gpu][slot] = run_name

    return commands

    # return build_sweep("pip-ablation", runs)

def test_command() -> Dict[str, str]:
    return {
        "test-session-1": CommandBuilder().with_conda("godiffusion").with_sleep(5).add("echo Hello World 1").build(),
        "test-session-2": CommandBuilder().with_conda("godiffusion").with_sleep(10).add("echo Hello World 2").exit_on_finish().build(),
        "test-session-3": CommandBuilder().with_conda("godiffusion").wait_for_session("test-session-2").with_sleep(3).add("echo Hello World 3").build()
    }
# ════════════════════════════════════════════════════════════════════════════════
# Main
# ════════════════════════════════════════════════════════════════════════════════

# Select which sweep to run
# COMMANDS = test_command()
# COMMANDS = generate_revalidation_sweep()
# COMMANDS = generate_pip_sweep()
# COMMANDS = generate_full_sweep()
# COMMANDS = generate_gp_ablation_sweep()
# COMMANDS = regenerate_inference_datasets()
COMMANDS = generate_idleness_sensitivity_sweep()

# COMMANDS = parallel_inference_commands(
#     cuda_device=1,
#     use_idleness_encoding=True,
#     wandb_run_name="volcanic-snow-84",
#     workers=100,
#     env_name="WaitAtGoal",
# )


@draccus.wrap()
def main(cfg: ScreenRunnerConfig) -> int:
    
    if cfg.dryrun:
        print("Dry run mode enabled. No changes will be made.\n")
        start_sessions(COMMANDS, cfg.startup_stagger_interval_seconds, dryrun=True)
        return 0
    elif cfg.stop:
        return stop_sessions(COMMANDS)
    else:
        return start_sessions(COMMANDS, cfg.startup_stagger_interval_seconds)
    


if __name__ == "__main__":
    main()
