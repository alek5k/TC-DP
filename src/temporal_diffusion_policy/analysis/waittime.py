from dataclasses import dataclass, field
from enum import Enum
import json
import os
from pathlib import Path
import re
import sys
import draccus
from temporal_diffusion_policy.analysis.study_configs import load_study_configs
from temporal_diffusion_policy.analysis.report import load_report_config, format_spread
from temporal_diffusion_policy.replay_buffer import ReplayBuffer
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import cv2
import textwrap
from scipy.stats.mstats import winsorize

OUTPUT_DIR = Path("outputs/analysis/waittime")

# The comparison reported in the main Results section.  Keeping this explicit
# prevents auxiliary ablations from silently appearing in the paper figure.
MAIN_PAPER_DISTRIBUTION_METHODS = tuple(load_report_config("paper_main")["methods"])

class DatasetType(str, Enum):
    UNKNOWN = "Unknown"
    DEMO = "Demo"
    BASELINE = "Baseline"
    INFERENCE = "Inference"


class Study(str, Enum):
    """Independent analysis collections a dataset may belong to."""

    TC_DP_PAPER = "tc_dp_paper"
    TC_DP_PAPER_MULTISEED = "tc_dp_paper_multiseed"
    PIP_SWEEP = "pip_sweep"
    SIP_SWEEP = "sip_sweep"


# These curated inputs reproduce the appendix's broad comparison without
# including exploratory temporal-encoder studies.  Their order also ensures
# PIP-sweep metadata (rather than the curated main-paper aliases) determines
# the detailed DP+PIP row labels.
FULL_SWEEP_STUDIES = (
    Study.PIP_SWEEP,
    Study.SIP_SWEEP,
    Study.TC_DP_PAPER,
)


@dataclass
class DatasetLocator:
    path: str
    label: str = None
    dataset_type: DatasetType = DatasetType.UNKNOWN
    # Optional display/grouping label for a paper table, e.g. ``PTP``.
    table_method: str | None = None
    # Give repeated inference runs the same value to report one aggregate W
    # result.  This is deliberately manual: filenames are not used to infer a
    # method or seed grouping.
    aggregate_key: str | None = None

@dataclass
class ParsedDatasetInfo:
    path: str
    config: dict
    is_demo_dataset: bool
    dataset_type: DatasetType = DatasetType.UNKNOWN
    ID: bool = False
    ID_theta: float | None = None
    ID_alpha: float | None = None
    SaP: bool = False
    SiP: bool = False
    SiP_dim: int | str | None = None
    PIP: bool = False
    PIP_epsilon: float | None = None
    PIP_window_size: int | None = None
    PIP_sigma: float | None = None

    # Short label for plots
    short_label: str = None

    # Optional override for table method label/grouping.
    table_method: str | None = None

    # Manual grouping key copied from DatasetLocator for multi-run aggregates.
    aggregate_key: str | None = None

    # Read only when the optional training-parameter graph column is enabled.
    training_parameters: tuple[tuple[str, str], ...] = ()

    # Histogram data
    wait_times: list[float] = None
    step_counts: list[int] = None
    poses: list[np.ndarray] = None
    goal_visit_counts: list[int] = None
    goal_stages: list[int] = None
    success_rate: float | None = None
    
    # Wasserstein metrics
    W_pause: float | None = None
    W_episode: float | None = None
    W_total: float | None = None


@dataclass
class AnalysisConfig:
    env_name: str = 'None'  # Analyze all three paper environments by default.
    studies: list[Study] = field(default_factory=lambda: [Study.TC_DP_PAPER])  # Empty means include all studies.
    histogram_bins: int = 81 # if columns are too spread out, reduce histogram_bins
    include_rankings: bool = False
    include_abbreviation_legend: bool = False
    include_heatmap: bool = False
    include_training_parameters: bool = True
    include_demo_dataset: bool = False
    include_visit_counts: bool = False
    include_full_sweep: bool = False
    # With ``env_name=None``, print the curated main-paper table, or the
    # appendix full-sweep table when ``include_full_sweep`` is enabled.
    generate_tex_tables: bool = False
    # Generate the curated paper-comparison distribution figure for supported
    # temporal ambiguity environments.
    generate_main_paper_distribution: bool = False
    print_failure_modes: bool = False
    winsorize_durations: bool = True  # Apply 1%%/99%% tail clipping for W metrics.
    winsorize_using_demo_distribution: bool = False  # Use demo-derived clipping bounds for every dataset. # downside: force all rollout samples into the demo’s chosen support. so can still detect differences in mass distribution within these bounds, but completely loses information about "how far outside" the demonstrated timing a rollout went.
    aggregate_wasserstein: bool = True  # Report aggregate W metrics for manual aggregate_key groups.
    aggregate_uncertainty: str = "standard_deviation"  # Or "student_t_95".
    aggregate_confidence_level: float = 0.95  # Used only for optional t-CIs.
    aggregate_histograms_by_key: bool = True  # Collapse manually grouped runs into one histogram row with per-bin sample SD.
    auto_resolve_datasets: bool = True  # Discover compatible local datasets instead of using the static study lists.

    h_space: float = 0.05 # 0.05 # vertical spacing between rows
    w_space: float = 0.05 # horizontal spacing between columns

    subplot_width_scale: float = 9.0  # width per subplot column
    subplot_height_scale: float = 1.8  # height per subplot row
    include_sip_dim_in_label: bool = True
    use_single_overlay_legend: bool = False
    hide_second_col_y_tick_labels: bool = False
    axis_label_font_size: int | None = None
    overlay_legend_font_size: int | None = None
    demo_legend_label: str = "Demonstration Dataset"

font_size = 18
legend_size = 14
tick_label_size = 14

# sudo apt-get update
# sudo apt-get install -y texlive-latex-base texlive-latex-extra texlive-fonts-recommended dvipng cm-super

matplotlib.rcParams.update({
    "font.size": font_size,          # base text
    "axes.titlesize": font_size,
    "axes.labelsize": font_size,
    "xtick.labelsize": tick_label_size,
    "ytick.labelsize": tick_label_size,
    "legend.fontsize": legend_size,
    
    "text.usetex": True,
    "text.latex.preamble": r"\usepackage{times}",
    "font.family": "serif",   # LaTeX will use Computer Modern by default
    "axes.unicode_minus": False,

})


#####################
## STUDY CONFIGS ##
#####################


def load_manual_study_configs() -> dict[str, dict[Study, list[DatasetLocator]]]:
    """Convert the editable JSON study configuration into typed locators."""
    raw_configs = load_study_configs()
    typed_configs: dict[str, dict[Study, list[DatasetLocator]]] = {}
    for environment, raw_studies in raw_configs.items():
        typed_configs[environment] = {}
        for study_name, runs in raw_studies.items():
            try:
                study = Study(study_name)
            except ValueError as error:
                raise ValueError(
                    f"Unknown study {study_name!r} in the analysis JSON config."
                ) from error
            typed_configs[environment][study] = [
                DatasetLocator(
                    path=run["path"],
                    label=run.get("label"),
                    dataset_type=DatasetType(run.get("dataset_type", DatasetType.UNKNOWN)),
                    table_method=run.get("table_method"),
                    aggregate_key=run.get("aggregate_key"),
                )
                for run in runs
            ]
    return typed_configs


_manual_study_configs = load_manual_study_configs()
wait_at_goal = _manual_study_configs.get("WaitAtGoal", {})
liftqa = _manual_study_configs.get("LiftQA", {})
liftqa_realenv = _manual_study_configs.get("LiftQA-Real", {})

def select_datasets_by_type(
    datasets: list[DatasetLocator], allowed_types: set[DatasetType]
) -> list[DatasetLocator]:
    return [dataset for dataset in datasets if dataset.dataset_type in allowed_types]


def select_datasets_by_studies(
    datasets: dict[Study, list[DatasetLocator]], selected_studies: list[Study]
) -> list[DatasetLocator]:
    """Return requested study lists, deduplicated by dataset path."""
    requested = selected_studies or list(Study)
    selected = []
    seen_paths = set()
    for study in requested:
        for dataset in datasets.get(study, []):
            if dataset.path not in seen_paths:
                selected.append(dataset)
                seen_paths.add(dataset.path)
    return selected


def select_datasets(
    datasets: dict[Study, list[DatasetLocator]],
    allowed_types: set[DatasetType],
    selected_studies: list[Study],
) -> list[DatasetLocator]:
    return select_datasets_by_type(
        select_datasets_by_studies(datasets, selected_studies), allowed_types
    )


def _dataset_env_name(dataset_path: Path) -> str | None:
    """Read ``env_name`` without opening the potentially large Zarr arrays."""
    try:
        attrs = json.loads((dataset_path / ".zattrs").read_text(encoding="utf-8"))
    except (FileNotFoundError, OSError, ValueError, json.JSONDecodeError):
        return None
    env_name = attrs.get("env_name")
    return str(env_name) if env_name is not None else None


def _dataset_name_matches_environment(dataset_path: Path, env_name: str) -> bool:
    """Match legacy datasets that predate the ``env_name`` Zarr attribute."""
    normalized_name = dataset_path.stem.lower().replace("-", "_")
    aliases = {
        "WaitAtGoal": ("waitatgoal", "wait_at_goal"),
        "LiftQA": ("liftqa", "lift_qa"),
        "LiftQA-Real": ("liftqa_real", "liftqa_realenv", "lift_qa_real"),
    }
    if env_name == "LiftQA":
        return (
            any(alias in normalized_name for alias in aliases[env_name])
            and "real" not in normalized_name
        )
    return any(alias in normalized_name for alias in aliases.get(env_name, ()))


def discover_environment_locators(
    env_name: str, repo_root: Path | None = None
) -> list[DatasetLocator]:
    """Discover complete local demo and inference datasets for one environment.

    This avoids historical, manually-maintained file paths.  The selected study
    flag does not filter this mode because a Zarr dataset does not record its
    study membership; use ``--auto_resolve_datasets False`` for those curated
    static study lists.
    """
    repo_root = Path.cwd() if repo_root is None else Path(repo_root)
    root_kinds = (
        (repo_root / "data" / "demonstration", DatasetType.DEMO),
        (repo_root / "data" / "demonstrations", DatasetType.DEMO),
        (repo_root / "data" / "inference", DatasetType.INFERENCE),
        (repo_root / "data" / "inference_extended", DatasetType.INFERENCE),
    )
    locators: list[DatasetLocator] = []
    seen_paths: set[Path] = set()
    for dataset_root, dataset_type in root_kinds:
        if not dataset_root.is_dir():
            continue
        for dataset_path in sorted(dataset_root.glob("*.zarr")):
            if (
                not dataset_path.is_dir()
                or not (dataset_path / "meta" / "episode_ends").exists()
                or dataset_path.resolve() in seen_paths
            ):
                continue
            recorded_env_name = _dataset_env_name(dataset_path)
            if recorded_env_name != env_name and not _dataset_name_matches_environment(
                dataset_path, env_name
            ):
                continue
            seen_paths.add(dataset_path.resolve())
            try:
                path = str(dataset_path.relative_to(repo_root))
            except ValueError:
                path = str(dataset_path)
            locators.append(
                DatasetLocator(
                    path=path,
                    label="Demo" if dataset_type == DatasetType.DEMO else None,
                    dataset_type=dataset_type,
                )
            )
    return sorted(locators, key=lambda locator: (locator.dataset_type != DatasetType.DEMO, locator.path))


def resolve_environment_datasets(
    cfg: AnalysisConfig,
    env_name: str,
    manual_datasets: dict[Study, list[DatasetLocator]],
    selected_studies: list[Study],
) -> list[DatasetLocator]:
    """Choose automatic discovery by default, with static studies as opt-in."""
    if cfg.auto_resolve_datasets:
        return discover_environment_locators(env_name)
    return select_datasets(manual_datasets, set(DatasetType), selected_studies)


def can_analyze_datasets(env_name: str, datasets: list[DatasetLocator]) -> bool:
    """Report a non-fatal selection problem before opening any dataset files."""
    if not datasets:
        print(
            f"No compatible {env_name} datasets found. "
            "Use --auto_resolve_datasets False to use the static study list."
        )
        return False
    if not any(dataset.dataset_type == DatasetType.DEMO for dataset in datasets):
        print(
            f"No {env_name} demonstration dataset was found; "
            "wait-time comparison requires a demo reference."
        )
        return False
    return True


def training_parameters_from_dataset(dataset_path: Path) -> tuple[tuple[str, str], ...]:
    """Read displayable training parameters using the CLI checker's lookup."""
    from temporal_diffusion_policy.tools.check_piano_note_sequence import get_run_configuration

    try:
        return get_run_configuration(dataset_path).training_parameters
    except (AttributeError, KeyError, OSError, ValueError):
        return ()


def study_output_name(studies: list[Study]) -> str:
    """Return a filename-safe label for the selected analysis collection."""
    if set(studies) == set(Study):
        return "all_studies"
    return "_and_".join(study.value for study in studies)


def print_table(datasets: list[ParsedDatasetInfo]):
    from tabulate import tabulate
    rows = []
    headers = [
        "label", "W_ps", "W_ep", "success",
    ]

    for d in datasets:
        rows.append([
            d.short_label,
            d.W_pause, d.W_episode, d.success_rate,
        ])

    
    print(tabulate(
        rows,
        headers=headers,
        tablefmt="github",        # other good ones: "fancy_grid", "rounded_grid", "psql"
        floatfmt=".2f",
        missingval="",
    ))


def _mean_t_confidence_interval(
    values: list[float],
    confidence_level: float,
) -> tuple[float, float, float]:
    """Return the mean and two-sided Student's t CI across inference runs."""
    from scipy.stats import t as student_t

    values_array = np.asarray(values, dtype=np.float64)
    if values_array.ndim != 1 or len(values_array) < 2:
        raise ValueError("A t confidence interval requires at least two runs.")
    if not 0 < confidence_level < 1:
        raise ValueError("aggregate_confidence_level must be between 0 and 1.")

    mean = float(values_array.mean())
    standard_error = float(values_array.std(ddof=1) / np.sqrt(len(values_array)))
    alpha = 1.0 - confidence_level
    critical_value = float(student_t.ppf(1.0 - alpha / 2.0, df=len(values_array) - 1))
    margin = critical_value * standard_error
    return mean, mean - margin, mean + margin


def _mean_and_spread(values, uncertainty="standard_deviation", confidence_level=0.95):
    """Mean and selected spread across independent runs."""
    if uncertainty == "student_t_95":
        mean, lower, upper = _mean_t_confidence_interval(values, confidence_level)
        return mean, (upper - lower) / 2
    if uncertainty != "standard_deviation":
        raise ValueError(f"Unknown aggregate uncertainty: {uncertainty}")
    if len(values) < 2:
        raise ValueError("Sample standard deviation requires at least two runs.")
    return float(np.mean(values)), float(np.std(values, ddof=1))


def print_aggregated_wasserstein(
    datasets: list[ParsedDatasetInfo],
    confidence_level: float,
    uncertainty: str = "standard_deviation",
) -> None:
    """Print mean W metrics and sample SD for manual run groups by default.

    Each DatasetLocator must explicitly declare the same ``aggregate_key`` for
    runs that represent the same method.  Individual datasets remain in the
    normal table and plots; this function only adds a summary table.
    """
    from collections import defaultdict
    from tabulate import tabulate

    groups: dict[str, list[ParsedDatasetInfo]] = defaultdict(list)
    for dataset in datasets:
        if not dataset.is_demo_dataset and dataset.aggregate_key:
            groups[dataset.aggregate_key].append(dataset)

    multi_run_groups = {
        key: group for key, group in groups.items() if len(group) >= 2
    }
    if not multi_run_groups:
        return

    rows = []
    for key, group in multi_run_groups.items():
        row = [key, len(group)]
        for metric in ("W_pause", "W_episode"):
            mean, margin = _mean_and_spread(
                [getattr(dataset, metric) for dataset in group],
                uncertainty=uncertainty,
                confidence_level=confidence_level,
            )
            row.append(f"{mean:.2f} ± {format_spread(margin)}")
        rows.append(row)

    spread_label = ("sample standard deviation" if uncertainty == "standard_deviation"
                    else f"{confidence_level * 100:g}% Student's t CI")
    print(
        "\nAggregate Wasserstein distance across independent inference runs "
        f"({spread_label}):"
    )
    print(tabulate(
        rows,
        headers=["aggregate key", "n runs", "W_ps", "W_ep"],
        tablefmt="github",
    ))


def print_analysis_tables(
    cfg: AnalysisConfig, parsed_datasets: dict[str, ParsedDatasetInfo]
) -> None:
    """Print per-run metrics, followed by configured multi-run aggregates."""
    datasets = list(parsed_datasets.values())
    print_table(datasets)
    if cfg.aggregate_wasserstein:
        print_aggregated_wasserstein(
            datasets,
            confidence_level=cfg.aggregate_confidence_level,
            uncertainty=cfg.aggregate_uncertainty,
        )


def extract_episode_metrics(
    path: str,
    replay_buffer: ReplayBuffer,
    num_episodes: int | None = None,
) -> list[float]:
           
    wait_times = []
    step_counts = []
    agent_poses = []
    num_goal_visits = []
    max_wait_per_visit = []
    max_goal_stages = []

    if num_episodes is None:
        num_episodes = replay_buffer.n_episodes

    environment = replay_buffer.root.attrs.get("env_name")
    is_liftqa_real = environment == "LiftQA-Real" or (
        environment is None and "LiftQA-Real" in path
    )
    max_steps = int(replay_buffer.root.attrs.get("max_steps", 400))

    for ep in range(num_episodes):
        # Do not use ``get_episode`` here: it loads every Zarr array for the episode, including image observations that these metrics never use.
        episode_slice = replay_buffer.get_episode_slice(ep)
        keys_to_extract = ("wait_time", "step_count", "wait_times_each_visit", "agent_pose")
        rb = {key: replay_buffer[key][episode_slice] for key in keys_to_extract}
        for key in ("goal_state", "reward"):
            if key in replay_buffer:
                rb[key] = replay_buffer[key][episode_slice]
        last_wait_time = rb['wait_time'][-1]
        last_step_count = rb['step_count'][-1]
        last_wait_times_each_visit = rb['wait_times_each_visit'][-1]
        last_wait_times_each_visit = last_wait_times_each_visit[last_wait_times_each_visit >= 0]
        ep_num_goal_visits = len(last_wait_times_each_visit)
        ep_max_wait_per_visit = np.max(last_wait_times_each_visit) if ep_num_goal_visits > 0 else 0.0 # use max, not average. Trouble with average: quickly entering then leaving goal area results in very small wait times, bringing down the average
        if is_liftqa_real and "demo" not in path:
            # Archived real-world rollouts omit done and task phase. Ending
            # before the configured limit is the proxy for COMPLETED; also
            # require strictly more than two seconds accumulated at the goal.
            episode_length = episode_slice.stop - episode_slice.start
            waited_at_goal = float(np.max(rb["wait_time"])) > 2.0
            ended_before_limit = episode_length < max_steps
            max_goal_stage = 2 if waited_at_goal and ended_before_limit else (
                1 if waited_at_goal else 0
            )
        else:
            try:
                max_goal_stage = np.max(rb['goal_state'])
            except KeyError:
                if "demo" in path:
                    max_goal_stage = 2 # demos always complete all stages
                else:
                    # Older simulation rollouts infer completion from reward
                    # and early termination because goal_state was not saved.
                    max_reward = rb['reward'].max()
                    num_steps = rb['step_count'][-1]
                    if np.isclose(max_reward, 0.0):
                        max_goal_stage = -1
                    if np.isclose(max_reward, 0.3):
                        max_goal_stage = 0
                    elif np.isclose(max_reward, 0.6) and num_steps >=399:
                        max_goal_stage = 1
                    elif np.isclose(max_reward, 0.6) and num_steps < 399:
                        max_goal_stage = 2
                    elif max_reward >= 0.9:
                        max_goal_stage = 2
        wait_times.append(last_wait_time)
        step_counts.append(last_step_count)
        num_goal_visits.append(ep_num_goal_visits)
        max_wait_per_visit.append(ep_max_wait_per_visit)
        agent_poses.append(rb['agent_pose'])
        max_goal_stages.append(max_goal_stage)

    processed_episodes = len(max_goal_stages)
    success_rate = 100.0 * np.sum(np.array(max_goal_stages) >= 2) / processed_episodes if processed_episodes > 0 else 0.0

    return max_wait_per_visit, step_counts, agent_poses, num_goal_visits, max_goal_stages, success_rate


def generate_short_label_from_dataset_metadata(
    dataset_path: str,
    dataset_type: DatasetType = DatasetType.UNKNOWN,
    include_sip_dim_in_label: bool = True,
) -> str:
    replay_buffer = ReplayBuffer.create_from_path(dataset_path, mode='r')
    attrs = replay_buffer.root.attrs

    using_pip = attrs.get("use_pause_induced_perturbation", False)
    using_id = attrs.get("use_idleness_encoding", False)
    using_direct_idleness = attrs.get("use_direct_idleness_conditioning", False)
    using_sap = attrs.get("use_saturating_progress_encoding", False)
    using_sip = attrs.get("use_sinusoidal_progress_encoding", False)

    is_demo = dataset_type == DatasetType.DEMO
    is_baseline = dataset_type == DatasetType.BASELINE

    if is_demo:
        label_str = "Demo"
    elif is_baseline:
        label_str = "DP Baseline"
    elif using_pip:
        pip_epsilon = attrs.get("pip_epsilon", None)
        pip_window_size = attrs.get("pip_window_size", None)
        pip_sigma = attrs.get("pip_sigma", None)
        label_str = f"PIP $\epsilon={pip_epsilon}, w={pip_window_size}, \sigma={pip_sigma}$"
    else:
        label_components = []
        if using_id:
            label_components.append("ID")
        if using_direct_idleness:
            label_components.append("DI")
        if using_sap:
            label_components.append("SaP")
        if using_sip:
            sip_dim = attrs.get("sinusoidal_progress_encoding_dim", None)
            label_components.append(f"SiP")
        label_str = "TC-DP"
        if len(label_components) > 0:
            label_str += " (" + "+".join(label_components) + ")"
        if using_sip and include_sip_dim_in_label:
            label_str += f" [$d={sip_dim}$]"

    return label_str


@draccus.wrap()
def main(cfg: AnalysisConfig):
    standard_studies = cfg.studies or (
        FULL_SWEEP_STUDIES if cfg.include_full_sweep else [Study.TC_DP_PAPER]
    )
    selected_study_name = (
        "automatic" if cfg.auto_resolve_datasets else study_output_name(standard_studies)
    )

    if cfg.env_name == 'None':
        paper_results: dict[str, ParsedDatasetInfo] = {}
        all_environments = [('WaitAtGoal', wait_at_goal, 'wait_at_goal_datasets'), ('LiftQA', liftqa, 'lift_qa_datasets'), ('LiftQA-Real', liftqa_realenv, 'lift_qa_realenv_datasets')]
        for (env_name, locators, save_prefix) in all_environments:
            cfg.env_name = env_name
            datasets = resolve_environment_datasets(cfg, env_name, locators, standard_studies)
            if not can_analyze_datasets(env_name, datasets):
                continue
            parsed_datasets = parse_datasets(cfg, datasets)
            if env_name in {'WaitAtGoal', 'LiftQA', 'LiftQA-Real'}:
                paper_results.update({f'{env_name}:{dataset_path}': dataset for (dataset_path, dataset) in parsed_datasets.items()})
            print(f'Generating graph for {env_name}')
            generate_graphs(cfg, parsed_datasets, save_prefix=save_prefix, study_name=selected_study_name)
            if cfg.generate_main_paper_distribution and env_name in {'WaitAtGoal', 'LiftQA'}:
                generate_main_paper_distribution(cfg, parsed_datasets, study_name=selected_study_name)
            print_analysis_tables(cfg, parsed_datasets)
        if cfg.generate_tex_tables:
            if cfg.include_full_sweep:
                print('\nAppendix full-sweep LaTeX table:')
                create_full_sweep_tex_table(paper_results)
            else:
                print('\nMain paper LaTeX table:')
                create_main_paper_tex_table(paper_results)
    elif cfg.env_name == 'WaitAtGoal':
        datasets = resolve_environment_datasets(cfg, 'WaitAtGoal', wait_at_goal, standard_studies)
        if not can_analyze_datasets('WaitAtGoal', datasets):
            return
        parsed_datasets = parse_datasets(cfg, datasets)
        generate_graphs(cfg, parsed_datasets, save_prefix='wait_at_goal_datasets', study_name=selected_study_name)
        if cfg.generate_main_paper_distribution:
            generate_main_paper_distribution(cfg, parsed_datasets, study_name=selected_study_name)
        print_analysis_tables(cfg, parsed_datasets)
    elif cfg.env_name == 'LiftQA':
        datasets = resolve_environment_datasets(cfg, 'LiftQA', liftqa, standard_studies)
        if not can_analyze_datasets('LiftQA', datasets):
            return
        parsed_datasets = parse_datasets(cfg, datasets)
        generate_graphs(cfg, parsed_datasets, save_prefix='lift_qa_datasets', study_name=selected_study_name)
        if cfg.generate_main_paper_distribution:
            generate_main_paper_distribution(cfg, parsed_datasets, study_name=selected_study_name)
        print_analysis_tables(cfg, parsed_datasets)
    elif cfg.env_name == 'LiftQA-Real':
        datasets = resolve_environment_datasets(cfg, 'LiftQA-Real', liftqa_realenv, standard_studies)
        if not can_analyze_datasets('LiftQA-Real', datasets):
            return
        parsed_datasets = parse_datasets(cfg, datasets)
        generate_graphs(cfg, parsed_datasets, save_prefix='lift_qa_realenv_datasets', study_name=selected_study_name)
        print_analysis_tables(cfg, parsed_datasets)


def report_archived_rebuttal_failure_modes(
    parsed_datasets: dict[str, ParsedDatasetInfo],
) -> None:
    """Reproduce the archived LiftQA-Real failure-mode counting procedure.

    This was formerly tied to a ``Rebuttal`` pseudo-environment.  The method
    remains here as an explicit record: delayed/premature actuation is defined
    relative to the demo wait-time range plus/minus two 15-Hz timesteps, and
    trajectory-repeat labels below came from manual visual inspection.

    Call this after ``parse_datasets`` only when reproducing that analysis.
    """
    demo_dataset = next(
        (dataset for dataset in parsed_datasets.values() if dataset.is_demo_dataset),
        None,
    )
    if demo_dataset is None:
        raise ValueError("Archived rebuttal failure analysis requires a demo dataset.")

    demo_min_wait = np.min(demo_dataset.wait_times)
    demo_max_wait = np.max(demo_dataset.wait_times)
    threshold = 2 / 15  # Two timesteps at 15 Hz.
    print(
        "Demo wait time stats: "
        f"min={demo_min_wait:.2f} median={np.median(demo_dataset.wait_times):.2f} "
        f"max={demo_max_wait:.2f}"
    )

    manually_inspected_repeats = {
        "DP Baseline": [
            'Y', 'Y', 'N', 'Y', 'N', 'Y', 'N', 'N', 'N', 'Y',
            'Y', 'Y', 'Y', 'Y', 'Y', 'Y', 'Y', 'Y', 'N', 'Y',
            'N', 'N', 'Y', 'N', 'N', 'Y', 'Y', 'N', 'Y', 'N',
            'N', 'Y', 'Y', 'Y', 'Y', 'Y', 'N', 'Y', 'Y', 'N',
            'N', 'Y', 'N', 'Y', 'Y', 'N', 'N', 'Y', 'N', 'Y',
        ],
        "TC-DP (ID+SiP)": ['N'] * 50,
    }

    for dataset in parsed_datasets.values():
        if dataset.is_demo_dataset:
            continue
        repeat_labels = manually_inspected_repeats.get(dataset.short_label)
        if repeat_labels is None:
            print(f"Skipping {dataset.short_label}: no archived repeat annotations.")
            continue
        if len(repeat_labels) != len(dataset.wait_times):
            raise ValueError(
                f"{dataset.short_label}: {len(repeat_labels)} manual repeat labels for "
                f"{len(dataset.wait_times)} episodes."
            )

        trajectory_repeat = {i for i, label in enumerate(repeat_labels) if label == 'Y'}
        delayed = {
            i for i, wait_time in enumerate(dataset.wait_times)
            if wait_time > demo_max_wait + threshold
        }
        premature = {
            i for i, wait_time in enumerate(dataset.wait_times)
            if wait_time < demo_min_wait - threshold
        }
        any_failure = trajectory_repeat | delayed | premature
        total = len(dataset.wait_times)
        print(f"Dataset: {dataset.short_label}")
        print(f"   Episodes with some failure mode: {len(any_failure)} / {total}")
        print(f"   Delayed actuation episodes: {len(delayed)} / {total}")
        print(f"   Premature actuation episodes: {len(premature)} / {total}")
        print(f"   Trajectory repeat episodes: {len(trajectory_repeat)} / {total}")


def parse_datasets(cfg: AnalysisConfig, datasets: list[DatasetLocator]) -> dict[str, ParsedDatasetInfo]:
    parsed_datasets = {}

    for dataset in datasets:

        desc = generate_short_label_from_dataset_metadata(
            dataset.path,
            dataset_type=dataset.dataset_type or DatasetType.UNKNOWN,
            include_sip_dim_in_label=cfg.include_sip_dim_in_label,
        ) if dataset.label is None else dataset.label
        path = dataset.path

        inference_replay_buffer = ReplayBuffer.create_from_path(path, mode='r')
        num_episodes = inference_replay_buffer.n_episodes
        print(f"Parsing dataset: {desc} (num_episodes={num_episodes})")
        (episode_wait_times, episode_step_counts, agent_poses, goal_visits, max_goal_stages, success_rate) = extract_episode_metrics(path, inference_replay_buffer)

        # Print out extreme episodes
        if cfg.print_failure_modes:
            for i in sorted(range(len(episode_step_counts)), key=episode_step_counts.__getitem__):
                goal_visit = goal_visits[i]
                step_count = episode_step_counts[i]
                wait_time = episode_wait_times[i]
                failure_types = []
                if goal_visit > 1:
                    failure_types.append("trajectory_restart")
                if "WaitAtGoal" in path and wait_time > 3.5:
                    failure_types.append("delayed_actuation")
                if "LiftQA" in path and wait_time > 5.5:
                    failure_types.append("delayed_actuation")
                if wait_time < 2.0:
                    failure_types.append("premature_actuation")
                if step_count > 390:
                    failure_types.append("maxsteps")
                failure_type = ", ".join(failure_types)
                if failure_type:
                    print(f"   Failure Mode in {dataset.path} Episode {i:03d}: goal_visit={goal_visit} step_count={step_count} failure_type=[{failure_type}]")

        parsed_datasets[path] = ParsedDatasetInfo(
            path=path,
            short_label=desc,
            # Some archived rollout datasets (including the retained PTP
            # evaluations) predate the ``env_name`` metadata field. The
            # analysis request already identifies their environment, so retain
            # that information for downstream cross-environment tables.
            config={
                **dict(inference_replay_buffer.root.attrs),
                "env_name": inference_replay_buffer.root.attrs.get(
                    "env_name", cfg.env_name
                ),
            },
            is_demo_dataset=(dataset.dataset_type == DatasetType.DEMO),
            dataset_type=dataset.dataset_type or DatasetType.UNKNOWN,
            table_method=dataset.table_method,
            aggregate_key=dataset.aggregate_key,
            training_parameters=(
                training_parameters_from_dataset(Path(path))
                if cfg.include_training_parameters else ()
            ),
            ID=inference_replay_buffer.root.attrs.get("use_idleness_encoding", False),
            ID_theta=inference_replay_buffer.root.attrs.get("idleness_encoding_rest_thresh", None),
            ID_alpha=inference_replay_buffer.root.attrs.get("idleness_encoding_alpha", None),
            SaP=inference_replay_buffer.root.attrs.get("use_saturating_progress_encoding", False),
            SiP=inference_replay_buffer.root.attrs.get("use_sinusoidal_progress_encoding", False),
            SiP_dim=inference_replay_buffer.root.attrs.get("sinusoidal_progress_encoding_dim", None),
            PIP=inference_replay_buffer.root.attrs.get("use_pause_induced_perturbation", False),
            PIP_epsilon=inference_replay_buffer.root.attrs.get("pip_epsilon", None),
            PIP_window_size=inference_replay_buffer.root.attrs.get("pip_window_size", None),
            PIP_sigma=inference_replay_buffer.root.attrs.get("pip_sigma", None),
            wait_times=episode_wait_times,
            step_counts=episode_step_counts,
            poses=agent_poses,
            goal_visit_counts=goal_visits,
            goal_stages=max_goal_stages,
            success_rate=success_rate,
        )

    calculate_wasserstein_distances(
        parsed_datasets,
        winsorize_durations=cfg.winsorize_durations,
        winsorize_using_demo_distribution=cfg.winsorize_using_demo_distribution,
    )
    return parsed_datasets


def timing_metric_name(env_name: str) -> str:
    """Return the display name for the first timing distribution."""
    return 'pause'


def _winsorize_duration_samples(samples: list[float] | list[int]) -> np.ndarray:
    """Winsorize duration tails at the 1st and 99th empirical percentiles."""
    return np.asarray(
        winsorize(np.asarray(samples), limits=(0.01, 0.01)), dtype=np.float64
    )


def _demo_winsor_bounds(samples: list[float] | list[int]) -> tuple[float, float]:
    """Return 1%/99% bounds implied by winsorizing demonstration samples."""
    winsorized = _winsorize_duration_samples(samples)
    return float(np.min(winsorized)), float(np.max(winsorized))


def calculate_wasserstein_distances(
    parsed_datasets: dict[str, ParsedDatasetInfo],
    winsorize_durations: bool = True,
    winsorize_using_demo_distribution: bool = False,
):
    from scipy.stats import wasserstein_distance

    demonstration_dataset: ParsedDatasetInfo = next(d for d in parsed_datasets.values() if d.is_demo_dataset)
    if winsorize_durations:
        if winsorize_using_demo_distribution:
            wait_bounds = _demo_winsor_bounds(demonstration_dataset.wait_times)
            episode_bounds = _demo_winsor_bounds(demonstration_dataset.step_counts)
            transform_wait_times = lambda samples: np.clip(samples, *wait_bounds)
            transform_episode_durations = lambda samples: np.clip(samples, *episode_bounds)
        else:
            transform_wait_times = _winsorize_duration_samples
            transform_episode_durations = _winsorize_duration_samples
    else:
        transform_wait_times = np.asarray
        transform_episode_durations = np.asarray
    reference_wait_times = transform_wait_times(demonstration_dataset.wait_times)
    reference_episode_durations = transform_episode_durations(demonstration_dataset.step_counts)

    for dataset_path, parsed_info in parsed_datasets.items():
        # Raw durations remain available for plots and failure-mode reporting.
        # When enabled, only W_pause/W_episode use 1%/99% winsorized data.
        # The optional demo mode applies the demonstration's same bounds to
        # every rollout rather than estimating separate bounds per dataset.
        wait_times = transform_wait_times(parsed_info.wait_times)
        episode_durations = transform_episode_durations(parsed_info.step_counts)
        w_distance_wait = wasserstein_distance(wait_times, reference_wait_times)
        w_distance_step = wasserstein_distance(
            episode_durations, reference_episode_durations
        )

        # divide step count by 10 (control Hz of environment) to get seconds, which aligns to wait_time, making both units seconds
        control_hz = demonstration_dataset.config.get("control_hz")
        if control_hz is None:
            env_cfg = demonstration_dataset.config.get("env")
            if isinstance(env_cfg, dict):
                control_hz = env_cfg.get("control_hz")
        if control_hz is None:
            control_hz = 10

        w_distance_step /= control_hz
        # Combine (L1)
        combined = w_distance_wait + w_distance_step

        parsed_datasets[dataset_path].W_pause = w_distance_wait
        parsed_datasets[dataset_path].W_episode = w_distance_step
        parsed_datasets[dataset_path].W_total = combined


def _histogram_plot_entries(
    parsed_datasets: dict[str, ParsedDatasetInfo],
    aggregate_by_key: bool,
) -> list[tuple[str, list[ParsedDatasetInfo], bool]]:
    """Build individual or manually aggregated rows for the histogram figure."""
    entries: list[tuple[str, list[ParsedDatasetInfo], bool]] = []
    grouped_entries: dict[str, int] = {}
    for dataset in parsed_datasets.values():
        is_aggregate = (
            aggregate_by_key
            and not dataset.is_demo_dataset
            and dataset.aggregate_key is not None
        )
        if not is_aggregate:
            entries.append((dataset.short_label, [dataset], dataset.is_demo_dataset))
            continue

        entry_index = grouped_entries.get(dataset.aggregate_key)
        if entry_index is None:
            grouped_entries[dataset.aggregate_key] = len(entries)
            entries.append((dataset.aggregate_key, [dataset], False))
        else:
            entries[entry_index][1].append(dataset)
    return entries


def select_main_paper_distribution_datasets(
    parsed_datasets: dict[str, ParsedDatasetInfo],
) -> dict[str, ParsedDatasetInfo]:
    """Return exactly the demo and curated Results-section comparison rows.

    Selection is based on the manually declared ``table_method`` values in the
    study configuration, rather than inferred from filenames or encoder flags.
    This makes the figure reproducible and prevents standalone ablations from
    changing the main-paper comparison.
    """
    demos = [
        (path, dataset)
        for path, dataset in parsed_datasets.items()
        if dataset.is_demo_dataset
    ]
    if len(demos) != 1:
        raise ValueError(
            "The main-paper distribution requires exactly one demonstration "
            f"dataset; found {len(demos)}."
        )

    methods: dict[str, tuple[str, ParsedDatasetInfo]] = {}
    for path, dataset in parsed_datasets.items():
        method = dataset.table_method
        if method not in MAIN_PAPER_DISTRIBUTION_METHODS:
            continue
        if method in methods:
            raise ValueError(
                "The main-paper distribution requires one rollout dataset per "
                f"method; found multiple entries for {method!r}."
            )
        methods[method] = (path, dataset)

    missing = [method for method in MAIN_PAPER_DISTRIBUTION_METHODS if method not in methods]
    if missing:
        raise ValueError(
            "Cannot generate the main-paper distribution; missing configured "
            f"methods: {', '.join(missing)}."
        )

    selected = dict(demos)
    selected.update(methods[method] for method in MAIN_PAPER_DISTRIBUTION_METHODS)
    return selected


def generate_main_paper_distribution(
    cfg: AnalysisConfig,
    parsed_datasets: dict[str, ParsedDatasetInfo],
    study_name: str,
) -> None:
    """Generate the fixed paper-comparison distribution for one environment."""
    selected_datasets = select_main_paper_distribution_datasets(parsed_datasets)
    save_prefix_by_environment = {
        "WaitAtGoal": "wait_at_goal_main_results",
        "LiftQA": "lift_qa_main_results",
    }
    try:
        save_prefix = save_prefix_by_environment[cfg.env_name]
    except KeyError as error:
        raise ValueError(
            "Curated paper distributions are supported for WaitAtGoal and LiftQA; "
            f"received {cfg.env_name!r}."
        ) from error
    print(f"Generating curated paper-comparison distribution for {cfg.env_name}")
    generate_graphs(
        cfg,
        selected_datasets,
        save_prefix=save_prefix,
        study_name=study_name,
    )


def _mean_histogram_and_spread(
    samples_per_run: list[np.ndarray],
    bins: np.ndarray,
    confidence_level: float,
    uncertainty: str = "standard_deviation",
) -> tuple[np.ndarray, np.ndarray]:
    """Average per-run histograms and return per-bin sample SD by default."""
    from scipy.stats import t as student_t

    histograms = np.asarray([
        np.histogram(samples, bins=bins)[0] / len(samples)
        for samples in samples_per_run
    ])
    mean = histograms.mean(axis=0)
    if len(histograms) < 2:
        return mean, np.zeros_like(mean)

    if uncertainty == "standard_deviation":
        return mean, histograms.std(axis=0, ddof=1)
    if uncertainty != "student_t_95":
        raise ValueError(f"Unknown aggregate uncertainty: {uncertainty}")
    standard_error = histograms.std(axis=0, ddof=1) / np.sqrt(len(histograms))
    alpha = 1.0 - confidence_level
    critical_value = student_t.ppf(1.0 - alpha / 2.0, df=len(histograms) - 1)
    return mean, critical_value * standard_error


def _plot_histogram_spread_whiskers(
    axis: plt.Axes,
    bins: np.ndarray,
    mean: np.ndarray,
    spread: np.ndarray,
) -> None:
    """Draw faint capless spread whiskers on non-empty bins."""
    mask = (mean != 0) | (spread != 0)
    centers = 0.5 * (bins[:-1] + bins[1:])
    # Histogram probabilities cannot be negative.  Keep the upper spread
    # intact, but truncate only the displayed lower whisker at zero.
    lower_margin = np.minimum(spread, mean)
    axis.errorbar(
        centers[mask],
        mean[mask],
        yerr=np.vstack((lower_margin[mask], spread[mask])),
        fmt="none",
        ecolor="0.35",
        elinewidth=0.65,
        capsize=0,
        alpha=0.55,
        zorder=3,
    )

def generate_graphs(
    cfg: AnalysisConfig,
    parsed_datasets: dict[str, ParsedDatasetInfo],
    save_prefix: str,
    study_name: str,
):


    demonstration_dataset: ParsedDatasetInfo = next(d for d in parsed_datasets.values() if d.is_demo_dataset)
    plot_entries = _histogram_plot_entries(
        parsed_datasets, aggregate_by_key=cfg.aggregate_histograms_by_key
    )
    displayed_entries = [
        entry for entry in plot_entries if cfg.include_demo_dataset or not entry[2]
    ]
    all_wait_times = [np.array(d.wait_times) for d in parsed_datasets.values()]
    all_step_counts = [np.array(d.step_counts) for d in parsed_datasets.values()]
    all_agent_goal_visit_counts = [d.goal_visit_counts for d in parsed_datasets.values()]

    # Compute common bin edges (e.g. from combined data)
    combined_wait_times = np.concatenate(all_wait_times)
    combined_step_counts = np.concatenate(all_step_counts)
    combined_visit_counts = np.concatenate(all_agent_goal_visit_counts)
    bins_wait_times = np.histogram_bin_edges(combined_wait_times, bins=cfg.histogram_bins) 
    bins_step_counts = np.histogram_bin_edges(combined_step_counts, bins=cfg.histogram_bins)
    bins_visit_counts = np.histogram_bin_edges(combined_visit_counts, bins=cfg.histogram_bins)

    ##########################################
    ## 2. Plot per-bin vs reference (row 0) ##
    ##########################################
    ref_wait_hist, _ = np.histogram(demonstration_dataset.wait_times, bins=bins_wait_times)
    ref_step_hist, _ = np.histogram(demonstration_dataset.step_counts, bins=bins_step_counts)
    ref_visit_hist, _ = np.histogram(demonstration_dataset.goal_visit_counts, bins=bins_visit_counts)

    wait_centers = 0.5 * (bins_wait_times[:-1] + bins_wait_times[1:])
    step_centers = 0.5 * (bins_step_counts[:-1] + bins_step_counts[1:])
    visit_centers = 0.5 * (bins_visit_counts[:-1] + bins_visit_counts[1:])

    wait_widths = np.diff(bins_wait_times)
    step_widths = np.diff(bins_step_counts)
    visit_widths = np.diff(bins_visit_counts)

    ncols = 2
    width_ratios = [1, 1]

    if cfg.include_training_parameters:
        ncols += 1
        width_ratios.append(0.55)

    if cfg.include_visit_counts:
        ncols += 1
        width_ratios.append(1)

    if cfg.include_heatmap:
        ncols += 1
        width_ratios.append(0.4)

    if cfg.include_rankings:
        ncols += 1
        width_ratios.append(0.2)
    
    
    num_rows = len(displayed_entries)

    cmap = plt.get_cmap("autumn") # viridis
    fig, axes = plt.subplots(
        nrows=num_rows,
        ncols=ncols,
        figsize=(cfg.subplot_width_scale * sum(width_ratios), cfg.subplot_height_scale * num_rows),
        sharex="col",
        sharey="col",
        gridspec_kw={'width_ratios': width_ratios}
    )
    if axes.ndim == 1:
        axes = axes[np.newaxis, :]

    from matplotlib.patches import Patch
    legend_method_handles = []
    demo_legend_handle = Patch(facecolor="none", edgecolor="blue", linewidth=1.5, label=cfg.demo_legend_label)
    axis_label_kwargs = {}
    if cfg.axis_label_font_size is not None:
        axis_label_kwargs["fontsize"] = cfg.axis_label_font_size
    timing_metric = timing_metric_name(cfg.env_name)

    for row, (entry_label, entry_datasets, is_demo_entry) in enumerate(displayed_entries):
        wait_time_runs = [np.asarray(dataset.wait_times) for dataset in entry_datasets]
        step_count_runs = [np.asarray(dataset.step_counts) for dataset in entry_datasets]
        goal_visit_runs = [np.asarray(dataset.goal_visit_counts) for dataset in entry_datasets]
        wait_time_data = np.concatenate(wait_time_runs)
        step_count_data = np.concatenate(step_count_runs)
        agent_pose_data = [pose for dataset in entry_datasets for pose in dataset.poses]
        goal_visits = np.concatenate(goal_visit_runs)
        is_aggregate = len(entry_datasets) > 1
        label = (
            f"{entry_label} (n={len(entry_datasets)})"
            if is_aggregate else entry_label
        )

        color = (
            cmap(1.0 - row / (len(displayed_entries) - 1))
            if len(displayed_entries) > 1 else cmap(0.0)
        )
        # color = cmap(row)

        # In aggregate mode, normalize each run before averaging so that a
        # longer rollout file cannot dominate a method's distribution.
        if cfg.aggregate_histograms_by_key:
            wait_hist, wait_hist_spread = _mean_histogram_and_spread(
                wait_time_runs, bins_wait_times, cfg.aggregate_confidence_level, cfg.aggregate_uncertainty
            )
            step_hist, step_hist_spread = _mean_histogram_and_spread(
                step_count_runs, bins_step_counts, cfg.aggregate_confidence_level, cfg.aggregate_uncertainty
            )
            visit_hist, visit_hist_spread = _mean_histogram_and_spread(
                goal_visit_runs, bins_visit_counts, cfg.aggregate_confidence_level, cfg.aggregate_uncertainty
            )
            ref_wait_plot = ref_wait_hist / len(demonstration_dataset.wait_times)
            ref_step_plot = ref_step_hist / len(demonstration_dataset.step_counts)
            ref_visit_plot = ref_visit_hist / len(demonstration_dataset.goal_visit_counts)
        else:
            wait_hist, _ = np.histogram(wait_time_data, bins=bins_wait_times)
            step_hist, _ = np.histogram(step_count_data, bins=bins_step_counts)
            visit_hist, _ = np.histogram(goal_visits, bins=bins_visit_counts)
            wait_hist_spread = step_hist_spread = visit_hist_spread = None
            ref_wait_plot = ref_wait_hist
            ref_step_plot = ref_step_hist
            ref_visit_plot = ref_visit_hist

        col_id = 0
        waittime_axis = axes[row, col_id]

        col_id += 1
        stepcount_axis = axes[row, col_id]

        if cfg.include_training_parameters:
            col_id += 1
            training_parameter_axis = axes[row, col_id]

        if cfg.include_visit_counts:
            col_id += 1
            visitcount_axis = axes[row, col_id]

        if cfg.include_heatmap:
            col_id += 1
            traj_axis = axes[row, col_id]

        if cfg.include_rankings:
            col_id += 1
            text_axis = axes[row, col_id]


        waittime_axis.grid(False)
        stepcount_axis.grid(False)

        if cfg.hide_second_col_y_tick_labels:
            stepcount_axis.tick_params(axis="y", which="both", left=False, labelleft=False)

        if cfg.include_training_parameters:
            training_parameter_axis.axis("off")
            parameter_texts = []
            for dataset in entry_datasets:
                parameters = dict(dataset.training_parameters)
                parameter_lines = [
                    f"{label}: {parameters[key]}"
                    for key, label in (
                        ("batch_size", "Batch size"),
                        ("lr", "Learning rate"),
                        ("num_epochs", "Epochs"),
                    )
                    if key in parameters
                ]
                if parameter_lines:
                    parameter_texts.append("\n".join(parameter_lines))
            unique_parameter_texts = list(dict.fromkeys(parameter_texts))
            training_parameter_axis.text(
                0.0,
                0.5,
                "Training parameters\n" + (
                    "\n\n".join(unique_parameter_texts)
                    if unique_parameter_texts else "Unavailable"
                ),
                transform=training_parameter_axis.transAxes,
                ha="left",
                va="center",
                fontsize=12,
            )

        if cfg.include_visit_counts:
            visitcount_axis.grid(True)
        
        # Filled bars = current run
        waittime_axis.bar(
            wait_centers,
            wait_hist,
            width=wait_widths,
            # color="red",
            color=color,
            alpha=0.7,          # transparency so dotted reference shows through
            edgecolor="black",
            linewidth=0.5, # 0.8
        )
        if is_aggregate:
            _plot_histogram_spread_whiskers(
                waittime_axis, bins_wait_times, wait_hist, wait_hist_spread
            )
        stepcount_axis.bar(
            step_centers,
            step_hist,
            width=step_widths,
            # color="red",
            color=color,
            alpha=0.7,
            edgecolor="black",
            linewidth=0.5,
        )
        if is_aggregate:
            _plot_histogram_spread_whiskers(
                stepcount_axis, bins_step_counts, step_hist, step_hist_spread
            )
        if cfg.include_visit_counts:
            visitcount_axis.bar(
                visit_centers,
                visit_hist,
                width=visit_widths,
                # color="red",
                color=color,
                alpha=0.7,
                edgecolor="black",
                linewidth=0.5,
            )
            if is_aggregate:
                _plot_histogram_spread_whiskers(
                    visitcount_axis, bins_visit_counts, visit_hist, visit_hist_spread
                )

        # Dotted outline = reference (ALWAYS)
        mask = ref_wait_plot != 0
        waittime_axis.bar(
            wait_centers[mask],
            ref_wait_plot[mask],
            width=wait_widths[mask],
            facecolor="none",
            edgecolor="blue"
            # edgecolor="black",
            # linestyle=(0, (1, 1)),
            # linewidth=1.8,
        )

        mask = ref_step_plot != 0
        stepcount_axis.bar(
            step_centers[mask],
            ref_step_plot[mask],
            width=step_widths[mask],
            facecolor="none",
            edgecolor="blue"
            # edgecolor="black",
            # linestyle=(0, (1, 1)),
            # linewidth=1.8,
        )
        if cfg.include_visit_counts:
            visitcount_axis.bar(
                visit_centers,
                ref_visit_plot,
                width=visit_widths,
                facecolor="none",
                edgecolor="blue"
                # edgecolor="black",
                # linestyle=(0, (1, 1)),
                # linewidth=1.8,
            )

        def add_traj(accum, agent_pose_data, extent, thickness=1):
            h, w = accum.shape

            x = agent_pose_data[:, 0]
            y = agent_pose_data[:, 1]

            xmin, xmax, ymin, ymax = extent

            # map world -> pixel
            u = (x - xmin) / (xmax - xmin) * (w - 1)
            v = (y - ymin) / (ymax - ymin) * (h - 1)
            pts = np.stack([u, v], axis=1).round().astype(np.int32)

            pts[:, 0] = np.clip(pts[:, 0], 0, w - 1)
            pts[:, 1] = np.clip(pts[:, 1], 0, h - 1)

            mask = np.zeros_like(accum, dtype=np.uint8)
            cv2.polylines(mask, [pts], False, 255, thickness=thickness, lineType=cv2.LINE_AA)

            accum += mask.astype(np.float32) / 255.0
            return accum


        def show_accum(ax, accum, extent, p=99, gamma=0.45, cmap="magma"):
            # for fainter details visible - lower gamma. 
            # for hotspot not dominating - lower p.
            vmax = np.percentile(accum[accum > 0], p) if np.any(accum > 0) else 1.0
            img = np.clip(accum / vmax, 0, 1) ** gamma
            xmin, xmax, ymin, ymax = extent
            ax.imshow(img, origin="upper", extent=[xmin, xmax, ymin, ymax], aspect="equal", cmap=cmap)


        all_xy = np.concatenate(agent_pose_data, axis=0)
        xmin, xmax = all_xy[:, 0].min(), all_xy[:, 0].max()
        ymin, ymax = all_xy[:, 1].min(), all_xy[:, 1].max()
        extent = (xmin, xmax, ymin, ymax)
        resolution_factor = 1 # accum shape ---> can be much larger for more detail
        H = int((ymax - ymin) * resolution_factor)
        W = int((xmax - xmin) * resolution_factor)
        accum = np.zeros((H, W), np.float32)
        for traj in agent_pose_data:
            accum = add_traj(accum, traj, extent, thickness=resolution_factor)

        if cfg.include_heatmap:
            show_accum(traj_axis, accum, extent, gamma=0.75) # increasing gamma brightens all trajectories, but can saturate overlaps too.
            traj_axis.set_title(label + " Trajectories")


        wd_waittime = float(np.mean([dataset.W_pause for dataset in entry_datasets]))
        wd_stepcount = float(np.mean([dataset.W_episode for dataset in entry_datasets]))
        wd_combined_l1 = float(np.mean([dataset.W_total for dataset in entry_datasets]))

        all_wd_totals = [
            float(np.mean([dataset.W_total for dataset in datasets]))
            for _, datasets, is_demo in displayed_entries if not is_demo
        ]

        is_best_l1 = wd_combined_l1 == min(all_wd_totals) if not is_demo_entry else False
        ranked_l1 = sorted(all_wd_totals).index(wd_combined_l1) + 1 if not is_demo_entry else 0
        # label_suffix = " $\\bf{best}$ $\\bf{L1(W_{wait}, W_{step})}$" if is_best_l1 else ""

        histogram_y_label = "Probability" if cfg.aggregate_histograms_by_key else "Frequency"
        waittime_axis.set_ylabel(histogram_y_label, fontweight="bold", **axis_label_kwargs)
        # stepcount_axis.set_ylabel("Count", fontweight="bold")
        def format_wasserstein_label(metric: str, tex_name: str) -> str:
            values = [getattr(dataset, metric) for dataset in entry_datasets]
            mean = float(np.mean(values))
            if len(values) == 1:
                return rf"$W_{{{tex_name}}}={mean:.2f}$"
            _, margin = _mean_and_spread(
                values, uncertainty=cfg.aggregate_uncertainty,
                confidence_level=cfg.aggregate_confidence_level,
            )
            return rf"$W_{{{tex_name}}}={mean:.2f}\pm{format_spread(margin)}$"

        label_waittime_wd = format_wasserstein_label("W_pause", timing_metric)
        label_stepcount_wd = format_wasserstein_label("W_episode", "episode")
        # waittime_axis.set_title(f"{label} ({label_waittime_wd})")
        # stepcount_axis.set_title(f"{label} ({label_stepcount_wd})")

        legend_handles_waittime = [
            Patch(facecolor=color, edgecolor="black", alpha=0.7, label=f"{label}, {label_waittime_wd}"), #label="Policy rollout"
            Patch(facecolor="none", edgecolor="blue", linewidth=1.5, label=cfg.demo_legend_label),
        ]
        legend_handles_stepcount = [
            Patch(facecolor=color, edgecolor="black", alpha=0.7, label=f"{label}, {label_stepcount_wd}"), #label="Policy rollout"
            Patch(facecolor="none", edgecolor="blue", linewidth=1.5, label=cfg.demo_legend_label),
        ]
        

        if cfg.use_single_overlay_legend:
            legend_method_handles.append(
                Patch(facecolor=color, edgecolor="black", alpha=0.7, label=label)
            )
        else:
            waittime_axis.legend(
                handles=legend_handles_waittime,
                loc="upper right",
                # frameon=False,
                # fontsize=9,
            )

            stepcount_axis.legend(
                handles=legend_handles_stepcount,
                loc="upper right",
                # frameon=False,
                # fontsize=9,
            )

        
        # Text axis
        if cfg.include_rankings:
            text_axis.axis("off")
            bold_yes = "$\\bf{Yes}$"
            text = f"""
            $L1(W_{{wait}} + W_{{step}})$: {wd_waittime:.2f} + {wd_stepcount:.2f} = {wd_combined_l1:.2f}
            Best L1: {bold_yes if is_best_l1 else "No"}
            Rank: {ranked_l1}
            """
            text_axis.text(0.0, 0.5, text, transform=text_axis.transAxes, ha="left", va="center", fontsize=10)

        if row == len(displayed_entries) - 1:
            timing_axis_label = 'Agent pause time at goal (seconds)'
            waittime_axis.set_xlabel(timing_axis_label, fontweight='bold', **axis_label_kwargs)
            stepcount_axis.set_xlabel('Episode step count', fontweight='bold', **axis_label_kwargs)
        else:
            waittime_axis.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
            stepcount_axis.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
        

    if cfg.use_single_overlay_legend and len(legend_method_handles) > 0:
        # Deduplicate labels while preserving order.
        unique_handles = []
        seen_labels = set()
        for handle in legend_method_handles:
            if handle.get_label() in seen_labels:
                continue
            unique_handles.append(handle)
            seen_labels.add(handle.get_label())

        overlay_axis = axes[num_rows - 1, ncols - 1]
        overlay_loc = "upper right"

        overlay_axis.legend(
            handles=[demo_legend_handle] + unique_handles,
            loc=overlay_loc,
            framealpha=0.9,
            fontsize=cfg.overlay_legend_font_size,
        )

    if cfg.include_abbreviation_legend:
        fig.subplots_adjust(
            hspace=0.2,
            top=0.90,
            bottom=0.15,
            left=0.06,
            right=0.99,
        )
        fig.text(
            0.5, 0.01,
            "ST = Scalar Time-at-Rest | ID = Idleness Encoding | "
            "SaP = Saturating Progress Encoding | SiP = Sinusoidal Progress Encoding (sinusoidal) | "
            "T = Static Temperature | DT = Dynamic Temperature | W = Wasserstein Distance to Demo (seconds)",
            ha="center",
            va="bottom",
            fontsize=9,
            wrap=True
        )
    else:

        if cfg.include_heatmap:
            cfg.w_space += 0.05
        fig.subplots_adjust(
            hspace=cfg.h_space,   # vertical spacing between rows
            wspace=cfg.w_space,   # horizontal spacing between columns
        )

    # fig.tight_layout(pad=0.05, h_pad=0.08, w_pad=0.15)
    os.makedirs("analysis_waittime", exist_ok=True)
    output_stem = f"{save_prefix}_{study_name}_analysis_reference_dotted_overlay"
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_DIR / f"{output_stem}.png", dpi=300, bbox_inches='tight')
    fig.savefig(OUTPUT_DIR / f"{output_stem}.pdf", dpi=300, bbox_inches='tight')
    print(f"Saved {OUTPUT_DIR / f'{output_stem}.png'}")


def create_tex_table_from_results(
    results: dict[str, ParsedDatasetInfo],
    best_only: bool = True,
    combine_pip: bool = True,
    include_success: bool = False,
    separate_tables_per_env: bool = False,
    env_order: list[str] | None = None,
    main_paper_only: bool = False,
):
    """Print a LaTeX table from parsed rollout results.

    ``main_paper_only`` selects the archived single-seed comparison rows. Those rows
    must have explicit ``table_method`` values in their study JSON entries;
    full individual-encoding ablations remain available when it is disabled.
    """
    if main_paper_only:
        from temporal_diffusion_policy.analysis.report import aggregate_results, render_table

        report = load_report_config("archive_older_single_seed_comparison")
        if env_order is not None:
            report["environments"] = env_order
        if not include_success:
            report["metrics"] = [m for m in report["metrics"] if m != "success_rate"]
        print(render_table(report, aggregate_results(report, results)), end="")
        return
    if env_order is None:
        env_order = ["WaitAtGoal", "LiftQA", "LiftQA-Real"]

    env_titles = {
        "WaitAtGoal": "WaitAtGoal",
        "LiftQA": "LiftQA",
        "LiftQA-Real": "LiftQA-Real",
        "can": "Can",
    }

    base_metrics = ["W_pause", "W_episode", "W_total"]
    min_metrics = base_metrics  # minima only for W_* (not success)

    # -----------------------------
    # Sort datasets
    # -----------------------------
    datasets = [r for r in results.values() if not r.is_demo_dataset]

    def method_sort_key(r: ParsedDatasetInfo):
        if r.table_method is not None:
            explicit_order = {
                "BC": -2,
                "BC-RNN": -1,
                "DP": 0,
                "DP+PIP": 1,
                "PTP": 2,
                "TC-DP (ID+SaP)": 5,
                "TC-DP (ID+SiP)": 6,
            }
            return (explicit_order.get(r.table_method, 99), 0)
        if not r.ID and not r.SaP and not r.SiP and not r.PIP:
            base = 0
        elif not r.ID and not r.SaP and not r.SiP and r.PIP:
            base = 1
        elif r.ID and not r.SaP and not r.SiP:
            base = 2
        elif not r.ID and r.SaP and not r.SiP:
            base = 3
        elif not r.ID and not r.SaP and r.SiP:
            base = 4
        elif r.ID and r.SaP and not r.SiP:
            base = 5
        elif r.ID and not r.SaP and r.SiP:
            base = 6
        else:
            base = 99

        if r.SiP:
            if r.SiP_dim == "best":
                sip_order = -1
            elif r.SiP_dim is None:
                sip_order = 1_000_000
            else:
                sip_order = int(r.SiP_dim)
        else:
            sip_order = 0

        return (base, sip_order)
    datasets = sorted(datasets, key=method_sort_key)

    # -----------------------------
    # best_only selection
    # -----------------------------
    best_idsip_dim = {env: None for env in env_order}
    best_idsip_val = {env: float("inf") for env in env_order}

    if best_only:
        for d in datasets:
            env = d.config["env_name"]
            if env not in env_order or d.W_total is None:
                continue
            if d.ID and (not d.SaP) and d.SiP and (d.SiP_dim is not None):
                if d.W_total < best_idsip_val[env]:
                    best_idsip_val[env] = d.W_total
                    best_idsip_dim[env] = d.SiP_dim

    # -----------------------------
    # Group across envs
    # -----------------------------
    groups = {}
    for d in datasets:
        env = d.config["env_name"]
        if env not in env_order:
            continue

        sip_dim = d.SiP_dim if d.SiP else None
        pip_epsilon = d.PIP_epsilon if d.PIP else None
        pip_window_size = d.PIP_window_size if d.PIP else None
        pip_sigma = d.PIP_sigma if d.PIP else None

        if d.table_method is not None:
            key = ("explicit", d.table_method)
        elif combine_pip:
            key = (d.ID, d.SaP, d.SiP, sip_dim, d.PIP, None, None, None)
        else:
            key = (d.ID, d.SaP, d.SiP, sip_dim, d.PIP, pip_epsilon, pip_window_size, pip_sigma)

        if best_only and d.table_method is None:
            if (not d.ID) and (not d.SaP) and d.SiP and (d.SiP_dim is not None):
                if d.SiP_dim != best_idsip_dim[env]:
                    continue
                key = (False, False, True, "best", False, None, None, None)

            elif d.ID and (not d.SaP) and d.SiP and (d.SiP_dim is not None):
                if d.SiP_dim != best_idsip_dim[env]:
                    continue
                key = (True, False, True, "best", False, None, None, None)

        groups.setdefault(key, {})[env] = d

    env_metrics = {}
    for env in env_order:
        env_metrics[env] = base_metrics + (["success_rate"] if include_success else [])

    # -----------------------------
    # Minima for bolding W_* per env
    # -----------------------------
    mins = {env: {m: float("inf") for m in min_metrics} for env in env_order}
    for env in env_order:
        for m in min_metrics:
            for env_map in groups.values():
                if env in env_map:
                    v = getattr(env_map[env], m, None)
                    if v is not None:
                        mins[env][m] = min(mins[env][m], v)

    # -----------------------------
    # Helpers for printing
    # -----------------------------
    def method_label(key):

        if isinstance(key[0], str) and key[0] == "explicit":
            return {
                "DP": r"DP~\cite{chi_diffusion_2024}",
                "DP+PIP": r"PIP~\cite{chen_exploiting_2025}",
                "PTP": r"PTP~\cite{torne_past-token_2025}",
            }.get(key[1], key[1])

        ID, SaP, SiP, SiP_dim, PIP, PIP_epsilon, PIP_window_size, PIP_sigma = key

        is_baseline = not (ID or SaP or SiP or PIP)
        if is_baseline:
            label = r"DP~\cite{chi_diffusion_2024}"
        elif PIP:
            label = r"PIP~\cite{chen_exploiting_2025}"
            if not combine_pip:
                label += f" ({PIP_epsilon}, {PIP_window_size}, {PIP_sigma})"
        else:
            label_components = []
            if ID:
                label_components.append("ID")
            if SaP:
                label_components.append("SaP")
            if SiP:
                label_components.append("SiP")
            label = "TC-DP"
            if len(label_components) > 0:
                label += " (" + "+".join(label_components) + ")"
            if SiP and SiP_dim not in (None, "best"):
                label += rf" [$d={SiP_dim}$]"
        return label

    def fmt_cell(env, d, m):
        v = getattr(d, m, None)
        if v is None:
            return "--"

        if m == "success_rate":
            return f"{float(v):.1f}"

        s = f"{float(v):.2f}"
        if m == "W_total" and abs(float(v) - mins[env][m]) < 5e-3:
            return rf"\textbf{{{s}}}"
        if m != "W_total":
            return rf"\textcolor{{lightgray}}{{{s}}}"
        return s

    def print_header_combined():
        colspec = "l" + "".join("c" * len(env_metrics[env]) for env in env_order)
        env_headers = " & ".join(
            [rf"\multicolumn{{{len(env_metrics[env])}}}{{c}}{{\textbf{{{env_titles.get(env, env)}}}}}" for env in env_order]
        )
        cmidrule_start = 2
        cmidrules = []
        metric_headers = {
            "W_pause": r"$W_{\text{pause}}$",
            "W_episode": r"$W_{\text{episode}}$",
            "W_total": r"$W_{\text{total}}$",
            "success_rate": r"Success (\%)",
        }
        metric_row_parts = []
        for env in env_order:
            env_metric_count = len(env_metrics[env])
            cmidrules.append(rf"\cmidrule(lr){{{cmidrule_start}-{cmidrule_start + env_metric_count - 1}}}")
            cmidrule_start += env_metric_count
            metric_row_parts.extend(metric_headers[m] for m in env_metrics[env])
        cmidrules_str = " ".join(cmidrules)
        metric_row = " & ".join(metric_row_parts)

        print(textwrap.dedent(f"""
        \\begin{{tabular}}{{{colspec}}}
        \\toprule
        \\textbf{{Method}} &
        {env_headers} \\\\
        {cmidrules_str}
        &
        {metric_row} \\\\
        \\midrule
        """).strip())

    def print_header_single(env_title: str):
        metric_headers = {
            "W_pause": r"$W_{\text{pause}}$",
            "W_episode": r"$W_{\text{episode}}$",
            "W_total": r"$W_{\text{total}}$",
            "success_rate": r"Success (\%)",
        }
        env_name = next(name for name, title in env_titles.items() if title == env_title) if env_title in env_titles.values() else env_title
        env_metric_list = env_metrics[env_name]
        colspec = "l" + "c" * len(env_metric_list)
        hdr = " & ".join(metric_headers[m] for m in env_metric_list)

        print(textwrap.dedent(f"""
        \\begin{{tabular}}{{{colspec}}}
        \\toprule
        \\textbf{{Method}} & \\multicolumn{{{len(env_metric_list)}}}{{c}}{{\\textbf{{{env_title}}}}} \\\\
        \\cmidrule(lr){{2-{1+len(env_metric_list)}}}
        & {hdr} \\\\
        \\midrule
        """).strip())

    def print_footer():
        print(textwrap.dedent(r"""
        \bottomrule
        \end{tabular}
        """).strip())

    # -----------------------------
    # Print table(s)
    # -----------------------------
    if not separate_tables_per_env:
        print_header_combined()

        for key, env_map in groups.items():
            row = [method_label(key)]
            for env in env_order:
                d = env_map.get(env)
                if d is None:
                    row += ["--"] * len(env_metrics[env])
                else:
                    row += [fmt_cell(env, d, m) for m in env_metrics[env]]
            print(" & ".join(row) + r" \\")
        print_footer()
        return

    # two separate tables: Env. A then Env. B
    for env in env_order:
        print_header_single(env_titles.get(env, env))
        for key, env_map in groups.items():
            if env not in env_map:
                continue  # <-- skip methods absent in this environment

            d = env_map[env]
            row = [method_label(key)] + [fmt_cell(env, d, m) for m in env_metrics[env]]
            print(" & ".join(row) + r" \\")
        print_footer()
        print()  # blank line between tables


def create_main_paper_tex_table(results: dict[str, ParsedDatasetInfo]) -> None:
    """Print the archived single-seed temporal-ambiguity results table.

    The table includes success rate and only the rows reported in the main
    comparison; individual ID, SaP, and SiP ablations are retained in the
    full-sweep table instead. Use analysis.report for the current three-seed table.
    """
    from temporal_diffusion_policy.analysis.report import aggregate_results, render_table

    report = load_report_config("archive_older_single_seed_comparison")
    print(render_table(report, aggregate_results(report, results)), end="")


def create_full_sweep_tex_table(results: dict[str, ParsedDatasetInfo]) -> None:
    """Print the reproducible Appendix sweep for WaitAtGoal and LiftQA.

    The inputs are ``FULL_SWEEP_STUDIES``.  PTP is an explicit table method,
    so it follows the detailed DP+PIP sweep rows and precedes TC-DP ablations.
    """
    create_tex_table_from_results(
        results,
        best_only=False,
        combine_pip=False,
        include_success=True,
        env_order=["WaitAtGoal", "LiftQA"],
    )
    
        
if __name__ == "__main__":
    main()
    
