"""Render configured paper tables and figures from preserved study lists."""

import argparse
import json
from pathlib import Path

import numpy as np

from temporal_diffusion_policy.analysis.study_configs import load_study_configs


REPO_ROOT = Path(__file__).resolve().parents[3]
REPORT_DIRECTORY = Path(__file__).with_name("reports")
FLAG_ATTRIBUTES = {
    "ID": "use_idleness_encoding", "SaP": "use_saturating_progress_encoding",
    "SiP": "use_sinusoidal_progress_encoding", "PIP": "use_pause_induced_perturbation",
}
METRIC_HEADERS = {
    "W_pause": r"$W_{\text{pause}}$", "W_episode": r"$W_{\text{episode}}$",
    "W_total": r"$W_{\text{total}}$", "success_rate": r"Success (\%)",
}


def format_spread(value, precision=2):
    """Add decimal places when rounding would hide a nonzero spread."""
    text = f"{value:.{precision}f}"
    while value > 0 and float(text) == 0:
        precision += 1
        text = f"{value:.{precision}f}"
    return text


def _uncertainty(report):
    return report.get("uncertainty", report.get("confidence_interval") or
                      ("standard_deviation" if report.get("aggregation") == "mean" else None))


def load_report_config(config="paper_main"):
    path = Path(config)
    if not path.is_file():
        path = REPORT_DIRECTORY / (path.name if path.suffix else path.name + ".json")
    report = json.loads(path.read_text(encoding="utf-8"))
    for key in ("name", "study", "environments", "methods", "metrics", "aggregation", "output"):
        if not report.get(key):
            raise ValueError(f"Missing report field {key!r} in {path}")
    for key in ("environments", "methods", "metrics"):
        if len(set(report[key])) != len(report[key]):
            raise ValueError(f"Duplicate {key} in {path}")
    if set(report["metrics"]) - METRIC_HEADERS.keys():
        raise ValueError(f"Unknown report metrics in {path}")
    if report["aggregation"] not in ("none", "mean"):
        raise ValueError("Report aggregation must be 'none' or 'mean'")
    if _uncertainty(report) not in (None, "standard_deviation", "student_t_95"):
        raise ValueError("Supported uncertainty: standard_deviation or student_t_95")
    if _uncertainty(report) and report["aggregation"] != "mean":
        raise ValueError("Uncertainty requires mean aggregation")
    for selector in report.get("selectors", {}).values():
        if set(selector) - {"table_method", "dataset_type", *FLAG_ATTRIBUTES}:
            raise ValueError(f"Unknown selector fields: {selector}")
    return report


def _environment_report(report, environment):
    """Allow single-policy real-world results beside multi-seed simulation."""
    return {**report, **report.get("environment_overrides", {}).get(environment, {})}


def _required_seeds(report, method):
    return report.get("training_seeds_by_method", {}).get(
        method, report.get("required_training_seeds", []))


def _matches(report, method, item, root=REPO_ROOT):
    """Use the same explicit predicates for study rows and parsed datasets."""
    selector = report.get("selectors", {}).get(method, {"table_method": method})
    raw = isinstance(item, dict)
    for key, expected in selector.items():
        if key in FLAG_ATTRIBUTES:
            continue
        actual = item.get(key) if raw else getattr(item, key, None)
        if actual != expected:
            return False
    flag_keys = selector.keys() & FLAG_ATTRIBUTES.keys()
    if not flag_keys:
        return True
    attrs = json.loads((root / item["path"] / ".zattrs").read_text()) if raw else None
    return all((bool(attrs.get(FLAG_ATTRIBUTES[key], False)) if raw
                else getattr(item, key)) == selector[key] for key in flag_keys)


def _allowed_missing(report, environment, method):
    return method in report.get("allow_missing", {}).get(environment, [])


def validate_study_runs(report, environment, runs, root=REPO_ROOT):
    report = _environment_report(report, environment)
    demos = [r for r in runs if r.get("dataset_type") == "Demo"]
    if len(demos) != 1:
        raise ValueError(f"{environment}: expected exactly one demonstration dataset")
    paths = [r["path"] for r in runs]
    if len(set(paths)) != len(paths):
        raise ValueError(f"{environment}: duplicate dataset paths")
    for method in report["methods"]:
        required = _required_seeds(report, method)
        group = [r for r in runs if r not in demos and _matches(report, method, r, root)]
        if not group and _allowed_missing(report, environment, method):
            continue
        if required:
            seeds = [r.get("training_seed") for r in group]
            if sorted(seeds, key=str) != sorted(required, key=str):
                raise ValueError(f"{environment}/{method}: expected training seeds "
                                 + ", ".join(map(str, required)))
        elif report["aggregation"] == "none" and len(group) != 1:
            raise ValueError(f"{environment}/{method}: expected one dataset, got {len(group)}")


def load_inputs(report, root=REPO_ROOT):
    """Select study entries explicitly; never discover or modify datasets."""
    configs = load_study_configs()
    inputs = {}
    for environment in report["environments"]:
        environment_report = _environment_report(report, environment)
        try:
            source = configs[environment][environment_report["study"]]
        except KeyError as error:
            raise ValueError(f"Missing study {environment}/{environment_report['study']}") from error
        selected = []
        for row in source:
            if row.get("dataset_type") == "Demo":
                selected.append(row)
                continue
            matches = [m for m in report["methods"] if _matches(report, m, row, root)]
            if len(matches) > 1:
                raise ValueError(f"Ambiguous report selection for {row['path']}: {matches}")
            if matches:
                selected.append(row)
        validate_study_runs(report, environment, selected, root)
        inputs[environment] = selected
    return inputs


def missing_inputs(report, inputs, root=REPO_ROOT):
    problems = []
    for environment, runs in inputs.items():
        for run in runs:
            path = root / run["path"]
            if "PLACEHOLDER" in run["path"]:
                problems.append(f"{environment}: replace {run['path']}")
            elif not (path / ".zgroup").is_file() or not (path / "meta/episode_ends/.zarray").is_file():
                problems.append(f"{environment}: missing dataset {path}")
            elif run.get("dataset_type") != "Demo":
                count = json.loads((path / "meta/episode_ends/.zarray").read_text())["shape"][0]
                expected = report.get("expected_episodes", {}).get(environment)
                if expected is not None and count != expected:
                    problems.append(f"{environment}: {path} has {count} episodes; expected {expected}")
    return problems


def parse_inputs(report, inputs):
    from temporal_diffusion_policy.analysis.waittime import AnalysisConfig, DatasetLocator, DatasetType, Study, parse_datasets

    results = {}
    for environment, runs in inputs.items():
        environment_report = _environment_report(report, environment)
        locators = [DatasetLocator(
            path=str(REPO_ROOT / r["path"]), label=r.get("label"),
            dataset_type=DatasetType(r["dataset_type"]),
            table_method=r.get("table_method"), aggregate_key=r.get("aggregate_key"),
        ) for r in runs]
        cfg = AnalysisConfig(env_name=environment, studies=[Study(environment_report["study"])],
                             auto_resolve_datasets=False, include_training_parameters=False,
                             **environment_report.get("analysis", {}))
        parsed = parse_datasets(cfg, locators)
        if any(d.config["env_name"] != environment for d in parsed.values()):
            raise ValueError(f"Dataset environment does not match {environment}")
        results.update({f"{environment}:{path}": d for path, d in parsed.items()})
    return results


def select_parsed_results(report, results, environment=None):
    environments = [environment] if environment else report["environments"]
    selected = {}
    for env in environments:
        demos = [(p, d) for p, d in results.items()
                 if d.config["env_name"] == env and d.is_demo_dataset]
        if len(demos) > 1:
            raise ValueError(f"{env}: multiple demonstration datasets")
        selected.update(demos)
        for method in report["methods"]:
            selected.update((p, d) for p, d in results.items()
                            if d.config["env_name"] == env and not d.is_demo_dataset
                            and _matches(report, method, d))
    return selected


def aggregate_results(report, results):
    from temporal_diffusion_policy.analysis.waittime import _mean_t_confidence_interval

    summary = {}
    for environment in report["environments"]:
        environment_report = _environment_report(report, environment)
        summary[environment] = {}
        for method in report["methods"]:
            runs = [d for d in results.values() if not d.is_demo_dataset
                    and d.config["env_name"] == environment and _matches(report, method, d)]
            if not runs and _allowed_missing(report, environment, method):
                continue
            expected_runs = len(_required_seeds(environment_report, method)) or 1
            if len(runs) != expected_runs:
                raise ValueError(f"{environment}/{method}: expected {expected_runs} parsed runs, got {len(runs)}")
            episodes = report.get("expected_episodes", {}).get(environment)
            if episodes and any(r.step_counts is not None and len(r.step_counts) != episodes for r in runs):
                raise ValueError(f"{environment}/{method}: each run must contain {episodes} episodes")
            metrics = {}
            for metric in report["metrics"]:
                values = [getattr(run, metric) for run in runs]
                if not all(v is not None and np.isfinite(v) and v >= 0 for v in values):
                    raise ValueError(f"Invalid {metric} for {environment}/{method}")
                mean = float(np.mean(values))
                std = float(np.std(values, ddof=1)) if len(values) > 1 else None
                lower = upper = margin = None
                uncertainty = _uncertainty(environment_report)
                if uncertainty == "standard_deviation":
                    margin = std
                elif uncertainty == "student_t_95":
                    mean, lower, upper = _mean_t_confidence_interval(values, 0.95)
                    margin = (upper - lower) / 2
                metrics[metric] = dict(mean=mean, std=std, lower=lower, upper=upper,
                                       margin=margin, uncertainty=uncertainty, per_seed=values)
            summary[environment][method] = metrics
    return summary


def render_table(report, summary):
    environments, methods, metrics = (report[k] for k in ("environments", "methods", "metrics"))
    style = report.get("format", {})
    nmetrics = len(metrics)
    lines = []
    if _uncertainty(report) == "standard_deviation":
        lines.append("% Mean +/- sample standard deviation across configured training seeds.")
    elif _uncertainty(report) == "student_t_95":
        lines.append("% Mean +/- 95% Student's t CI across configured training seeds.")
    columns = "@{}l"
    for env in environments:
        spacing = style.get("environment_spacing", {}).get(env)
        columns += ("c" * nmetrics if spacing is None else
                    r"@{\hspace{6pt}}c" + (rf"@{{\hspace{{{spacing}pt}}}}c" * (nmetrics - 1)))
    lines += [r"\begin{tabular}{" + columns + "@{}}", r"\toprule"]
    env_headers = " & ".join(rf"\multicolumn{{{nmetrics}}}{{c}}{{\textbf{{{e}}}}}" for e in environments)
    lines.append(r"\textbf{Method} & " + env_headers + r" \\")
    lines.append(" ".join(rf"\cmidrule(lr){{{2 + i*nmetrics}-{1 + (i+1)*nmetrics}}}"
                          for i in range(len(environments))))
    headers = {**METRIC_HEADERS, **style.get("headers", {})}
    lines.append("& " + " & ".join(headers[m] for _ in environments for m in metrics) + r" \\")
    lines.append(r"\midrule")
    ranked_metrics = set(style.get("bold_best", [])) | set(style.get("underline_second", []))
    ranked_metrics &= set(metrics)
    ranked = {(env, metric): sorted({summary[env][method][metric]["mean"]
                                    for method in methods if method in summary[env]},
                                   reverse=(metric == "success_rate"))
              for env in environments for metric in ranked_metrics}
    for method in methods:
        cells = [style.get("labels", {}).get(method, method)]
        for environment in environments:
            for metric in metrics:
                if method not in summary[environment]:
                    cells.append("--")
                    continue
                stats = summary[environment][method][metric]
                precision = style.get("precision", {}).get(metric, 2)
                number = value = f"{stats['mean']:.{precision}f}"
                order = ranked.get((environment, metric), [])
                bold = (metric in style.get("bold_best", []) and order
                        and stats["mean"] == order[0])
                underline = (metric in style.get("underline_second", []) and len(order) > 1
                             and stats["mean"] == order[1])
                if bold:
                    value = rf"\mathbf{{{value}}}"
                elif underline:
                    value = rf"\underline{{{value}}}"
                if stats["margin"] is not None:
                    margin = format_spread(stats["margin"], precision)
                    operator = r"{\pm}" if style.get("compact_ci") else r" \pm "
                    cell = f"${value}{operator}{margin}$"
                else:
                    if style.get("math_cells"):
                        cell = f"${value}$"
                    else:
                        cell = (rf"\textbf{{{number}}}" if bold else
                                rf"\underline{{{number}}}" if underline else number)
                if metric in style.get("gray_metrics", []):
                    cell = rf"\textcolor{{lightgray}}{{{cell}}}"
                cells.append(cell)
        suffix = rf"[{style['row_spacing_pt']}pt]" if style.get("row_spacing_pt") else ""
        lines.append(" & ".join(cells) + r" \\" + suffix)
    return "\n".join(lines + [r"\bottomrule", r"\end{tabular}"]) + "\n"


def generate_figures(report, results, output_dir):
    from temporal_diffusion_policy.analysis import waittime

    original_output = waittime.OUTPUT_DIR
    waittime.OUTPUT_DIR = output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    try:
        for environment in report["environments"]:
            environment_report = _environment_report(report, environment)
            selected = select_parsed_results(report, results, environment)
            if sum(d.is_demo_dataset for d in selected.values()) != 1:
                raise ValueError(f"{environment}: figures require one demonstration reference")
            cfg = waittime.AnalysisConfig(
                env_name=environment, studies=[waittime.Study(environment_report["study"])],
                auto_resolve_datasets=False, **environment_report.get("analysis", {}),
                **environment_report.get("figures", {}),
            )
            waittime.generate_graphs(cfg, selected,
                                     save_prefix=environment.lower().replace("-", "_"),
                                     study_name=report["name"])
    finally:
        waittime.OUTPUT_DIR = original_output


def run_report(config="paper_main", output=None, check_inputs=False,
               figures=False, figures_only=False):
    report = load_report_config(config)
    inputs = load_inputs(report)
    problems = missing_inputs(report, inputs)
    if problems:
        raise ValueError("Report inputs are not ready:\n" + "\n".join(problems))
    if check_inputs:
        print(f"Report {report['name']}: all configured datasets are ready.")
        return
    results = parse_inputs(report, inputs)
    summary = aggregate_results(report, results)
    destination = Path(output) if output else REPO_ROOT / report["output"]
    if not figures_only:
        table = render_table(report, summary)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(table, encoding="utf-8")
        destination.with_suffix(".json").write_text(json.dumps(summary, indent=2) + "\n")
        print(table)
        print(f"Saved {destination}")
    if figures or figures_only:
        generate_figures(report, results, destination.parent / (report["name"] + "_figures"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="paper_main", help="Report preset name or JSON path (default: paper_main).")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--check-inputs", action="store_true")
    parser.add_argument("--figures", action="store_true", help="Also generate figures from the same selected runs.")
    parser.add_argument("--figures-only", action="store_true")
    args = parser.parse_args()
    try:
        run_report(args.config, args.output, args.check_inputs, args.figures, args.figures_only)
    except (ValueError, FileNotFoundError) as error:
        parser.exit(1, str(error) + "\n")


if __name__ == "__main__":
    main()
