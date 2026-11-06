"""Compatibility entry point for the configured single-encoding appendix table."""

import argparse
from pathlib import Path

from temporal_diffusion_policy.analysis.report import aggregate_results, load_report_config, render_table, run_report
from temporal_diffusion_policy.analysis.waittime import ParsedDatasetInfo

REPO_ROOT = Path(__file__).resolve().parents[3]


def create_ablation_tex_table(results: dict[str, ParsedDatasetInfo]) -> str:
    report = load_report_config("appendix_ablation")
    try:
        summary = aggregate_results(report, results)
    except ValueError as error:
        message = str(error)
        if "parsed runs, got 0" in message:
            raise ValueError(f"Missing ablation runs: {message}") from error
        if "parsed runs, got" in message:
            raise ValueError(f"Multiple ablation runs: {message}") from error
        raise
    return render_table(report, summary)


def generate_ablation_table(output: Path) -> None:
    run_report("appendix_ablation", output=output)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path,
                        default=REPO_ROOT / load_report_config("appendix_ablation")["output"])
    generate_ablation_table(parser.parse_args().output)


if __name__ == "__main__":
    main()
