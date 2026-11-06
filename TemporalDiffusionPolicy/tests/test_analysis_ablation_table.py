"""Protect the compact table's environment and single-encoding selection."""

from dataclasses import replace

import pytest

from temporal_diffusion_policy.analysis.ablation_table import create_ablation_tex_table
from temporal_diffusion_policy.analysis.waittime import DatasetType, ParsedDatasetInfo


def _results():
    results = {}
    for environment, base in (("LiftQA", 10), ("WaitAtGoal", 20)):
        for index, flags in enumerate(((True, False, False), (False, True, False), (False, False, True))):
            key = f"{environment}:{index}"
            results[key] = ParsedDatasetInfo(
                path=key,
                config={"env_name": environment},
                is_demo_dataset=False,
                dataset_type=DatasetType.INFERENCE,
                ID=flags[0],
                SaP=flags[1],
                SiP=flags[2],
                W_pause=base + index + 0.1,
                W_episode=base + index + 0.2,
                W_total=2 * (base + index) + 0.3,
            )
        results[f"{environment}:DP"] = replace(
            results[f"{environment}:0"],
            path=f"{environment}:DP",
            ID=False,
            table_method="DP",
            W_pause=base + 3.1,
            W_episode=base + 3.2,
            W_total=2 * base + 6.3,
        )
    return results


def test_ablation_table_keeps_environment_columns_and_excludes_other_variants():
    results = _results()
    original = results["LiftQA:0"]
    results["combined"] = replace(original, path="combined", SiP=True, W_pause=999)
    results["real"] = replace(original, path="real", config={"env_name": "LiftQA-Real"}, W_pause=999)
    results["pip"] = replace(original, path="pip", PIP=True, W_pause=999)
    results["ptp"] = replace(original, path="ptp", ID=False, table_method="PTP", W_pause=999)
    table = create_ablation_tex_table(results)
    assert r"DP~\cite{chi_diffusion_2024} & 13.10 & 13.20 & 26.30 & 23.10 & 23.20 & 46.30" in table
    assert r"TC-DP (ID) & 10.10 & 10.20 & 20.30 & 20.10 & 20.20 & 40.30" in table
    assert r"TC-DP (SaP) & 11.10 & 11.20 & 22.30 & 21.10 & 21.20 & 42.30" in table
    assert r"TC-DP (SiP) & 12.10 & 12.20 & 24.30 & 22.10 & 22.20 & 44.30" in table
    assert "999" not in table
    assert r"$W_{\text{total}}$" in table and "Success" not in table


@pytest.mark.parametrize("missing", ["WaitAtGoal:2", "WaitAtGoal:DP"])
def test_ablation_table_rejects_missing_variant(missing):
    results = _results()
    del results[missing]
    with pytest.raises(ValueError, match="Missing ablation"):
        create_ablation_tex_table(results)


def test_ablation_table_rejects_multiple_runs_for_same_variant():
    results = _results()
    results["other_dim"] = replace(results["LiftQA:2"], path="other_dim", SiP_dim=64)
    with pytest.raises(ValueError, match="Multiple ablation runs"):
        create_ablation_tex_table(results)
