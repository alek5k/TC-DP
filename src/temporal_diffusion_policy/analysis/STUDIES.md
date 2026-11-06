# Paper studies, reports, and figures

Study JSON files remain the dataset lists. They supply the datasets for both
tables and figures. The original TC-DP paper study files are preserved unchanged.

The report presets in reports/ select the study, environments, methods, metrics,
row order, and formatting. The same selectors feed tables and figures:

- paper_main.json (default): the current three-seed simulation results alongside
  the single-policy LiftQA-Real results. Use this for the current paper table.
- main_multiseed.json: the current three-seed comparison for WaitAtGoal and LiftQA.
- appendix_ablation.json: DP and the standalone ID, SaP, and SiP ablations.

The main_multiseed.json report selects the separate tc_dp_paper_multiseed study
for WaitAtGoal and LiftQA. Its two study files list the completed TC-DP and
August DP/PIP evaluations and the completed three-seed PTP evaluations from ldp.
Original study files remain unchanged.

From the TemporalDiffusionPolicy repository root:

```bash
python src/temporal_diffusion_policy/analysis/report.py
python src/temporal_diffusion_policy/analysis/report.py --check-inputs
python src/temporal_diffusion_policy/analysis/report.py --config main_multiseed
```

Add --check-inputs to validate the listed datasets without generating outputs.
Use --output path.tex to override the table destination. Each table has a
companion JSON file containing its metric values.

Add --figures to generate figures from the same study and row selection, or use
--figures-only to generate only figures:

```bash
python src/temporal_diffusion_policy/analysis/report.py --figures
python src/temporal_diffusion_policy/analysis/report.py --figures-only
```

Figures go into a report-specific folder beside the generated table. Reports
use only explicitly listed datasets and never discover substitute inputs.
No report copies results into the manuscript automatically.

The existing appendix entry point remains supported:

```bash
python src/temporal_diffusion_policy/analysis/ablation_table.py
```

The multi-seed reports require training seeds 1, 2, 3 for DP/PIP/TC-DP and
42, 43, 44 for PTP, with 200 evaluation episodes per run (evaluation seeds
1000–1199). Each metric is computed per training seed, then averaged with a
sample standard deviation (ddof=1). PIP uses the corresponding DP
checkpoint with an inference-time intervention. LiftQA uses SiP d=256 and
WaitAtGoal uses d=32; the separate d=32 LiftQA experiment is not selected.

The default paper_main report adds LiftQA-Real from the preserved tc_dp_paper
study, with one policy and 50 rollouts per method; no training-seed SD is
assigned to those results. It writes outputs/analysis/reports/paper_main.tex
and paper_main.json. The manuscript inputs the generated tabular from
tables/temporal_results.tex; copy the generated table there after regeneration.

Paper formatting uses two decimal places for Wasserstein distances and one
for success rates, with inline standard deviations and SR (%) headings.
Small nonzero SDs use extra decimal places when they would otherwise round
to zero; exactly zero observed SDs retain the usual precision. The lowest mean
W_total is bold and the second-lowest distinct mean is underlined, ranked
before rounding. Only the means are marked; SD overlap does not define a tie.
Success rates are black, with the highest mean bold and second-highest distinct
mean underlined; exact ties share the same marking. These settings are explicit in the
current report presets.

Exploratory figure generation can still select a study directly:

```bash
python src/temporal_diffusion_policy/analysis/waittime.py \
    --env_name WaitAtGoal --studies '["tc_dp_paper"]' --auto_resolve_datasets False
```

The bare waittime command analyzes the three paper environments using the
original tc_dp_paper study. Use the report entry point for the current multi-seed
paper table, and disable discovery when selecting an exact study list.

Robomimic remains independent in its own repository:

Its default combined-run analysis computes Wasserstein distance for each
training-seed rollout file against the demonstrations, then reports the mean
and sample standard deviation across files. Success rates use the same sample SD.
Pooled rollout samples are retained for distribution figures, but the reported
timing distance is the mean of per-seed distances. A single run has no
across-seed SD. Tables mark the best and second-best means for both metrics.

```bash
python -m robomimic.scripts.analysis_temporal \
    --task_names '["can", "lift", "square"]' --dataset_split ph \
    --rollout_sampling_mode all --rollout_selection_mode combine
```

## Completed WaitAtGoal results

The waitatgoal_multiseed preset selects the completed DP, PIP, PTP, ID+SaP,
and ID+SiP three-seed evaluations from the same tc_dp_paper_multiseed study.

```bash
python src/temporal_diffusion_policy/analysis/report.py --config waitatgoal_multiseed
```

Outputs: outputs/analysis/paper_multiseed/waitatgoal.tex and waitatgoal.json.

## Completed LiftQA results

The liftqa_multiseed preset selects the completed DP, PIP, PTP, ID+SaP, and
ID+SiP (d=256) evaluations. It uses the same mean and sample standard deviation
as the WaitAtGoal preset, with the method-specific training seeds listed above.

```bash
python src/temporal_diffusion_policy/analysis/report.py --config liftqa_multiseed
```

Outputs: outputs/analysis/paper_multiseed/liftqa.tex and liftqa.json.
