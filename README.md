# TC-DP supplementary materials

This bundle accompanies Mitigating Temporal Failures in Diffusion-Based Robotic
Imitation Learning. This bundle contains three separate repositories, each
with its own installation instructions and dependencies. Install them in separate
environments.

The three code folders are at the root of this repository. Paths below are relative
to the relevant repository unless stated otherwise.

## Repositories

### [TemporalDiffusionPolicy](TemporalDiffusionPolicy/)

TC-DP and DP/PIP baselines; WaitAtGoal, LiftQA, and LiftQA-Real environments;
temporal analysis. Installation and usage: README.md

### [robomimic](robomimic/)

TC-DP and baseline experiments on the Robomimic benchmark.
Installation and usage: README_TC-DP.md

### [ldp](ldp/)

Past-Token Prediction (PTP) baseline, including WaitAtGoal and LiftQA integration.
Installation and usage: README.md and TEMPORAL_ENVS.md

## Video

TC-DP vs. DP - LiftQA-Real - Real World Env - Failure Modes.mp4, in the supplementary ZIP,
compares TC-DP and DP failure modes in the real-world environment.

## Implementation

In TemporalDiffusionPolicy, temporal encodings are in
src/temporal_diffusion_policy/temporal_encodings.py.
The same package contains training.py, inference.py, and the task environments
under envs/, including the distinct simulated and real-world LiftQA environments.

In robomimic, the implementation uses robomimic/utils/temporal_encodings.py
and TemporalEncodingWrapper in robomimic/envs/wrappers.py.
Experiment configurations are in robomimic/exps/temporaldp/.

For PTP in ldp, start with the three-training-seed protocol in TEMPORAL_ENVS.md
and the saved configurations in experiment_configs/paper_ptp_seeds/20261029_234821/.
Use eval_temporal_policy.sh to evaluate each selected checkpoint. The task
configurations are
diffusion_policy/config/task/waitatgoal_image.yaml and
diffusion_policy/config/task/liftqa_image.yaml.

## Data and analysis

Datasets, trained checkpoints, recorded rollouts, and generated analysis outputs
are excluded. The repository READMEs contain download instructions and explain
the expected data layouts. Set local dataset and checkpoint paths before training,
evaluation, or analysis; some configurations retain paths from the original
working environment.

The current main-paper data selection contains three custom demonstration
datasets, 63 selected checkpoints, and 70 rollout datasets including images.
Can, Lift and Square demonstrations are obtained from robomimic upstream and
are not duplicated. Ablations are outside this data upload selection.

Custom environments and PTP:
https://huggingface.co/datasets/erdrsvhr9/temporally_conditioned_diffusion_policy
Robomimic checkpoints and rollouts:
https://huggingface.co/datasets/erdrsvhr9/temporally_conditioned_diffusion_policy_robomimic

Download from each repository root using its exact paper selection:
TemporalDiffusionPolicy:
```bash
python src/temporal_diffusion_policy/tools/dataset_upload.py --download
```
robomimic and ldp (run separately in each repository):
```bash
python dataset_upload.py --download
```

These commands extract Zarr archives into their expected directories. PTP
uses the demonstrations downloaded into TemporalDiffusionPolicy. Adjust local
paths in the analysis study lists and configurations before running reports.
The main paper tables and plots use this selection; the separate appendix
ablation table requires its additional ablation data.

Compacted inference datasets store retained image arrays as uint8 pixels in
the range 0-255. LiftQA keeps full_image and omits the redundant agentview_image
field; LiftQA-Real retains full_image_raw at its original resolution. Convert
uint8 policy images to float32 and divide by 255 before model use. The image
array's _uint8_compaction attributes record its original dtype and the divisor
needed to restore its original pixel scale. Any rounded fractional images are
marked explicitly in those attributes. The table and distribution analyses
read non-image arrays and do not require image conversion.

For the ambiguity-environment results and distribution plots in
TemporalDiffusionPolicy, see src/temporal_diffusion_policy/analysis/report.py and
the adjacent STUDIES.md, reports/, and study_*.json files. The default paper_main
report uses mean and sample standard deviation across three training seeds for
WaitAtGoal and LiftQA, alongside single-policy LiftQA-Real results. The separate
appendix ablation table retains the original single-training-seed evaluations.
Once the dependencies and analysis datasets are installed, run these from the
TemporalDiffusionPolicy repository root:

```bash
python -m temporal_diffusion_policy.analysis.report
python -m temporal_diffusion_policy.analysis.report --figures
python -m temporal_diffusion_policy.analysis.ablation_table
```

Use the default paper_main report for the current three-seed comparison.
For simulation-only results, use --config main_multiseed; the environment-specific
presets are --config waitatgoal_multiseed and --config liftqa_multiseed.
Review the study lists and set dataset paths to your local copies before running
analysis. Add --check-inputs to validate the listed datasets without generating
outputs. The default report writes
outputs/analysis/reports/paper_main.tex and a companion paper_main.json file.

In robomimic, result analysis is implemented in
robomimic/scripts/analysis_temporal.py. Combined-run analysis reports means and
sample standard deviations for episode-duration distance and success rate across
training-seed rollout files. From the robomimic repository root, run:

```bash
python -m robomimic.scripts.analysis_temporal --task_names '["can", "lift", "square"]' --dataset_split ph --rollout_sampling_mode all --rollout_selection_mode combine
```

Use one rollout file for each of the three training seeds per method and task.
The expected directory layout is documented in README_TC-DP.md.

Training and evaluation commands are documented in each repository's README.
