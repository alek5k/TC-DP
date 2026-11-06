# TC-DP paper reproduction

This folder contains the paper-specific data selection, Hugging Face dataset
card and reproduction instructions. For installation, collection, training and
inference workflows, see the [main README](../README.md).

Run every command below from the TemporalDiffusionPolicy repository root,
with the project environment active. Code and data paths in command examples
are relative to that root.

- [paper_upload_files.json](paper_upload_files.json): exact main-paper upload and download selection.
- [PAPER_DATASET_CARD.md](PAPER_DATASET_CARD.md): dataset description published on Hugging Face.

## Paper data

The current main-paper data are hosted at
[TC-DP data](https://huggingface.co/datasets/erdrsvhr9/temporally_conditioned_diffusion_policy).
The exact selection in `tcdp_paper/paper_upload_files.json` contains the WaitAtGoal, LiftQA
and LiftQA-Real demonstrations, 21 selected DP/TC-DP checkpoints and 28 recorded
rollouts. The same Hugging Face repository contains the six PTP checkpoints and
six PTP rollouts under `ldp/`; download those from the ldp repository.
Ablations and exploratory runs are excluded from this upload selection.

Zarr archives retain all image arrays and metadata. Inference images are uint8
pixels (0–255); convert to float32 and divide by 255 when normalized model inputs
are needed. LiftQA keeps `full_image`; LiftQA-Real keeps `full_image_raw`.
Demonstration images retain their original representation.

### Download the current paper data

```bash
# From the repository root, in the installed Python environment:
python src/temporal_diffusion_policy/tools/dataset_upload.py --download
# Or use the existing shell entry point:
./scripts/download_all.sh
```

The downloader extracts each `.zarr.zip` beside its expected dataset path and
restores checkpoints and saved configurations. The shell scripts
`download_demos.sh`, `download_checkpoints.sh`, and `download_inference.sh`
select individual categories using the same file list. This avoids pulling older
experiments into the current comparison. Set local paths in the analysis study
lists, including the PTP paths in your sibling ldp repository, before analysis.

### Upload the selected paper data

```bash
python src/temporal_diffusion_policy/tools/dataset_upload.py --dry-run
python src/temporal_diffusion_policy/tools/dataset_upload.py
```

Upload requires Hugging Face write access. The script archives selected Zarr
directories without altering their contents and uploads only the explicit file
list. It preserves existing hosted files and visibility. Staging defaults to
`/tmp/tcdp-paper-hf-upload/TemporalDiffusionPolicy`; use `--stage-dir` to change it.

## Results tables and figures

Generate the current paper results table, optionally with matching figures:

```bash
python -m temporal_diffusion_policy.analysis.report
python -m temporal_diffusion_policy.analysis.report --figures
```

The default `paper_main` report uses three-training-seed means and sample
standard deviations for WaitAtGoal and LiftQA, alongside the single-policy
LiftQA-Real results. Dataset lists and report presets are documented in
[`analysis/STUDIES.md`](../src/temporal_diffusion_policy/analysis/STUDIES.md).
Install the analysis datasets and adjust their paths before running.
Use `--config main_multiseed` for the three-seed simulation comparison without
LiftQA-Real, or `--config waitatgoal_multiseed` / `--config liftqa_multiseed`
for an individual environment.

## Real-world success and appendix ablation

For LiftQA-Real rollout success, analysis requires strictly more than two
seconds accumulated in `wait_time` and an episode shorter than its recorded
`max_steps` limit (400 for the paper runs). Early termination is the proxy for
the environment's `COMPLETED` phase, assuming no manual stops. This success
criterion does not change the Wasserstein calculations or simulation criteria.

The appendix ablation uses additional datasets outside the main-paper upload
selection. Once those inputs are available, generate its table independently:

```bash
python -m temporal_diffusion_policy.analysis.ablation_table
```

This writes `outputs/analysis/waittime/temporal_encoding_ablation.tex` with
DP, ID-only, SaP-only, and SiP-only rows for LiftQA and WaitAtGoal, reporting
`W_pause`, `W_episode`, and `W_total`. Inputs come from the two `tc_dp_paper` study JSON
files; SiP uses their explicitly configured dimensions (256 and 32,
respectively). Metric settings match the main analysis defaults. Use
`--output /path/to/table.tex` to choose another destination.
