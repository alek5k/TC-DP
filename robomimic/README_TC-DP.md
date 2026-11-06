# Overview
This is a fork of the [Robomimic study](https://robomimic.github.io/study/) study and [repository](https://github.com/ARISE-Initiative/robomimic) for the Temporally-Conditioned Diffusion Policy Paper.


Temporal conditioning encodings are implemented in:
`robomimic/utils/temporal_encodings.py`

Temporal encoding wrapper is implemented in:
`robomimic/envs/wrappers.py`, class `TemporalEncodingWrapper`

Temporal analysis is implemented in `robomimic/scripts/analysis_temporal.py`.


### Installing
Specific installation instructions for robomimic can be found [here](https://robomimic.github.io/docs/introduction/installation.html).

```bash
conda create -n robomimic2 python=3.10
conda activate robomimic2
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
cd robomimic
pip install -e .
pip install "robosuite==1.5.1"
```

### Download dataset
Dataset info can be found [here](https://robomimic.github.io/docs/datasets/robomimic_v0.1.html).

```
cd robomimic/scripts
python download_datasets.py --tasks sim --dataset_types ph --hdf5_types raw 
```

Then postprocess the dataset to get image datasets:

```
./extract_obs_from_raw_datasets.sh
```

### Download TC-DP trained checkpoints and inference datasets

The [TC-DP robomimic data](https://huggingface.co/datasets/erdrsvhr9/temporally_conditioned_diffusion_policy_robomimic)
contain the 36 selected checkpoints and 36 recorded evaluations for BC, BC-RNN,
DP and TC-DP: three training seeds per method for Can, Lift and Square.
The exact main-paper selection is in `paper_upload_files.json`.

```bash
python dataset_upload.py --download
# Equivalent shell entry point:
./download_checkpoints_and_inference_datasets.sh
```

Use a clean rollout directory containing only these selected files per method
and task when combining results. Download demonstrations from robomimic using
the instructions above; Can, Lift and Square demonstrations are not duplicated
in the TC-DP upload. Rollout HDF5 files already contain no image arrays.

To upload the selected checkpoints, saved configurations and rollout files:

```bash
python dataset_upload.py --dry-run
python dataset_upload.py
```

Upload requires Hugging Face write access. Only the explicit current-paper
selection is uploaded; demonstrations and ablations are excluded. Existing
hosted files and repository visibility are preserved. Use `--stage-dir` to
change the default `/tmp/tcdp-paper-hf-upload/robomimic` staging directory.

### Config Setup
The configs for the paper are already included in the repository. Within these configs, the `train/data` config variables need to be set to correspond to your repo location in the following configs:

```
robomimic/exps/temporaldp/can/ph/image/bc.json
robomimic/exps/temporaldp/can/ph/image/bc_rnn.json
robomimic/exps/temporaldp/can/ph/image/diffusion_policy_mod.json
robomimic/exps/temporaldp/can/ph/image/tc_diffusion_policy_mod.json
```

Paths are relative to the repository root. Equivalent configurations are provided
for `lift` and `square`. The `_mod` DP and TC-DP configurations are used for the
paper comparison.

### Training
Replace ALGO below with:
bc, bc_rnn, diffusion_policy_mod and tc_diffusion_policy_mod.

Train three independent seeds for every method and task:

```bash
conda activate robomimic2 && cd ~/Repos/robomimic
ENV=lift
SPLIT=ph
ALGO=tc_diffusion_policy_mod
GPU=1
for SEED in 1 2 3; do
    CUDA_VISIBLE_DEVICES=${GPU} MUJOCO_GL=egl python robomimic/scripts/train.py --config robomimic/exps/temporaldp/${ENV}/${SPLIT}/image/${ALGO}.json --seed=${SEED}
done
```

### Current paper analysis

Use the combined results from all three training seeds for Can, Lift, and Square.
After downloading or generating the demonstration and rollout datasets, run from
the repository root:

```bash
python -m robomimic.scripts.analysis_temporal \
    --task_names '["can", "lift", "square"]' --dataset_split ph \
    --rollout_sampling_mode all --rollout_selection_mode combine
```

This is also the default analysis configuration. Put the demonstration dataset at
`datasets/<task>/ph/image_v15.hdf5` and one rollout HDF5 file per training seed in
each of these directories:

```text
rollouts/<task>/ph/bc/
rollouts/<task>/ph/bc_rnn/
rollouts/<task>/ph/diffusion_policy_mod/
rollouts/<task>/ph/tc_diffusion_policy_mod/
```

Keep only the three intended training-seed files in each directory: the analysis
combines all matching files. Episode-duration Wasserstein distances are computed
per file against the demonstrations, then reported as mean and sample standard
deviation across training seeds. Success rates use the same aggregation. All
rollouts are used; avoid `--rollout_selection_mode best` for this comparison.
Distribution plots pool rollouts across seeds. Plots are written under
`robomimic/scripts/analysis_temporal/`, and the LaTeX table is printed to stdout.
 
### Evaluations
Replace with your RUN_ID generated during training.
```bash
conda activate robomimic2 && cd ~/Repos/robomimic
for RUN_ID in 20260715120741 20260716000116 20260716105927; do
    N_ROLLOUTS=100
    ENV=lift
    ALGO=tc_diffusion_policy_mod
    GPU=1
    SPLIT=ph
    SEED=$((RUN_ID % 4294967295))
    BASE="trained_models/temporaldp/${ALGO}/${ENV}/${SPLIT}/image/trained_models/temporaldp_${ALGO}_${ENV}_${SPLIT}_image/${RUN_ID}"
    AGENT=$(find "$BASE/models" -name "*.pth" | awk 'match($0,/success_([0-9.]+)/){s=substr($0,RSTART+8,RLENGTH-8); print s,$0}' | sort -nr | head -n1 | cut -d" " -f2-)
    NAME=$(basename "$AGENT")
    DATASET="rollouts/${ENV}/${SPLIT}/${ALGO}/${RUN_ID}_${NAME}.hdf5"
    INFO="rollouts/${ENV}/${SPLIT}/${ALGO}/${RUN_ID}_${NAME}_info.txt"

    mkdir -p "$(dirname "$INFO")"

    printf "RUN_ID=%s\nSEED=%s\nN_ROLLOUTS=%s\nENV=%s\nALGO=%s\nGPU=%s\nSPLIT=%s\nBASE=%s\nAGENT=%s\nNAME=%s\nDATASET=%s\n" "$RUN_ID" "$SEED" "$N_ROLLOUTS" "$ENV" "$ALGO" "$GPU" "$SPLIT" "$BASE" "$AGENT" "$NAME" "$DATASET" > "$INFO"

    CUDA_VISIBLE_DEVICES=${GPU} MUJOCO_GL=egl python robomimic/scripts/run_trained_agent.py --agent="$AGENT" --dataset_path="$DATASET" --n_rollouts=${N_ROLLOUTS} --seed=${SEED} --dataset_obs
done
```
