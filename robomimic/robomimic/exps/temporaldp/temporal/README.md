# Temporal environment baselines

Install the optional environment dependencies before using train-time rollouts:

```bash
conda activate robomimic2_temporalenvs_mujoco350
pip install -e ".[temporal-envs]"
```

Convert the source demonstrations before training. The converter preserves the
LDP split rule (2% validation, seed 42) and writes `mask/train` and
`mask/valid` filter keys.

```bash
python robomimic/scripts/conversion/zarr_to_temporal_hdf5.py \
  --task waitatgoal \
  --source /mnt/shared/TemporalDiffusionPolicy/data/demonstration/wait_at_goal_dataset_v4.zarr \
  --output datasets/temporal/waitatgoal_image.hdf5

python robomimic/scripts/conversion/zarr_to_temporal_hdf5.py \
  --task liftqa \
  --source /mnt/shared/TemporalDiffusionPolicy/data/demonstration/lift_qa_v2.zarr \
  --output datasets/temporal/liftqa_image.hdf5
```

The twenty configs enable robomimic's standard train-time rollout loop through
the native `EnvTemporal` adapter. It applies the LiftQA 3-D-to-7-D action map,
uses the tasks' final completion criterion for success, and supplies robomimic
with uint8 HWC images.

Every conversion verifies its schema, LDP-compatible split, actions, poses,
native velocity, terminal flags, and pixel round-trip before reporting
success. Re-check an existing output without rewriting it by adding `--verify`
to the same command.

For each environment, the following variants are available:

- `bc` / `bc_rnn`: the existing `can/ph/image` visual settings
  (SpatialSoftmax with a 76×76 random crop).
- `bc_nocrop`: SpatialSoftmax with no crop.
- `bc_mod` / `bc_rnn_mod`: SpatialMeanPool with the same crop.
- `bc_mod_nocrop` / `bc_rnn_mod_nocrop`: SpatialMeanPool with no crop.
- `diffusion_policy_mod`: Diffusion Policy with SpatialMeanPool, a 2-step
  observation horizon, a 16-step prediction horizon, and min-max action
  normalization.
- `tc_diffusion_policy_mod`: the same Diffusion Policy architecture augmented
  with native-velocity idleness and sinusoidal progress encodings.
- `diffusion_policy_mod_nocrop` / `tc_diffusion_policy_mod_nocrop`: the same
  respective Diffusion Policy variants with SpatialMeanPool and no crop.

All temporal variants use min-max action normalization. This is required for
the coordinate-valued WaitAtGoal actions and is also applied to LiftQA's
three-dimensional controls.

Train from the repository root, for example:

```bash
conda activate robomimic2_temporalenvs_mujoco350
python robomimic/scripts/train.py --config robomimic/exps/temporaldp/temporal/waitatgoal/bc.json
python robomimic/scripts/train.py --config robomimic/exps/temporaldp/temporal/liftqa/bc_rnn_mod_nocrop.json
python robomimic/scripts/train.py --config robomimic/exps/temporaldp/temporal/waitatgoal/diffusion_policy_mod.json
python robomimic/scripts/train.py --config robomimic/exps/temporaldp/temporal/liftqa/tc_diffusion_policy_mod.json
```

For a checkpoint-free integration test of a Diffusion Policy config, run one
gradient update and one native rollout with a short DDIM schedule:

```bash
python robomimic/scripts/smoke_test_temporal_diffusion.py \
  --config robomimic/exps/temporaldp/temporal/waitatgoal/tc_diffusion_policy_mod.json
```
