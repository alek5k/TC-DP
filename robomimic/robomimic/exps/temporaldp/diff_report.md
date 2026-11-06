
can you tell me how temporaldp/can/mh/image/hbc.json is generated. does it reflect a combination of low_dim/hbc.json and image/bc.json

Let me show you by comparing the three key files:   

Read [](file://~/Repos/robomimic/robomimic/exps/temporaldp/can/mh/image/hbc.json#1-1), lines 1 to 80
Read [](file://~/Repos/robomimic/robomimic/exps/paper/core/can/mh/low_dim/hbc.json#1-1), lines 1 to 80
Read [](file://~/Repos/robomimic/robomimic/exps/paper/core/can/mh/image/bc.json#30-30), lines 30 to 60

Let me show you the algo sections which are most revealing:   

Read [](file://~/Repos/robomimic/robomimic/exps/paper/core/can/mh/image/bc.json#70-70), lines 70 to 150
Read [](file://~/Repos/robomimic/robomimic/exps/paper/core/can/mh/low_dim/hbc.json#60-60), lines 60 to 250
Read [](file://~/Repos/robomimic/robomimic/exps/temporaldp/can/mh/image/hbc.json#180-180), lines 180 to 280

Perfect! Yes, it's exactly a **hybrid combination**. Let me show you the breakdown:

## temporaldp/can/mh/image/hbc.json = **low_dim/hbc.json (algo structure) + image/bc.json (training params & encoders)**

### 1. **ALGO STRUCTURE** ← Taken from **low_dim/hbc.json**
```
✓ Hierarchical planner-actor architecture
✓ Planner VAE with latent_dim: 16, kl_weight: 0.0005
✓ Actor RNN enabled with hidden_dim: 400 (low_dim value)
✓ hdf5_load_next_obs: true (required for HBC subgoals)
✓ mode: "separate", subgoal_horizon: 10
```

### 2. **TRAINING PARAMETERS** ← Taken from **image/bc.json**
```
✓ batch_size: 16         (vs 100 in low_dim)
✓ num_epochs: 600        (vs 2000 in low_dim)
✓ num_data_workers: 2    (vs 0 in low_dim)
✓ hdf5_cache_mode: "low_dim"
✓ epoch_every_n_steps: 500        (vs 100 in low_dim)
✓ validation_epoch_every_n_steps: 50  (vs 10 in low_dim)
```

### 3. **OBSERVATION ENCODERS** ← Taken from **image/bc.json**
```
✓ Visual encoder: ResNet18Conv (backbone)
✓ Pooling: SpatialSoftmax (32 keypoints)
✓ Augmentation: CropRandomizer (76×76)
✓ Modalities: Both RGB images + low_dim (eef_pos, eef_quat, gripper_qpos)
✓ Applied to both planner & actor in HBC structure
```

### 4. **ONE KEY MODIFICATION**
The actor RNN `hidden_dim` was upgraded from **400 → 1000** to better process image features:

| Parameter | low_dim/hbc | temporaldp/image/hbc |
|-----------|------------|---------------------|
| hidden_dim | 400 | 1000 |

This makes sense because image encoders produce richer features than low_dim inputs, so the RNN needs higher capacity.

### Summary
```
temporaldp/hbc = {
  algo: low_dim/hbc.json      # hierarchical structure, planner VAE, subgoal logic
  train: image/bc.json         # image-optimized batch/epoch settings
  encoder: image/bc.json       # visual backbones + augmentation
  tweak: hidden_dim 400→1000   # boost for richer image features
}
```


ok do the same for diffusion_policy.json
Read [](file://~/Repos/robomimic/robomimic/exps/temporaldp/can/mh/image/diffusion_policy.json#1-1), lines 1 to 150

Read [](file://~/Repos/robomimic/robomimic/exps/templates/diffusion_policy.json#1-1), lines 1 to 150

Perfect! Let me compare:  

Read [](file://~/Repos/robomimic/robomimic/exps/templates/diffusion_policy.json#100-100), lines 100 to 184

Read [](file://~/Repos/robomimic/robomimic/exps/temporaldp/can/mh/image/diffusion_policy.json#150-150), lines 150 to 206

## temporaldp/can/mh/image/diffusion_policy.json = **templates/diffusion_policy.json (algo) + image/bc.json (training & encoders)**

### 1. **ALGO STRUCTURE** ← **templates/diffusion_policy.json** (100% IDENTICAL)
```
✓ UNet architecture with diffusion_step_embed_dim: 256
✓ Down dims: [256, 512, 1024]
✓ Kernel size: 5, n_groups: 8
✓ DDPM: num_train_timesteps: 100, beta_schedule: "squaredcos_cap_v2"
✓ EMA enabled with power: 0.75
✓ Horizons: observation_horizon: 2, action_horizon: 8, prediction_horizon: 16
✓ Optimizer: adamw with cosine scheduler, warmup_steps: 500
✓ seq_length: 16, frame_stack: 2 (diffusion-specific sequences)
```

### 2. **TRAINING PARAMETERS** ← **image/bc.json**

| Parameter | Template | temporaldp/image |
|-----------|----------|-----------------|
| batch_size | 100 | **16** |
| num_epochs | 2000 | **600** |
| num_data_workers | 0 | **2** |
| hdf5_cache_mode | "all" | **"low_dim"** |
| epoch_every_n_steps | 100 | **500** |
| validation_epoch_every_n_steps | 10 | **50** |
| validate | false | **true** |
| rollout.rate | 50 | **20** |
| rollout.horizon | 400 (generic) | **500** (env-specific) |

### 3. **OBSERVATION ENCODERS** ← **image/bc.json**
**Before (Template - low_dim only):**
```json
"rgb": {
    "core_class": "VisualCore",
    "core_kwargs": {},           // ← Empty!
    "obs_randomizer_class": null
}
```

**After (temporaldp - visual encoders):**
```json
"rgb": {
    "core_class": "VisualCore",
    "core_kwargs": {
        "feature_dimension": 64,
        "backbone_class": "ResNet18Conv",
        "backbone_kwargs": {"pretrained": false, "input_coord_conv": false},
        "pool_class": "SpatialSoftmax",
        "pool_kwargs": {"num_kp": 32, ...}
    },
    "obs_randomizer_class": "CropRandomizer",
    "obs_randomizer_kwargs": {"crop_height": 76, "crop_width": 76, ...}
}
```

### 4. **OBSERVATION MODALITIES** ← **image/bc.json**
**Before:** `"object"` in low_dim only
```
low_dim: ["robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos", "object"]
rgb: []
```

**After:** RGB images added
```
low_dim: ["robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos"]
rgb: ["agentview_image", "robot0_eye_in_hand_image"]
```

### Summary
```
temporaldp/diffusion_policy = {
  algo: templates/diffusion_policy.json    # UNet, DDPM, horizons (unchanged)
  train: image/bc.json                     # batch_size: 16, epochs: 600, workers: 2
  encoder: image/bc.json                   # ResNet18Conv + SpatialSoftmax + Crop
  modalities: image/bc.json                # Add RGB images, remove "object"
  seq_params: templates/diffusion_policy   # Keep seq_length: 16, frame_stack: 2
}
```

**Key insight:** Diffusion Policy has **no paper baseline**, so the template provides the diffusion-specific algorithm structure (UNet/DDPM/horizons), then we retrofit it with image-optimized training parameters and visual encoders from the BC image config.