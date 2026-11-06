---
license: cc-by-4.0
task_categories:
- robotics
language:
- en
pretty_name: TC-DP
size_categories:
- 10K<n<100K
---

# Temporally Conditioned Diffusion Policy (TC-DP) Datasets

This dataset collection contains expert demonstration data for training diffusion policies with temporal conditioning on robotic manipulation and navigation tasks.

[![Paper preview](docs/Architecture.png)](docs/Architecture.pdf)

## Dataset Details

### Dataset Description

This dataset includes teleoperated expert demonstrations for two distinct robotic task environments:

1. **Warehouse Pickup and Delivery (WaitAtGoal)**: 2D navigation task in pygame requiring temporal reasoning and waiting behavior at goal locations. 
2. **Robotic Visual Quality Inspection (LiftQA)**: Robotic manipulation task based on Robosuite requiring pick-and-place operations with visual quality assessment.  

The datasets are designed to support research in temporal conditioning for robot learning, specifically for tasks involving idle states, waiting periods, and long-horizon temporal reasoning.

- **Curated by:** [withheld for blind review]
- **Format:** Zarr compressed arrays
- **License:** Creative Commons Attribution 4.0

### Dataset Sources

- **Repository:** [GitHub repository link]
- **Paper:** Temporal Conditioning for Diffusion-Based Robotic Imitation Learning


## Dataset Structure

All demo and inference datasets are stored in Zarr (version 2) format, checkpoints stored as torch `.pth` files. The data has the following structure:

```
data/
├── checkpoints/                 # checkpoints trained on human demonstrations
│   ├── 20260114_193124/
│   │   └── final.pth
│   ├── 20260115_024401/
│   │   └── final.pth
│   ├── 20260115_024411/
│   │   └── final.pth
│   └── ...
│
├── demonstration/               # human demonstration datasets (200 episodes)
│   ├── <demo_dataset_1>.zarr/
│   │   ├── .zattrs              # Dataset info
│   │   ├── .zgroup
│   │   ├── data/                # Episode data arrays
│   │   └── meta/                # Episode metadata
│   ├── <demo_dataset_2>.zarr/
│   │   ├── .zattrs
│   │   ├── .zgroup
│   │   ├── data/
│   │   └── meta/
│   └── ...
│
└── inference/                   # rollouts of trained policies (200 episodes)
    ├── <env>_<host_hash>_ckpt<timestamp>_<method>.zarr/
    │   ├── .zattrs
    │   ├── .zgroup
    │   ├── data/
    │   └── meta/
    ├── <env>_<host_hash>_ckpt<timestamp>_<method>.zarr/
    │   ├── .zattrs
    │   ├── .zgroup
    │   ├── data/
    │   └── meta/
    └── ...
```

## Uses

### Direct Use

This dataset is intended for:

- Training diffusion-based policies for robotic control
- Research on temporal encoding mechanisms (Idleness Encoding, Saturating Progress, Sinusoidal Progress)
- Benchmarking imitation learning algorithms on tasks requiring temporal reasoning
- Studying waiting behavior and idle state detection in robotic systems

### Out-of-Scope Use

- Real-world deployment without sim-to-real transfer validation
- Tasks significantly different from navigation or pick-and-place
- Applications requiring safety-critical guarantees without additional validation


## Dataset Creation

### Curation Rationale

These datasets were created to:
1. Support research on temporal awareness in robotic policies
2. Provide benchmark tasks for evaluating temporal encoding methods
3. Enable study of idle state detection and waiting behavior
4. Facilitate diffusion policy research on long-horizon tasks

### Data Collection and Processing

**Warehouse Pickup and Delivery (WaitAtGoal):**
- Human operators control a 2D agent using mouse
- Episodes include navigation to goal locations with required waiting periods
- Seeds are incremented for reproducibility

**Robotic Visual Quality Inspection (LiftQA):**
- Human operators use SpaceMouse 3D controllers
- Motion constrained to horizontal plane for task simplification

#### Who are the source data producers?
[withheld for blind review]


#### Personal and Sensitive Information
No personal and sensitive data is contained in the dataset.


## Bias, Risks, and Limitations

**Limitations:**
- Limited to simulation environments (MuJoCo and pygame based)
- Expert demonstrations may not cover all edge cases
- Fixed control frequency (10 Hz) may not suit all applications
- Constrained motion for LiftQA (horizontal plane only)

**Biases:**
- Operator-specific behaviors in teleoperation
- Task design may favor certain waiting strategies
- Visual observations limited to single camera viewpoint

<!--### Recommendations
    - 
-->

## Citation 

**BibTeX:**
[withheld for blind review]

## Dataset Card Contact
For questions or issues, please refer to the repository issue tracker.