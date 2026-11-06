---
language:
- en
tags:
- robotics
- imitation-learning
- diffusion-policy
---
# TC-DP robomimic paper reproduction data

Selected checkpoints, training configurations and evaluation rollouts for Can,
Lift and Square in the main TC-DP paper comparison. The current selection has
36 checkpoints and 36 HDF5 rollouts: BC, BC-RNN, DP and TC-DP, with three training
seeds per method and task. Evaluation sidecars record the checkpoint and arguments.

Can, Lift and Square demonstration datasets are not duplicated here. Obtain the
PH demonstrations from robomimic upstream and prepare `image_v15.hdf5` as described
in the matching code release's `README_TC-DP.md`.

From the matching robomimic code release, run:

```bash
python dataset_upload.py --download
python -m robomimic.scripts.analysis_temporal \
    --task_names '["can", "lift", "square"]' --dataset_split ph \
    --rollout_sampling_mode all --rollout_selection_mode combine
```

Checkpoints retain the exact selected training snapshots. HDF5 rollouts contain
actions, states and evaluation metadata; they already contain no image arrays.
The matching code's `paper_upload_files.json` selects only the current comparison;
the upload does not delete older hosted experiments. Use a clean rollout directory
with just these three selected files per method/task when combining results.

WaitAtGoal, LiftQA, LiftQA-Real and PTP data are hosted at
https://huggingface.co/datasets/erdrsvhr9/temporally_conditioned_diffusion_policy.

## Citation

Temporary citation for this repository during anonymous review. Replace this
entry with the final publication details when available.

```bibtex
@misc{tcdp2026_robomimic,
  title={Mitigating Temporal Failures in Diffusion-Based Robotic Imitation Learning},
  author={{Anonymous Authors}},
  year={2026},
  howpublished={Hugging Face dataset repository},
  url={https://huggingface.co/datasets/erdrsvhr9/temporally_conditioned_diffusion_policy_robomimic},
  note={Anonymous review version; temporary citation}
}
```

If you use the robomimic artifacts, also cite the original robomimic work:

```bibtex
@inproceedings{robomimic2021,
  title={What Matters in Learning from Offline Human Demonstrations for Robot Manipulation},
  author={Mandlekar, Ajay and Xu, Danfei and Wong, Josiah and Nasiriany, Soroush and Wang, Chen and Kulkarni, Rohun and Fei-Fei, Li and Savarese, Silvio and Zhu, Yuke and Martin-Martin, Roberto},
  booktitle={Conference on Robot Learning},
  year={2021}
}
```

