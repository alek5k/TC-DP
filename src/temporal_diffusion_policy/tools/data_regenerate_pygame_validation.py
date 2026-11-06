from dataclasses import dataclass
from datetime import datetime
import os
from pathlib import Path
import sys
import draccus
import numpy as np
from temporal_diffusion_policy.replay_buffer import ReplayBuffer

@dataclass
class Config:
    source: str = 'data/demonstration/wait_at_goal_dataset_v2.zarr'
    output: str = 'data/demonstration/wait_at_goal_dataset_v3.zarr'
    
@draccus.wrap()
def main(cfg: Config):

    new_dataset_valid = True

    source_buffer = ReplayBuffer.create_from_path(cfg.source, 'r')
    source_num_episodes = source_buffer.n_episodes

    new_buffer = ReplayBuffer.create_from_path(cfg.output, mode='r')
    new_num_episodes = new_buffer.n_episodes

    assert source_num_episodes == new_num_episodes, f"Number of episodes do not match: source {source_num_episodes}, new {new_num_episodes}"

    for ep_idx in range(source_num_episodes):
        source_episode = source_buffer.get_episode(ep_idx)
        new_episode = new_buffer.get_episode(ep_idx)

        for key in source_episode.keys():
            source_data = source_episode[key]
            if key not in new_episode:
                # print(f"Key '{key}' not found in new episode {ep_idx}")
                continue
            new_data = new_episode[key]

            if source_data.shape != new_data.shape:
                print(f"Shape mismatch in episode {ep_idx}, key '{key}': source shape {source_data.shape}, new shape {new_data.shape}")
                new_dataset_valid = False
                continue

            if not np.allclose(source_data, new_data):
                print(f"Mismatch found in episode {ep_idx}, key '{key}'")
                new_dataset_valid = False
                # diff_indices = np.where(source_data != new_data)
                # print(f"Differences at indices: {diff_indices}")

    if not new_dataset_valid:
        print("New dataset validation failed.")
        sys.exit(1)
    else:
        sys.exit(0)

if __name__ == "__main__":
    main()
