#!/usr/bin/env python3
"""Add scalar agent velocity reconstructed from 2D pose trajectories to a Zarr dataset."""

from __future__ import annotations

import argparse
from pathlib import Path
import shutil

import numpy as np
import zarr


def calculate_velocity_magnitudes(
    agent_poses: np.ndarray,
    episode_ends: np.ndarray,
    control_hz: float,
) -> np.ndarray:
    """Return ``||(p_t - p_{t-1}) * control_hz||_2`` for every timestep.

    The first observation of every episode has no prior pose, so its velocity
    is zero.  Differencing is never allowed to cross an episode boundary.
    """
    poses = np.asarray(agent_poses, dtype=np.float32)
    ends = np.asarray(episode_ends, dtype=np.int64)
    if poses.ndim != 2 or poses.shape[1] != 2:
        raise ValueError(f"Expected agent_pose with shape (N, 2), got {poses.shape}.")
    if control_hz <= 0:
        raise ValueError(f"control_hz must be positive, got {control_hz}.")
    if len(ends) == 0 or ends[-1] != len(poses) or np.any(np.diff(ends) <= 0):
        raise ValueError("meta/episode_ends must be strictly increasing and end at N.")

    velocity = np.zeros(len(poses), dtype=np.float32)
    episode_start = 0
    for episode_end in ends:
        if episode_end - episode_start > 1:
            displacement = poses[episode_start + 1:episode_end] - poses[episode_start:episode_end - 1]
            velocity[episode_start + 1:episode_end] = (
                np.linalg.norm(displacement * control_hz, axis=1).astype(np.float32)
            )
        episode_start = int(episode_end)
    return velocity


def add_reconstructed_velocity(
    source_path: Path,
    destination_path: Path,
    overwrite: bool = False,
) -> None:
    """Copy a dataset, then add a reconstructed scalar ``agent_velocity`` array."""
    if not source_path.is_dir():
        raise FileNotFoundError(f"Source dataset does not exist: {source_path}")
    if destination_path.exists():
        if not overwrite:
            raise FileExistsError(
                f"Destination already exists: {destination_path}. Use --overwrite to replace it."
            )
        shutil.rmtree(destination_path)

    shutil.copytree(source_path, destination_path)
    root = zarr.open(str(destination_path), mode="a")
    try:
        poses = root["data"]["agent_pose"][:]
        episode_ends = root["meta"]["episode_ends"][:]
        control_hz = float(root.attrs["control_hz"])
    except KeyError as error:
        raise KeyError(
            f"{source_path} must contain data/agent_pose, meta/episode_ends, and root control_hz."
        ) from error

    velocity = calculate_velocity_magnitudes(poses, episode_ends, control_hz)
    pose_array = root["data"]["agent_pose"]
    root["data"].array(
        name="agent_velocity",
        data=velocity,
        chunks=(pose_array.chunks[0],),
        compressor=pose_array.compressor,
        overwrite=True,
    )
    root.attrs["agent_velocity_source"] = "euclidean_pose_difference_times_control_hz"
    print(
        f"Wrote {destination_path}/data/agent_velocity "
        f"({len(velocity)} scalar velocities at {control_hz:g} Hz)."
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="Input .zarr dataset")
    parser.add_argument("destination", type=Path, help="Output .zarr dataset")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing destination dataset.",
    )
    args = parser.parse_args()
    add_reconstructed_velocity(args.source, args.destination, args.overwrite)


if __name__ == "__main__":
    main()
