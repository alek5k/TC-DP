"""Apply a fixed local-frame 6-DoF transform to pose fields in a dataset.

The transform is right-composed with every pose: ``T_out = T_in @ T_delta``.
Consequently, positions use ``p_out = p_in + R_in @ t_delta`` and rotvec
orientations use ``R_out = R_in @ R_delta``. Both ``(T, 4, 4)`` homogeneous
matrices and ``(T, D)`` arrays containing ``[x, y, z, rx, ry, rz, ...]`` are
supported. Any trailing values, such as a gripper command, are preserved.
"""

from pathlib import Path
from typing import Sequence

import numpy as np
from scipy.spatial.transform import Rotation

from diffusion_policy_contrib.common.replay_buffer import ReplayBuffer



def apply_transform(
    poses: np.ndarray,
    translation: np.ndarray,
    rotation: Rotation,
) -> np.ndarray:
    """Right-compose a local transform with a batch of recorded poses."""
    poses = np.asarray(poses)
    if not np.issubdtype(poses.dtype, np.number):
        raise TypeError(f"Pose data must be numeric, got {poses.dtype}.")

    if poses.ndim == 3 and poses.shape[1:] == (4, 4):
        delta_transform = np.eye(4, dtype=np.float64)
        delta_transform[:3, :3] = rotation.as_matrix()
        delta_transform[:3, 3] = translation
        return np.matmul(poses.astype(np.float64), delta_transform).astype(
            poses.dtype, copy=False
        )

    if poses.ndim != 2 or poses.shape[1] < 6:
        raise ValueError(
            "Local-frame transforms require pose orientation. Expected shape "
            f"(T, 4, 4) or (T, D) with D >= 6, got {poses.shape}."
        )

    orientations = Rotation.from_rotvec(poses[:, 3:6].astype(np.float64))
    transformed = poses.copy()
    transformed[:, :3] = (
        poses[:, :3].astype(np.float64) + orientations.apply(translation)
    ).astype(poses.dtype, copy=False)
    transformed[:, 3:6] = (orientations * rotation).as_rotvec().astype(
        poses.dtype, copy=False
    )
    return transformed


def do_transform(
    source: Path,
    destination: Path,
    translation: Sequence[float],
    rotation_rotvec: Sequence[float],
    pose_keys: Sequence[str],
    overwrite: bool,
) -> None:
    source = source.resolve()
    destination = destination.resolve()
    if source == destination:
        raise ValueError("Source and destination dataset paths must differ.")
    if not source.exists():
        raise FileNotFoundError(f"Source dataset does not exist: {source}")
    if destination.exists() and not overwrite:
        raise FileExistsError(
            f"Destination already exists: {destination}. Use --overwrite to replace it."
        )

    translation = np.asarray(translation, dtype=np.float64)
    rotation_rotvec = np.asarray(rotation_rotvec, dtype=np.float64)
    if translation.shape != (3,):
        raise ValueError(f"translation must contain three values, got {translation.shape}.")
    if rotation_rotvec.shape != (3,):
        raise ValueError(
            f"rotation-rotvec must contain three values, got {rotation_rotvec.shape}."
        )
    if not np.isfinite(translation).all() or not np.isfinite(rotation_rotvec).all():
        raise ValueError("Transform translation and rotation must be finite.")
    if not pose_keys:
        raise ValueError("At least one pose key must be provided.")

    source_buffer = ReplayBuffer.create_from_path(source, mode="r")
    present_keys = set(source_buffer.data.keys())
    keys_to_transform = tuple(key for key in pose_keys if key in present_keys)
    missing_keys = tuple(key for key in pose_keys if key not in present_keys)
    if not keys_to_transform:
        raise KeyError(
            f"None of the requested pose keys exist in {source}: {', '.join(pose_keys)}"
        )
    if missing_keys:
        print(f"Skipping absent pose keys: {', '.join(missing_keys)}")

    delta_rotation = Rotation.from_rotvec(rotation_rotvec)
    destination_buffer = ReplayBuffer.create_from_path(destination, mode="w")
    for key, value in source_buffer.meta.items():
        if key != "episode_ends":
            destination_buffer.meta.update({key: value})
    for key, value in source_buffer.root.attrs.items():
        destination_buffer.root.attrs[key] = value
    destination_buffer.root.attrs["applied_transform"] = {
        "translation": translation.tolist(),
        "rotation_rotvec": rotation_rotvec.tolist(),
        "pose_keys": list(keys_to_transform),
        "composition": "T_out = T_in @ T_delta",
    }

    for episode_id in range(source_buffer.n_episodes):
        episode = source_buffer.get_episode(episode_id, copy=True)
        for key in keys_to_transform:
            episode[key] = apply_transform(episode[key], translation, delta_rotation)
        destination_buffer.add_episode(episode, compressors="disk")
        print(f"Transformed episode {episode_id + 1}/{source_buffer.n_episodes}")

    print(f"Wrote transformed dataset to {destination}")


if __name__ == "__main__":
    
    do_transform(
        source=Path("data_new/demonstration/piano_vive_corr.zarr"),
        destination=Path("data_new/demonstration/piano_vive_corr_transformed.zarr"),
        translation=[0.0, 0.0, 0.044], 
        rotation_rotvec=[0.0, 0.0, 0.0],
        pose_keys=["vive_pose"],
        overwrite=True,
    )
