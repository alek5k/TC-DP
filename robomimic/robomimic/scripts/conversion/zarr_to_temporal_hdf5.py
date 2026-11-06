"""Convert TemporalDiffusionPolicy Zarr demonstrations to robomimic HDF5.

The conversion preserves every source ``data/*`` array in each HDF5 episode's
``source_data`` group, as well as every ``meta/*`` array in ``source_meta``.
It additionally writes the Robomimic-compatible observations, actions,
rewards, and terminal flags required for training. Source images are float
CHW values in [0, 1]; robomimic's standard image path expects uint8 HWC
values, so they are restored to their original 8-bit pixels. No source Zarr
data is modified.
"""
import argparse
import json
from pathlib import Path

import h5py
import numpy as np
import zarr


TASK_SPECS = {
    "waitatgoal": {
        "env_name": "WaitAtGoal",
        "env_type": 4,  # robomimic native Temporal environment adapter
        "expected_action_dim": 2,
        "expected_pose_dim": 2,
    },
    "liftqa": {
        "env_name": "LiftQA",
        "env_type": 4,  # robomimic native Temporal environment adapter
        "expected_action_dim": 3,
        "expected_pose_dim": 3,
    },
}


def _read_source_images(images, start, stop):
    """Return source CHW images as uint8 HWC without lossy resize/crop."""
    batch = np.asarray(images[start:stop])
    if batch.ndim != 4 or batch.shape[1] != 3:
        raise ValueError(f"Expected source images with shape (T, 3, H, W), got {batch.shape}")
    if batch.dtype == np.uint8:
        rgb = batch
    else:
        if not np.isfinite(batch).all():
            raise ValueError("Source images contain NaN or infinity")
        high = float(batch.max()) if batch.size else 1.0
        if high <= 1.0 + 1e-6:
            rgb = np.rint(np.clip(batch, 0.0, 1.0) * 255.0).astype(np.uint8)
        elif high <= 255.0 + 1e-6:
            rgb = np.rint(np.clip(batch, 0.0, 255.0)).astype(np.uint8)
        else:
            raise ValueError(f"Unsupported source image range; maximum is {high}")
    return np.moveaxis(rgb, 1, -1)


def _terminal_dones(source_dones, length):
    """Create transition-aligned done flags and always mark the final action."""
    dones = np.zeros(length, dtype=np.bool_)
    if source_dones is not None:
        raw = np.asarray(source_dones, dtype=np.bool_).reshape(-1)
        if raw.shape[0] != length:
            raise ValueError(f"Expected {length} done flags, found {raw.shape[0]}")
        dones[:] = raw
    dones[-1] = True
    return dones


def _write_array(group, name, values, compression):
    group.create_dataset(name, data=values, compression=compression, shuffle=True)


def _copy_source_meta(source_root, output, compression):
    """Preserve every source meta array at the HDF5 root."""
    source_meta = output.create_group("source_meta")
    for key in source_root["meta"].keys():
        _write_array(source_meta, key, np.asarray(source_root["meta"][key][:]), compression)


def _environment_metadata(source_root, task, images):
    """Metadata consumed by :class:`robomimic.envs.env_temporal.EnvTemporal`."""
    spec = TASK_SPECS[task]
    return {
        "env_name": spec["env_name"],
        "type": spec["env_type"],
        "env_version": "temporal-env-v1",
        "env_kwargs": {
            "render_size": int(source_root.attrs.get("render_size", images.shape[-1])),
            "control_hz": int(source_root.attrs.get("control_hz", 10)),
            "constrain_motion": bool(source_root.attrs.get("constrain_motion", True)),
            "max_episode_length": int(source_root.attrs.get("max_episode_length", 1000)),
            "seed": int(source_root.attrs.get("seed", 0)),
        },
    }


def _split_demo_names(demo_names, val_ratio, split_seed):
    """Return the LDP-compatible deterministic train / validation split."""
    n_val = (
        min(max(1, round(len(demo_names) * val_ratio)), len(demo_names) - 1)
        if val_ratio > 0 else 0
    )
    rng = np.random.default_rng(split_seed)
    val_indices = set(rng.choice(len(demo_names), size=n_val, replace=False).tolist()) if n_val else set()
    train_names = [name for index, name in enumerate(demo_names) if index not in val_indices]
    val_names = [name for index, name in enumerate(demo_names) if index in val_indices]
    return train_names, val_names


def verify(source_path, output_path, task, val_ratio=0.02, split_seed=42):
    """Verify HDF5 schema and sampled values against a source temporal Zarr store.

    The first, middle, and final demonstrations are checked for exact action
    and pose values, terminal flags, and a pixel-for-pixel image round-trip.
    """
    source_path = Path(source_path).expanduser().resolve()
    output_path = Path(output_path).expanduser().resolve()
    if task not in TASK_SPECS:
        raise ValueError(f"Unsupported task {task!r}; choose from {sorted(TASK_SPECS)}")
    if not output_path.is_file():
        raise FileNotFoundError(f"HDF5 output does not exist: {output_path}")

    source = zarr.open_group(str(source_path), mode="r")
    episode_ends = np.asarray(source["meta/episode_ends"][:], dtype=np.int64)
    source_actions = source["data/action"]
    source_poses = source["data/agent_pose"]
    source_images = source["data/full_image"]
    source_velocities = source["data/agent_velocity"]
    expected_env_args = _environment_metadata(source, task, source_images)
    with h5py.File(output_path, "r") as output:
        actual_env_args = json.loads(output["data"].attrs["env_args"])
        if actual_env_args != expected_env_args:
            raise AssertionError("HDF5 environment metadata does not select the native temporal adapter")
        demos = sorted(output["data"].keys(), key=lambda name: int(name.split("_")[1]))
        if len(demos) != len(episode_ends):
            raise AssertionError(f"Expected {len(episode_ends)} demos, found {len(demos)}")
        transitions = sum(output["data"][demo].attrs["num_samples"] for demo in demos)
        if transitions != int(episode_ends[-1]):
            raise AssertionError(f"Expected {episode_ends[-1]} transitions, found {transitions}")

        expected_train, expected_valid = _split_demo_names(demos, val_ratio, split_seed)
        train_names = [name.decode("utf-8") for name in output["mask/train"][:]]
        valid_names = [name.decode("utf-8") for name in output["mask/valid"][:]]
        if train_names != expected_train or valid_names != expected_valid:
            raise AssertionError("HDF5 train / validation split does not match the requested LDP split")

        for demo_name in (demos[0], demos[len(demos) // 2], demos[-1]):
            demo = output["data"][demo_name]
            start, stop = demo.attrs["source_start"], demo.attrs["source_end"]
            if not np.array_equal(demo["actions"][:], source_actions[start:stop].astype(np.float32)):
                raise AssertionError(f"Action mismatch in {demo_name}")
            if not np.array_equal(demo["obs/agent_pose"][:], source_poses[start:stop].astype(np.float32)):
                raise AssertionError(f"Agent-pose mismatch in {demo_name}")
            expected_velocity = np.asarray(source_velocities[start:stop], dtype=np.float32).reshape(-1, 1)
            if not np.array_equal(demo["obs/agent_velocity"][:], expected_velocity):
                raise AssertionError(f"Agent-velocity mismatch in {demo_name}")
            expected_images = _read_source_images(source_images, start, stop)
            if not np.array_equal(demo["obs/image"][:], expected_images):
                raise AssertionError(f"Image round-trip mismatch in {demo_name}")
            if not demo["dones"][-1]:
                raise AssertionError(f"Final action is not terminal in {demo_name}")
            for key in source["data"].keys():
                if not np.array_equal(demo["source_data"][key][:], source["data"][key][start:stop]):
                    raise AssertionError(f"Source-data mismatch for {key!r} in {demo_name}")

        for key in source["meta"].keys():
            if not np.array_equal(output["source_meta"][key][:], source["meta"][key][:]):
                raise AssertionError(f"Source-meta mismatch for {key!r}")

    return {
        "episodes": len(episode_ends),
        "transitions": int(episode_ends[-1]),
        "train_episodes": len(expected_train),
        "valid_episodes": len(expected_valid),
    }


def convert(source_path, output_path, task, val_ratio=0.02, split_seed=42, image_chunk=32, compression="lzf"):
    """Convert one complete temporal Zarr replay buffer into robomimic HDF5."""
    if task not in TASK_SPECS:
        raise ValueError(f"Unsupported task {task!r}; choose from {sorted(TASK_SPECS)}")
    if not 0.0 <= val_ratio < 1.0:
        raise ValueError("val_ratio must be in [0, 1)")
    if image_chunk < 1:
        raise ValueError("image_chunk must be positive")

    source_path = Path(source_path).expanduser().resolve()
    output_path = Path(output_path).expanduser().resolve()
    if not source_path.is_dir():
        raise FileNotFoundError(f"Zarr source does not exist: {source_path}")
    if output_path.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    root = zarr.open_group(str(source_path), mode="r")
    for key in ("action", "agent_pose", "agent_velocity", "full_image"):
        if key not in root["data"]:
            raise KeyError(f"Missing data/{key} in {source_path}")
    episode_ends = np.asarray(root["meta/episode_ends"][:], dtype=np.int64)
    if episode_ends.ndim != 1 or episode_ends.size == 0 or np.any(np.diff(episode_ends) <= 0):
        raise ValueError("meta/episode_ends must be a non-empty strictly increasing vector")

    spec = TASK_SPECS[task]
    actions = root["data/action"]
    poses = root["data/agent_pose"]
    velocities = root["data/agent_velocity"]
    images = root["data/full_image"]
    if actions.shape[1:] != (spec["expected_action_dim"],):
        raise ValueError(f"Expected {spec['expected_action_dim']}-D actions, got {actions.shape[1:]}")
    if poses.shape[1:] != (spec["expected_pose_dim"],):
        raise ValueError(f"Expected {spec['expected_pose_dim']}-D poses, got {poses.shape[1:]}")
    if velocities.shape[0] != actions.shape[0]:
        raise ValueError("agent_velocity and actions do not agree on transition count")
    if episode_ends[-1] != actions.shape[0] or poses.shape[0] != actions.shape[0] or images.shape[0] != actions.shape[0]:
        raise ValueError("Source arrays and episode_ends do not agree on transition count")

    source_rewards = root["data/reward"] if "reward" in root["data"] else None
    source_dones = root["data/done"] if "done" in root["data"] else None
    source_data_keys = list(root["data"].keys())
    for key in source_data_keys:
        if root["data"][key].shape[0] != actions.shape[0]:
            raise ValueError(
                f"data/{key} has {root['data'][key].shape[0]} transitions, expected {actions.shape[0]}"
            )
    env_args = _environment_metadata(root, task, images)

    try:
        with h5py.File(output_path, "x", libver="latest") as output:
            data_group = output.create_group("data")
            data_group.attrs["env_args"] = json.dumps(env_args, sort_keys=True)
            data_group.attrs["temporal_source_zarr"] = str(source_path)
            data_group.attrs["temporal_task"] = task
            data_group.attrs["temporal_source_attrs"] = json.dumps(dict(root.attrs), sort_keys=True, default=str)
            _copy_source_meta(root, output, compression)

            start = 0
            demo_names = []
            for episode_index, stop in enumerate(episode_ends):
                stop = int(stop)
                length = stop - start
                demo_name = f"demo_{episode_index}"
                demo_names.append(demo_name)
                demo = data_group.create_group(demo_name)
                demo.attrs["num_samples"] = length
                demo.attrs["source_start"] = start
                demo.attrs["source_end"] = stop
                obs = demo.create_group("obs")
                source_data = demo.create_group("source_data")

                for key in source_data_keys:
                    values = np.asarray(root["data"][key][start:stop])
                    _write_array(source_data, key, values, compression)
                    if key not in {"action", "reward", "done", "full_image", "agent_pose", "agent_velocity"}:
                        _write_array(obs, key, values, compression)

                image_shape = (length, images.shape[2], images.shape[3], images.shape[1])
                image_chunks = (min(image_chunk, length),) + image_shape[1:]
                destination = obs.create_dataset(
                    "image", shape=image_shape, dtype=np.uint8, chunks=image_chunks,
                    compression=compression, shuffle=True,
                )
                for batch_start in range(start, stop, image_chunk):
                    batch_stop = min(batch_start + image_chunk, stop)
                    destination[batch_start - start:batch_stop - start] = _read_source_images(images, batch_start, batch_stop)

                _write_array(obs, "agent_pose", np.asarray(poses[start:stop], dtype=np.float32), compression)
                _write_array(
                    obs,
                    "agent_velocity",
                    np.asarray(velocities[start:stop], dtype=np.float32).reshape(-1, 1),
                    compression,
                )
                _write_array(demo, "actions", np.asarray(actions[start:stop], dtype=np.float32), compression)
                rewards = np.zeros(length, dtype=np.float32) if source_rewards is None else np.asarray(source_rewards[start:stop], dtype=np.float32)
                _write_array(demo, "rewards", rewards, compression)
                dones = _terminal_dones(None if source_dones is None else source_dones[start:stop], length)
                _write_array(demo, "dones", dones, compression)
                start = stop

            train_names, val_names = _split_demo_names(demo_names, val_ratio, split_seed)
            mask = output.create_group("mask")
            mask.create_dataset("train", data=np.asarray(train_names, dtype="S"))
            mask.create_dataset("valid", data=np.asarray(val_names, dtype="S"))
    except Exception:
        if output_path.exists():
            output_path.unlink()
        raise

    return {"episodes": len(episode_ends), "transitions": int(episode_ends[-1]), "train_episodes": len(train_names), "valid_episodes": len(val_names)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, help="Input TemporalDiffusionPolicy .zarr directory")
    parser.add_argument("--output", required=True, help="New robomimic .hdf5 file")
    parser.add_argument("--task", required=True, choices=sorted(TASK_SPECS))
    parser.add_argument("--verify", action="store_true", help="Verify an existing output without converting")
    parser.add_argument("--val-ratio", type=float, default=0.02)
    parser.add_argument("--split-seed", type=int, default=42, help="Matches LDP TemporalZarrImageDataset")
    parser.add_argument("--image-chunk", type=int, default=32, help="Images processed per write")
    parser.add_argument("--compression", choices=("lzf", "gzip", "none"), default="lzf")
    args = parser.parse_args()
    compression = None if args.compression == "none" else args.compression
    if not args.verify:
        convert(
            source_path=args.source, output_path=args.output, task=args.task,
            val_ratio=args.val_ratio, split_seed=args.split_seed,
            image_chunk=args.image_chunk, compression=compression,
        )
    summary = verify(
        source_path=args.source, output_path=args.output, task=args.task,
        val_ratio=args.val_ratio, split_seed=args.split_seed,
    )
    action = "Verified" if args.verify else "Converted and verified"
    print(f"{action} {{episodes}} episodes / {{transitions}} transitions "
          "({train_episodes} train, {valid_episodes} valid)".format(**summary))


if __name__ == "__main__":
    main()
