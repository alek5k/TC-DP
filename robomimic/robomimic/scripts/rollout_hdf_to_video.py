#!/usr/bin/env python3
"""Archive rollout image observations as a compressed FFmpeg video.

By default the script is non-destructive. Pass ``--delete-images`` only after
reviewing the output video: it rewrites the HDF5 without the exported image
datasets, then atomically replaces the original file so disk space is freed.
"""

from __future__ import annotations

import argparse
import itertools
import math
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
from typing import Iterator

import h5py
import numpy as np


def natural_demo_key(name: str) -> tuple[int, str]:
    suffix = name.removeprefix("demo_")
    return (int(suffix), name) if suffix.isdigit() else (10**12, name)


def image_keys_for_demo(demo: h5py.Group, observation_group: str) -> list[str]:
    if observation_group not in demo:
        return []
    observation = demo[observation_group]
    return sorted(
        name
        for name, dataset in observation.items()
        if isinstance(dataset, h5py.Dataset)
        and "image" in name.lower()
        and dataset.ndim >= 4
        and dataset.shape[-1] in {1, 3, 4}
    )


def frame_from_dataset(dataset: h5py.Dataset, index: int, frame_index: int) -> np.ndarray:
    frame = dataset[index]
    # Rollout observations may be [T, H, W, C] or [T, obs_horizon, H, W, C].
    if frame.ndim == 4:
        frame = frame[frame_index]
    if frame.ndim != 3 or frame.shape[-1] not in {1, 3, 4}:
        raise ValueError(f"Unsupported image shape for {dataset.name}: {dataset.shape}")
    if frame.dtype != np.uint8:
        frame = np.clip(frame, 0, 255).astype(np.uint8)
    if frame.shape[-1] == 1:
        frame = np.repeat(frame, 3, axis=-1)
    elif frame.shape[-1] == 4:
        frame = frame[..., :3]
    return np.ascontiguousarray(frame)


def tile_frames(frames: list[np.ndarray], columns: int) -> np.ndarray:
    if not frames:
        raise ValueError("No image frames to tile.")
    height = max(frame.shape[0] for frame in frames)
    width = max(frame.shape[1] for frame in frames)
    columns = min(columns, len(frames))
    rows = math.ceil(len(frames) / columns)
    canvas = np.zeros((rows * height, columns * width, 3), dtype=np.uint8)
    for index, frame in enumerate(frames):
        row, column = divmod(index, columns)
        y = row * height
        x = column * width
        canvas[y:y + frame.shape[0], x:x + frame.shape[1]] = frame
    # yuv420p requires even dimensions.
    padded_height = canvas.shape[0] + canvas.shape[0] % 2
    padded_width = canvas.shape[1] + canvas.shape[1] % 2
    if (padded_height, padded_width) != canvas.shape[:2]:
        padded = np.zeros((padded_height, padded_width, 3), dtype=np.uint8)
        padded[:canvas.shape[0], :canvas.shape[1]] = canvas
        return padded
    return canvas


def iter_video_frames(
    hdf: h5py.File,
    image_keys: list[str],
    observation_group: str,
    frame_index: int,
    columns: int,
    max_demos: int | None,
) -> Iterator[np.ndarray]:
    data = hdf.get("data")
    if not isinstance(data, h5py.Group):
        raise ValueError("Expected a top-level /data group.")
    demo_names = sorted((name for name in data if name.startswith("demo_")), key=natural_demo_key)
    if max_demos is not None:
        demo_names = demo_names[:max_demos]
    if not demo_names:
        raise ValueError("No demo_* groups found under /data.")

    for demo_name in demo_names:
        demo = data[demo_name]
        if observation_group not in demo:
            raise ValueError(f"{demo.name} has no /{observation_group} group.")
        observation = demo[observation_group]
        missing = [key for key in image_keys if key not in observation]
        if missing:
            raise ValueError(f"{demo.name}/{observation_group} is missing image key(s): {missing}")
        length = min(observation[key].shape[0] for key in image_keys)
        for index in range(length):
            yield tile_frames(
                [frame_from_dataset(observation[key], index, frame_index) for key in image_keys],
                columns,
            )


def image_dataset_paths(hdf: h5py.File, image_keys: list[str]) -> set[str]:
    return observation_dataset_paths(hdf, image_keys)


def observation_dataset_paths(hdf: h5py.File, dataset_keys: list[str]) -> set[str]:
    """Return /obs and /next_obs paths for the requested exact dataset keys."""
    paths: set[str] = set()
    data = hdf.get("data")
    if not isinstance(data, h5py.Group):
        return paths
    for demo_name, demo in data.items():
        if not isinstance(demo, h5py.Group) or not demo_name.startswith("demo_"):
            continue
        for observation_group in ("obs", "next_obs"):
            group = demo.get(observation_group)
            if not isinstance(group, h5py.Group):
                continue
            for dataset_key in dataset_keys:
                if dataset_key in group and isinstance(group[dataset_key], h5py.Dataset):
                    paths.add(group[dataset_key].name.lstrip("/"))
    return paths


def copy_without_datasets(source: h5py.Group, destination: h5py.Group, skipped_paths: set[str]) -> None:
    """Copy an HDF5 tree, excluding only explicit dataset paths."""
    for name, object_ in source.items():
        source_path = object_.name.lstrip("/")
        if source_path in skipped_paths:
            continue
        if isinstance(object_, h5py.Group):
            destination_group = destination.create_group(name)
            for attr_name, attr_value in object_.attrs.items():
                destination_group.attrs[attr_name] = attr_value
            copy_without_datasets(object_, destination_group, skipped_paths)
        else:
            source.copy(name, destination, name=name)


def remove_images_and_repack(hdf_path: Path, skipped_paths: set[str]) -> None:
    """Atomically replace an HDF5 file with a compact copy without images."""
    with tempfile.NamedTemporaryFile(
        prefix=f".{hdf_path.stem}.without_images.",
        suffix=hdf_path.suffix,
        dir=hdf_path.parent,
        delete=False,
    ) as temporary:
        temporary_path = Path(temporary.name)
    try:
        with h5py.File(hdf_path, "r") as source, h5py.File(temporary_path, "w") as destination:
            for attr_name, attr_value in source.attrs.items():
                destination.attrs[attr_name] = attr_value
            copy_without_datasets(source, destination, skipped_paths)
        os.replace(temporary_path, hdf_path)
    except BaseException:
        temporary_path.unlink(missing_ok=True)
        raise


def strip_lang_embeddings(hdf_path: Path) -> int:
    """Compactly remove only rollout language embeddings; return path count."""
    with h5py.File(hdf_path, "r") as hdf:
        paths = observation_dataset_paths(hdf, ["lang_emb"])
    if not paths:
        print(f"No lang_emb datasets in {hdf_path}; skipped.")
        return 0
    before_size = hdf_path.stat().st_size
    remove_images_and_repack(hdf_path, paths)
    after_size = hdf_path.stat().st_size
    print(
        f"Removed {len(paths)} lang_emb dataset(s) from {hdf_path}. "
        f"Freed {(before_size - after_size) / 2**20:.1f} MiB."
    )
    return len(paths)


def encode_video(
    hdf_path: Path,
    video_path: Path,
    image_keys: list[str],
    observation_group: str,
    frame_index: int,
    columns: int,
    fps: int,
    crf: int,
    max_demos: int | None,
    overwrite: bool,
) -> int:
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise FileNotFoundError("ffmpeg was not found on PATH.")
    with h5py.File(hdf_path, "r") as hdf:
        frames = iter_video_frames(hdf, image_keys, observation_group, frame_index, columns, max_demos)
        try:
            first_frame = next(frames)
        except StopIteration as exc:
            raise ValueError("The selected rollout contains no image frames.") from exc
        height, width = first_frame.shape[:2]
        ffmpeg_command = [
            ffmpeg, "-y" if overwrite else "-n",
            "-f", "rawvideo",
            "-pixel_format", "rgb24",
            "-video_size", f"{width}x{height}",
            "-framerate", str(fps),
            "-i", "-",
            "-an",
            "-c:v", "libx264",
            "-preset", "medium",
            "-crf", str(crf),
            "-pix_fmt", "yuv420p",
            "-movflags", "+faststart",
            str(video_path),
        ]
        print("FFmpeg command:")
        print("  " + " ".join(ffmpeg_command))
        process = subprocess.Popen(ffmpeg_command, stdin=subprocess.PIPE)
        assert process.stdin is not None
        frame_count = 0
        try:
            for frame in itertools.chain((first_frame,), frames):
                if frame.shape[:2] != (height, width):
                    raise ValueError("Image layout changed within the rollout; cannot encode a fixed-size video.")
                process.stdin.write(frame.tobytes())
                frame_count += 1
        except BaseException:
            process.stdin.close()
            process.wait()
            raise
        process.stdin.close()
        if process.wait() != 0:
            raise RuntimeError("ffmpeg failed; the HDF5 was not modified.")
    if not video_path.is_file() or video_path.stat().st_size == 0:
        raise RuntimeError("ffmpeg exited successfully but did not produce a non-empty video.")
    return frame_count


def archive_rollout(args: argparse.Namespace, hdf_path: Path, video_path: Path) -> None:
    if hdf_path.suffix.lower() not in {".hdf", ".hdf5"}:
        raise ValueError("Rollout must be an .hdf or .hdf5 file.")
    if video_path == hdf_path:
        raise ValueError("Output video cannot overwrite the rollout HDF5.")
    if video_path.exists() and not args.overwrite:
        raise FileExistsError(f"Output video already exists: {video_path} (pass --overwrite to replace it)")
    video_path.parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(hdf_path, "r") as hdf:
        data = hdf.get("data")
        if not isinstance(data, h5py.Group):
            raise ValueError("Expected a top-level /data group.")
        demo_names = sorted((name for name in data if name.startswith("demo_")), key=natural_demo_key)
        if not demo_names:
            raise ValueError("No demo_* groups found under /data.")
        image_keys = args.image_keys or image_keys_for_demo(data[demo_names[0]], args.observation_group)
        if not image_keys:
            raise ValueError(f"No image datasets found under {data[demo_names[0]].name}/{args.observation_group}.")
        delete_paths: set[str] = set()
        if args.delete_images:
            delete_paths.update(image_dataset_paths(hdf, image_keys))
        if args.delete_lang_emb:
            delete_paths.update(observation_dataset_paths(hdf, ["lang_emb"]))

    columns = args.columns or math.ceil(math.sqrt(len(image_keys)))
    print(f"\n{hdf_path}\nExporting image keys: {', '.join(image_keys)}")
    frame_count = encode_video(
        hdf_path, video_path, image_keys, args.observation_group,
        args.frame_index, columns, args.fps, args.crf, args.max_demos,
        args.overwrite,
    )
    print(f"Wrote {frame_count} frame(s) to {video_path} ({video_path.stat().st_size / 2**20:.1f} MiB).")

    if args.delete_images or args.delete_lang_emb:
        if args.max_demos is not None:
            raise ValueError("Refusing to delete datasets after --max-demos; export the full rollout first.")
        if not delete_paths:
            raise ValueError("No matching datasets were found to delete.")
        before_size = hdf_path.stat().st_size
        remove_images_and_repack(hdf_path, delete_paths)
        after_size = hdf_path.stat().st_size
        print(
            f"Deleted {len(delete_paths)} archived dataset(s) and repacked {hdf_path}. "
            f"Freed {(before_size - after_size) / 2**20:.1f} MiB."
        )


def input_rollouts(inputs: list[Path], recursive: bool) -> list[Path]:
    rollouts = []
    for input_path in inputs:
        path = input_path.expanduser().resolve()
        if path.is_file():
            rollouts.append(path)
        elif path.is_dir() and recursive:
            rollouts.extend(child for child in path.rglob("*") if child.suffix.lower() in {".hdf", ".hdf5"})
        elif path.is_dir():
            raise ValueError(f"{path} is a directory; pass --recursive to archive its rollout HDF5 files.")
        else:
            raise FileNotFoundError(f"Rollout path not found: {path}")
    return sorted(set(rollouts))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("rollout", nargs="+", type=Path, help="rollout .hdf5 file(s), or a directory with --recursive")
    parser.add_argument("--recursive", action="store_true", help="recursively archive every .hdf and .hdf5 below a directory")
    parser.add_argument("--output", type=Path, help="output MP4 path (single input only; defaults beside the rollout)")
    parser.add_argument("--image-keys", nargs="+", help="image observation keys to tile; defaults to every /obs/*image* key")
    parser.add_argument("--observation-group", default="obs", help="group used to render frames (default: obs)")
    parser.add_argument("--frame-index", type=int, default=-1, help="observation-horizon frame to display (default: -1)")
    parser.add_argument("--columns", type=int, help="mosaic columns; default is a near-square layout")
    parser.add_argument("--fps", type=int, default=20, help="output video frame rate")
    parser.add_argument("--crf", type=int, default=23, help="H.264 constant-rate-factor, lower is higher quality")
    parser.add_argument("--max-demos", type=int, help="encode only the first N demos; useful for verification")
    parser.add_argument("--overwrite", action="store_true", help="overwrite an existing output video")
    parser.add_argument(
        "--strip-lang-emb-only",
        action="store_true",
        help="compactly remove only lang_emb; do not encode video or modify other datasets",
    )
    parser.add_argument(
        "--delete-lang-emb",
        action="store_true",
        help="also delete /obs and /next_obs lang_emb datasets when compacting",
    )
    parser.add_argument(
        "--delete-images",
        action="store_true",
        help="after a verified video is written, compactly rewrite the HDF5 without these image datasets",
    )
    args = parser.parse_args()

    if args.fps <= 0 or args.crf < 0 or args.columns is not None and args.columns <= 0:
        raise ValueError("fps and columns must be positive; crf must be non-negative.")
    if args.strip_lang_emb_only and (args.output or args.delete_images or args.delete_lang_emb or args.max_demos):
        raise ValueError("--strip-lang-emb-only cannot be combined with video or dataset-selection options.")
    rollouts = input_rollouts(args.rollout, args.recursive)
    if not rollouts:
        raise ValueError("No rollout HDF5 files found.")
    if args.output is not None and len(rollouts) != 1:
        raise ValueError("--output can only be used with exactly one rollout file.")

    failures = []
    removed = 0
    for hdf_path in rollouts:
        try:
            if args.strip_lang_emb_only:
                removed += strip_lang_embeddings(hdf_path)
            else:
                video_path = (args.output or hdf_path.with_suffix(".mp4")).expanduser().resolve()
                archive_rollout(args, hdf_path, video_path)
        except (OSError, ValueError, RuntimeError) as exc:
            failures.append((hdf_path, exc))
            print(f"Failed: {exc}")
    if failures:
        raise RuntimeError(f"Failed to archive {len(failures)} of {len(rollouts)} rollout(s).")
    if args.strip_lang_emb_only:
        print(f"Removed lang_emb from {removed} dataset(s) across {len(rollouts)} rollout file(s).")


if __name__ == "__main__":
    main()
