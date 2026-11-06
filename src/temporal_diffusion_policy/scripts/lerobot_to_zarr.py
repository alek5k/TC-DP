#!/usr/bin/env python3
"""Convert a LeRobot dataset into this repo's zarr replay-buffer format.

Example:
    python3 scripts/lerobot_to_zarr.py path/to/lerobot_dataset \
        --output data/demonstration/converted_lerobot.zarr \
        --image-key observation.images.cam_high \
        --image-size 144 \
        --overwrite
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from collections.abc import Iterable
from pathlib import Path
from typing import TYPE_CHECKING, Any


REPO_ROOT = Path.cwd()
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

if TYPE_CHECKING:
    import numpy as np

    from replay_buffer import ReplayBuffer


DEFAULT_STATE_KEYS = (
    "observation.state",
    "observation.states",
    "agent_pose",
    "state",
)
DEFAULT_ACTION_KEYS = ("action",)
DEFAULT_IMAGE_PREFIXES = (
    "observation.images.",
    "observation.image.",
    "observation.image",
)
DEFAULT_VELOCITY_KEYS = (
    "observation.velocity",
    "observation.velocities",
    "agent_velocity",
    "velocity",
)
DEFAULT_REWARD_KEYS = ("reward", "next.reward")
DEFAULT_DONE_KEYS = ("done", "next.done", "next_done")
DEFAULT_STEP_COUNT_KEYS = ("frame_index", "step_count")
np: Any = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Convert a local or Hub LeRobotDataset into zarr format with "
            "data/<key> arrays and meta/episode_ends."
        )
    )
    parser.add_argument(
        "source",
        help=(
            "LeRobot Hub repo id, or a local LeRobot dataset directory. For a "
            "local directory, --repo-id can be supplied if LeRobot requires it."
        ),
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Destination .zarr directory.",
    )
    parser.add_argument(
        "--repo-id",
        default=None,
        help="Repo id to pass to LeRobotDataset when SOURCE is a local path.",
    )
    parser.add_argument(
        "--root",
        default=None,
        help="Optional local root/cache path passed to LeRobotDataset.",
    )
    parser.add_argument(
        "--revision",
        default=None,
        help="Optional Hub revision passed to LeRobotDataset.",
    )
    parser.add_argument(
        "--episodes",
        default=None,
        help="Comma-separated episode indices to convert, e.g. 0,1,5.",
    )
    parser.add_argument(
        "--max-episodes",
        type=int,
        default=None,
        help="Convert at most this many episodes.",
    )
    parser.add_argument(
        "--state-key",
        default=None,
        help="LeRobot key to write as data/agent_pose. Defaults to auto-detect.",
    )
    parser.add_argument(
        "--action-key",
        default=None,
        help="LeRobot key to write as data/action. Defaults to auto-detect.",
    )
    parser.add_argument(
        "--image-key",
        default=None,
        help="LeRobot image key to write as data/full_image. Defaults to first observation image key.",
    )
    parser.add_argument(
        "--no-images",
        action="store_true",
        help="Do not write data/full_image.",
    )
    parser.add_argument(
        "--allow-missing-image",
        action="store_true",
        help="Continue without full_image if no image key is found.",
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=144,
        help="Resize images to square size. Use 0 to preserve original size. Default: 144.",
    )
    parser.add_argument(
        "--keep-uint8-images",
        action="store_true",
        help="Store images as uint8 instead of float32 in [0, 1].",
    )
    parser.add_argument(
        "--velocity-key",
        default=None,
        help=(
            "LeRobot key to write as data/agent_velocity. Defaults to auto-detect; "
            "if absent, velocity is synthesized from agent_pose frame deltas."
        ),
    )
    parser.add_argument(
        "--reward-key",
        default=None,
        help="LeRobot key to write as data/reward. Defaults to auto-detect, else zeros.",
    )
    parser.add_argument(
        "--done-key",
        default=None,
        help="LeRobot key to write as data/done. Defaults to auto-detect, else false.",
    )
    parser.add_argument(
        "--mark-last-done",
        action="store_true",
        help="When synthesizing done, mark the last frame of each episode true.",
    )
    parser.add_argument(
        "--step-count-key",
        default=None,
        help="LeRobot key to write as data/step_count. Defaults to frame_index if present.",
    )
    parser.add_argument(
        "--extra-key",
        action="append",
        default=[],
        metavar="SOURCE:DEST",
        help=(
            "Additional LeRobot key mapping. Can be repeated, e.g. "
            "--extra-key observation.images.wrist:hand_image."
        ),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace OUTPUT if it already exists.",
    )
    parser.add_argument(
        "--no-download-videos",
        action="store_true",
        help="Pass download_videos=False to LeRobotDataset.",
    )
    return parser.parse_args()


def parse_episode_list(value: str | None) -> list[int] | None:
    if value is None or value.strip() == "":
        return None
    episodes = []
    for chunk in value.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        episodes.append(int(chunk))
    return episodes


def parse_extra_mappings(values: Iterable[str]) -> dict[str, str]:
    mappings: dict[str, str] = {}
    for value in values:
        if ":" not in value:
            raise ValueError(f"Invalid --extra-key mapping '{value}'. Expected SOURCE:DEST.")
        source, dest = value.split(":", 1)
        source = source.strip()
        dest = dest.strip()
        if not source or not dest:
            raise ValueError(f"Invalid --extra-key mapping '{value}'. Expected SOURCE:DEST.")
        mappings[source] = dest
    return mappings


def import_lerobot_dataset():
    import_paths = (
        "lerobot.datasets",
        "lerobot.datasets.lerobot_dataset",
        "lerobot.common.datasets.lerobot_dataset",
    )
    errors = []
    for import_path in import_paths:
        try:
            module = __import__(import_path, fromlist=["LeRobotDataset"])
            return module.LeRobotDataset
        except (ImportError, AttributeError) as exc:
            errors.append(f"{import_path}: {exc}")

    raise SystemExit(
        "Could not import LeRobotDataset. Install LeRobot in this environment, for example:\n"
        "  pip install lerobot\n"
        "or install the version used to create your dataset.\n"
        "Tried:\n"
        + "\n".join(f"  - {error}" for error in errors)
    )


def import_numpy():
    try:
        import numpy
    except ImportError as exc:
        raise SystemExit(
            "Could not import numpy. Install this repo's dependencies before converting:\n"
            "  pip install -r requirements.txt"
        ) from exc
    return numpy


def import_replay_buffer():
    try:
        from replay_buffer import ReplayBuffer
    except ImportError as exc:
        raise SystemExit(
            "Could not import this repo's ReplayBuffer. Make sure project dependencies "
            "are installed, including zarr<3."
        ) from exc
    return ReplayBuffer


class LocalLeRobotV3Dataset:
    """Small local reader for LeRobot v3 datasets.

    This avoids depending on the full LeRobot Python stack when all data is
    already present on disk. It assumes row order matches video frame order,
    which is the standard v3 layout.
    """

    def __init__(self, root: Path, episodes: list[int] | None = None):
        try:
            import av
            import pyarrow.parquet as pq
        except ImportError as exc:
            raise SystemExit(
                "Local LeRobot v3 conversion needs pyarrow and av. Install them, "
                "or run with an environment that has LeRobot installed."
            ) from exc

        self.root = root
        self._av = av
        with (root / "meta" / "info.json").open("r", encoding="utf-8") as f:
            self.info = json.load(f)
        self.features = self.info.get("features", {})

        data_tables = []
        for parquet_path in sorted((root / "data").glob("chunk-*/file-*.parquet")):
            data_tables.append(pq.read_table(parquet_path))
        if not data_tables:
            raise FileNotFoundError(f"No parquet data files found under {root / 'data'}")

        table = data_tables[0] if len(data_tables) == 1 else pa_concat_tables(data_tables)
        self.rows = table.to_pylist()

        if episodes is not None:
            episode_set = set(episodes)
            self.rows = [row for row in self.rows if int(row["episode_index"]) in episode_set]

        self.video_keys = [
            key
            for key, spec in self.features.items()
            if isinstance(spec, dict) and spec.get("dtype") == "video"
        ]
        self._video_readers = {
            key: self._open_video_reader(key)
            for key in self.video_keys
            if (root / "videos" / key).exists()
        }

    def _open_video_reader(self, video_key: str):
        video_files = sorted((self.root / "videos" / video_key).glob("chunk-*/file-*.mp4"))
        if not video_files:
            return None
        if len(video_files) != 1:
            raise NotImplementedError(
                f"Local v3 video reader currently supports one file per key; "
                f"found {len(video_files)} for {video_key}."
            )
        container = self._av.open(str(video_files[0]))
        return {
            "container": container,
            "frames": container.decode(video=0),
            "next_index": 0,
            "last_frame": None,
        }

    def _read_video_frame(self, video_key: str, row_index: int) -> np.ndarray:
        reader = self._video_readers.get(video_key)
        if reader is None:
            raise KeyError(f"No local video reader for key '{video_key}'.")

        if row_index < reader["next_index"] - 1:
            raise RuntimeError("Local video reader only supports sequential access.")
        while reader["next_index"] <= row_index:
            try:
                frame = next(reader["frames"])
            except StopIteration as exc:
                raise IndexError(f"Video for {video_key} ended before frame {row_index}.") from exc
            reader["last_frame"] = frame.to_ndarray(format="rgb24")
            reader["next_index"] += 1
        return reader["last_frame"]

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        row = dict(self.rows[idx])
        absolute_index = int(row.get("index", idx))
        for video_key in self.video_keys:
            if video_key in self._video_readers:
                row[video_key] = self._read_video_frame(video_key, absolute_index)
        return row

    def close(self) -> None:
        for reader in self._video_readers.values():
            reader["container"].close()


def pa_concat_tables(tables: list[Any]):
    import pyarrow as pa

    return pa.concat_tables(tables)


def is_local_lerobot_v3(source_path: Path) -> bool:
    info_path = source_path / "meta" / "info.json"
    if not info_path.exists():
        return False
    try:
        with info_path.open("r", encoding="utf-8") as f:
            info = json.load(f)
    except json.JSONDecodeError:
        return False
    return str(info.get("codebase_version", "")).startswith("v3")


def load_lerobot_dataset(args: argparse.Namespace):
    source_path = Path(args.source).expanduser()
    if source_path.exists() and is_local_lerobot_v3(source_path):
        return LocalLeRobotV3Dataset(source_path, episodes=parse_episode_list(args.episodes))

    LeRobotDataset = import_lerobot_dataset()
    root = Path(args.root).expanduser() if args.root else None
    repo_id = args.source

    if source_path.exists():
        root = root or source_path
        repo_id = args.repo_id or f"local/{source_path.name}"
    elif args.repo_id is not None:
        repo_id = args.repo_id

    kwargs: dict[str, Any] = {
        "repo_id": repo_id,
        "root": root,
        "episodes": parse_episode_list(args.episodes),
        "revision": args.revision,
        "download_videos": not args.no_download_videos,
        "return_uint8": True,
    }
    kwargs = {key: value for key, value in kwargs.items() if value is not None}

    optional_kwargs = ("return_uint8", "download_videos", "revision", "episodes", "root")
    while True:
        try:
            return LeRobotDataset(**kwargs)
        except TypeError as exc:
            message = str(exc)
            removed = False
            for key in optional_kwargs:
                if key in kwargs and key in message:
                    kwargs.pop(key, None)
                    if key == "root" and source_path.exists() and args.repo_id is None:
                        kwargs["repo_id"] = str(source_path)
                    removed = True
                    break
            if not removed:
                raise


def feature_keys(dataset: Any) -> list[str]:
    features = getattr(dataset, "features", None)
    if isinstance(features, dict):
        return list(features.keys())

    hf_dataset = getattr(dataset, "hf_dataset", None)
    if hf_dataset is not None and hasattr(hf_dataset, "column_names"):
        return list(hf_dataset.column_names)

    first = dataset[0]
    return list(first.keys())


def choose_key(
    explicit_key: str | None,
    candidates: Iterable[str],
    available: set[str],
    label: str,
    required: bool = True,
) -> str | None:
    if explicit_key:
        if explicit_key not in available:
            raise KeyError(
                f"{label} key '{explicit_key}' not found. Available keys:\n"
                + "\n".join(f"  - {key}" for key in sorted(available))
            )
        return explicit_key

    for key in candidates:
        if key in available:
            return key

    if required:
        raise KeyError(
            f"Could not auto-detect {label} key. Available keys:\n"
            + "\n".join(f"  - {key}" for key in sorted(available))
        )
    return None


def choose_image_key(explicit_key: str | None, available: set[str]) -> str | None:
    if explicit_key:
        if explicit_key not in available:
            raise KeyError(
                f"Image key '{explicit_key}' not found. Available keys:\n"
                + "\n".join(f"  - {key}" for key in sorted(available))
            )
        return explicit_key

    for key in sorted(available):
        if any(key.startswith(prefix) for prefix in DEFAULT_IMAGE_PREFIXES):
            return key
    return None


def to_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach") and hasattr(value, "cpu"):
        value = value.detach().cpu().numpy()
    elif hasattr(value, "numpy") and not isinstance(value, np.ndarray):
        try:
            value = value.numpy()
        except TypeError:
            pass
    return np.asarray(value)


def as_vector(value: Any) -> np.ndarray:
    arr = to_numpy(value)
    if arr.ndim == 0:
        arr = arr.reshape(1)
    return arr.astype(np.float32) if arr.dtype == np.float64 else arr


def resize_chw_image(image: np.ndarray, image_size: int) -> np.ndarray:
    if image_size <= 0:
        return image
    if image.shape[-2:] == (image_size, image_size):
        return image

    try:
        import cv2
    except ImportError as exc:
        raise SystemExit(
            "--image-size requires opencv-python. Install it or pass --image-size 0."
        ) from exc

    hwc = np.moveaxis(image, 0, -1)
    resized = cv2.resize(hwc, (image_size, image_size), interpolation=cv2.INTER_AREA)
    if resized.ndim == 2:
        resized = resized[:, :, None]
    return np.moveaxis(resized, -1, 0)


def as_image(value: Any, image_size: int, keep_uint8: bool) -> np.ndarray:
    image = to_numpy(value)

    if image.ndim == 4 and image.shape[0] == 1:
        image = image[0]
    if image.ndim != 3:
        raise ValueError(f"Expected image with 3 dimensions, got shape {image.shape}.")

    if image.shape[0] not in (1, 3, 4) and image.shape[-1] in (1, 3, 4):
        image = np.moveaxis(image, -1, 0)

    if image.shape[0] == 4:
        image = image[:3]
    if image.shape[0] not in (1, 3):
        raise ValueError(f"Expected CHW or HWC image, got shape {image.shape}.")

    image = resize_chw_image(image, image_size)

    if keep_uint8:
        if image.dtype == np.uint8:
            return image
        clipped = np.clip(image, 0.0, 1.0) if np.nanmax(image) <= 1.5 else np.clip(image, 0, 255)
        return (clipped * 255.0).round().astype(np.uint8) if clipped.max() <= 1.5 else clipped.astype(np.uint8)

    image = image.astype(np.float32)
    if np.nanmax(image) > 1.5:
        image = image / 255.0
    return image


def frame_episode_index(frame: dict[str, Any], fallback: int) -> int:
    for key in ("episode_index", "episode"):
        if key in frame:
            return int(to_numpy(frame[key]).item())
    return fallback


def stack_episode(values: list[np.ndarray], key: str) -> np.ndarray:
    try:
        return np.stack(values, axis=0)
    except ValueError as exc:
        shapes = sorted({tuple(value.shape) for value in values})
        raise ValueError(f"Could not stack key '{key}'. Per-frame shapes: {shapes}") from exc


def synthesize_velocity(agent_pose: np.ndarray) -> np.ndarray:
    pose = agent_pose.astype(np.float32)
    if pose.ndim == 1:
        pose = pose[:, None]
    diffs = np.diff(pose, axis=0, prepend=pose[:1])
    return np.linalg.norm(diffs, axis=1).astype(np.float32)


def convert_episode(
    episode_frames: list[dict[str, np.ndarray]],
    mappings: dict[str, str],
    image_destinations: set[str],
    velocity_dest: str,
    reward_dest: str,
    done_dest: str,
    step_count_dest: str,
    mark_last_done: bool,
) -> dict[str, np.ndarray]:
    episode: dict[str, np.ndarray] = {}

    for dest in sorted(set(mappings.values())):
        values = [frame[dest] for frame in episode_frames if dest in frame]
        if len(values) != len(episode_frames):
            raise KeyError(f"Episode is missing mapped destination '{dest}' in some frames.")
        episode[dest] = stack_episode(values, dest)

    length = len(episode_frames)

    if step_count_dest not in episode:
        episode[step_count_dest] = np.arange(length, dtype=np.int64)
    else:
        episode[step_count_dest] = episode[step_count_dest].reshape(length)

    if reward_dest not in episode:
        episode[reward_dest] = np.zeros(length, dtype=np.float32)
    else:
        episode[reward_dest] = episode[reward_dest].reshape(length).astype(np.float32)

    if done_dest not in episode:
        done = np.zeros(length, dtype=bool)
        if mark_last_done and length > 0:
            done[-1] = True
        episode[done_dest] = done
    else:
        episode[done_dest] = episode[done_dest].reshape(length).astype(bool)

    if velocity_dest not in episode and "agent_pose" in episode:
        episode[velocity_dest] = synthesize_velocity(episode["agent_pose"])
    elif velocity_dest in episode:
        episode[velocity_dest] = episode[velocity_dest].reshape(length).astype(np.float32)

    for dest in image_destinations:
        if dest in episode and episode[dest].dtype == np.float64:
            episode[dest] = episode[dest].astype(np.float32)

    return episode


def write_episode(
    replay_buffer: ReplayBuffer,
    episode_frames: list[dict[str, np.ndarray]],
    mappings: dict[str, str],
    image_destinations: set[str],
    args: argparse.Namespace,
) -> None:
    episode = convert_episode(
        episode_frames=episode_frames,
        mappings=mappings,
        image_destinations=image_destinations,
        velocity_dest="agent_velocity",
        reward_dest="reward",
        done_dest="done",
        step_count_dest="step_count",
        mark_last_done=args.mark_last_done,
    )
    replay_buffer.add_episode(episode, compressors="disk")


def main() -> None:
    global np
    args = parse_args()
    np = import_numpy()

    output = Path(args.output).expanduser()
    if output.exists():
        if not args.overwrite:
            raise SystemExit(f"Output already exists: {output}. Pass --overwrite to replace it.")
        if output.is_dir():
            shutil.rmtree(output)
        else:
            output.unlink()
    output.parent.mkdir(parents=True, exist_ok=True)

    dataset = load_lerobot_dataset(args)
    keys = feature_keys(dataset)
    available = set(keys)

    state_key = choose_key(args.state_key, DEFAULT_STATE_KEYS, available, "state")
    action_key = choose_key(args.action_key, DEFAULT_ACTION_KEYS, available, "action")
    step_key = choose_key(args.step_count_key, DEFAULT_STEP_COUNT_KEYS, available, "step count", required=False)
    reward_key = choose_key(args.reward_key, DEFAULT_REWARD_KEYS, available, "reward", required=False)
    done_key = choose_key(args.done_key, DEFAULT_DONE_KEYS, available, "done", required=False)

    velocity_candidates = DEFAULT_VELOCITY_KEYS if args.velocity_key is None else (args.velocity_key,)
    velocity_key = choose_key(args.velocity_key, velocity_candidates, available, "velocity", required=False)

    image_key = None
    if not args.no_images:
        image_key = choose_image_key(args.image_key, available)
        if image_key is None and not args.allow_missing_image:
            raise KeyError(
                "Could not auto-detect an image key. Pass --image-key, --no-images, "
                "or --allow-missing-image."
            )

    source_to_dest: dict[str, str] = {
        state_key: "agent_pose",
        action_key: "action",
    }
    if step_key is not None:
        source_to_dest[step_key] = "step_count"
    if reward_key is not None:
        source_to_dest[reward_key] = "reward"
    if done_key is not None:
        source_to_dest[done_key] = "done"
    if velocity_key is not None:
        source_to_dest[velocity_key] = "agent_velocity"
    if image_key is not None:
        source_to_dest[image_key] = "full_image"

    source_to_dest.update(parse_extra_mappings(args.extra_key))
    missing_extra = sorted(source for source in source_to_dest if source not in available)
    if missing_extra:
        raise KeyError(
            "Mapped source keys were not found:\n"
            + "\n".join(f"  - {key}" for key in missing_extra)
            + "\nAvailable keys:\n"
            + "\n".join(f"  - {key}" for key in sorted(available))
        )

    image_destinations = {
        dest
        for source, dest in source_to_dest.items()
        if source == image_key or any(source.startswith(prefix) for prefix in DEFAULT_IMAGE_PREFIXES)
    }

    ReplayBuffer = import_replay_buffer()
    replay_buffer = ReplayBuffer.create_from_path(output, mode="w")
    replay_buffer.root.attrs["source_lerobot"] = args.source
    replay_buffer.root.attrs["source_lerobot_features"] = keys
    replay_buffer.root.attrs["converter"] = "scripts/lerobot_to_zarr.py"
    replay_buffer.root.attrs["key_mapping"] = dict(source_to_dest)
    replay_buffer.root.attrs["image_size"] = args.image_size
    replay_buffer.root.attrs["image_dtype"] = "uint8" if args.keep_uint8_images else "float32"

    print("Converting LeRobot dataset")
    print(f"  source: {args.source}")
    print(f"  output: {output}")
    print("  mappings:")
    for source, dest in sorted(source_to_dest.items(), key=lambda item: item[1]):
        print(f"    {source} -> {dest}")

    current_episode_index: int | None = None
    current_episode_frames: list[dict[str, np.ndarray]] = []
    converted_episodes = 0

    for row_index in range(len(dataset)):
        raw_frame = dataset[row_index]
        episode_index = frame_episode_index(raw_frame, fallback=converted_episodes)

        if current_episode_index is None:
            current_episode_index = episode_index

        if episode_index != current_episode_index:
            write_episode(
                replay_buffer,
                current_episode_frames,
                source_to_dest,
                image_destinations,
                args,
            )
            converted_episodes += 1
            print(
                f"  wrote episode {current_episode_index} "
                f"({len(current_episode_frames)} frames)"
            )
            if args.max_episodes is not None and converted_episodes >= args.max_episodes:
                break
            current_episode_index = episode_index
            current_episode_frames = []

        frame: dict[str, np.ndarray] = {}
        for source, dest in source_to_dest.items():
            if dest in image_destinations:
                frame[dest] = as_image(
                    raw_frame[source],
                    image_size=args.image_size,
                    keep_uint8=args.keep_uint8_images,
                )
            else:
                frame[dest] = as_vector(raw_frame[source])
        current_episode_frames.append(frame)

    should_flush_last = current_episode_frames and (
        args.max_episodes is None or converted_episodes < args.max_episodes
    )
    if should_flush_last:
        write_episode(
            replay_buffer,
            current_episode_frames,
            source_to_dest,
            image_destinations,
            args,
        )
        converted_episodes += 1
        print(f"  wrote episode {current_episode_index} ({len(current_episode_frames)} frames)")

    replay_buffer.root.attrs["converted_episodes"] = converted_episodes
    replay_buffer.root.attrs["converted_frames"] = int(replay_buffer.n_steps)

    print(f"Done. Converted {converted_episodes} episodes / {int(replay_buffer.n_steps)} frames.")


if __name__ == "__main__":
    main()
