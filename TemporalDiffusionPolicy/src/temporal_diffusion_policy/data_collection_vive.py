"""Record raw VIVE tracker trajectories without connecting to the robot."""

import time
from dataclasses import asdict, dataclass
from queue import Empty, Queue
from threading import Thread

import draccus
import numpy as np

from diffusion_policy_contrib.common.replay_buffer import ReplayBuffer
from embodied_io.vive_tracker import ViveTrackerModule


@dataclass
class ViveDataCollectionConfig:
    dataset_path: str = "data_new/demonstration/vive.zarr"
    episodes: int = 5
    max_episode_length: int = 1_000
    control_hz: int = 15
    vive_tracker_name: str = "tracker_1"


def _validate_vive_pose(pose: np.ndarray) -> np.ndarray:
    pose = np.asarray(pose, dtype=np.float64)
    if pose.shape != (4, 4):
        raise ValueError(f"Expected VIVE pose shape (4, 4), got {pose.shape}.")
    if not np.isfinite(pose).all():
        raise ValueError("VIVE pose contains non-finite values.")
    return pose


def _read_terminal_input(command_queue: Queue) -> None:
    while True:
        try:
            command_queue.put(input().strip().lower())
        except EOFError:
            command_queue.put("q")
            return


def _save_episode(
    replay_buffer: ReplayBuffer,
    episode: list[dict],
    episode_id: int,
    dataset_path: str,
) -> None:
    if not episode:
        return
    episode[-1]["done"] = True
    replay_buffer.add_episode(
        {key: np.stack([step[key] for step in episode]) for key in episode[0]},
        compressors="disk",
    )
    print(f"Saved episode {episode_id} ({len(episode)} steps) -> {dataset_path}")


@draccus.wrap()
def main(cfg: ViveDataCollectionConfig) -> None:
    if cfg.control_hz <= 0:
        raise ValueError("control_hz must be positive.")

    tracker_module = ViveTrackerModule()
    tracker = tracker_module.devices.get(cfg.vive_tracker_name)
    if tracker is None:
        available = ", ".join(sorted(tracker_module.devices)) or "none"
        raise RuntimeError(
            f"VIVE tracker {cfg.vive_tracker_name!r} was not found "
            f"(available: {available})."
        )

    replay_buffer = ReplayBuffer.create_from_path(cfg.dataset_path, mode="w")
    for key, value in asdict(cfg).items():
        replay_buffer.root.attrs[key] = value

    commands: Queue = Queue()
    Thread(target=_read_terminal_input, args=(commands,), daemon=True).start()
    episode_id = 0
    recording = False
    episode: list[dict] = []
    period = 1.0 / cfg.control_hz
    print(
        "Press Enter to start recording. Press Enter again to save the episode. "
        "Type q then Enter to save and exit."
    )

    try:
        while episode_id < cfg.episodes:
            try:
                command = commands.get_nowait()
            except Empty:
                command = None

            if command == "q":
                if recording:
                    _save_episode(replay_buffer, episode, episode_id, cfg.dataset_path)
                break
            if command == "":
                if recording:
                    _save_episode(replay_buffer, episode, episode_id, cfg.dataset_path)
                    episode_id += 1
                    if episode_id >= cfg.episodes:
                        break
                    recording = False
                    print("Press Enter to start the next episode.")
                    continue
                episode = []
                recording = True
                print(f"Recording VIVE episode {episode_id}")

            if not recording:
                time.sleep(0.01)
                continue

            step_start = time.time()
            vive_pose = _validate_vive_pose(tracker.get_pose_matrix()).astype(np.float32)
            episode.append(
                {
                    "vive_pose": vive_pose,
                    "timestamp": np.float64(step_start),
                    "reward": np.float32(0.0),
                    "done": False,
                }
            )
            if len(episode) >= cfg.max_episode_length:
                _save_episode(replay_buffer, episode, episode_id, cfg.dataset_path)
                episode_id += 1
                recording = False
                print("Episode limit reached. Press Enter to start the next episode.")
                continue
            time.sleep(max(0.0, period - (time.time() - step_start)))
    except KeyboardInterrupt:
        print("VIVE recording cancelled.")
        if recording:
            _save_episode(replay_buffer, episode, episode_id, cfg.dataset_path)


if __name__ == "__main__":
    main()

