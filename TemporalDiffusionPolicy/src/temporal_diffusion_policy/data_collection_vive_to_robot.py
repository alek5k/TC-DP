"""Replay recorded VIVE trajectories and collect active robot observations."""

from dataclasses import asdict, dataclass, field
from typing import Optional

import draccus
import numpy as np
from scipy.spatial.transform import Rotation

from diffusion_policy_contrib.common.replay_buffer import ReplayBuffer
from diffusion_policy_contrib.real_world.rtde_extended_interpolation_controller import (
    RTDEReceiveKeys,
)
from temporal_diffusion_policy.envs.realworld.piano import Piano, PianoConfig

# find interface name:
# ip -br addr
# sudo ip addr add 192.168.0.221/24 dev eno1


@dataclass
class ViveToRobotDataCollectionConfig:
    vive_dataset_path: str = "data_new/demonstration/vive_corr_transformed.zarr"
    robot_dataset_path: str = "data_new/demonstration/vive_to_robot.zarr"
    start_episode: int = 0
    max_episodes: Optional[int] = None

    # Piano / RTDE replay configuration. Robot movement is always active.
    control_hz: int = 15
    scheduled_waypoint_time_offset: float = 1 / 15
    rtde_controller_gain: int = 300
    rtde_lookahead_time: float = 0.1
    robot_ip_real: str = "192.168.0.120"
    camera_serial: str = "938422072678"
    resolution: tuple[int, int] = (320, 240)
    img_dim: int = 144
    camera_capture_fps: int = 30
    init_robot_joints: np.ndarray = field(
        default_factory=lambda: np.array(
            np.deg2rad([58.86, -49.74, 100.40, -141.05, -89.82, -28.03]),
            dtype=np.float32,
        )
    )
    max_pos_speed: float = 1.0
    max_rot_speed: float = 1.0
    reset_waypoint_duration_s: float = 2.0
    constrain_motion: bool = False
    target_position_offset: tuple[float, float, float] = (0.0, 0.0, 0.0)


class ViveToRobotMapper:
    """Map tracker motion relative to its first pose onto a robot reference TCP."""

    def __init__(self) -> None:
        self.tracker_ref_to_robot_ref_t = np.zeros(3, dtype=np.float64)
        self.tracker_ref_to_robot_ref_r = Rotation.from_euler(
            seq="ZYX", angles=[0, 0, 0], degrees=True
        )
        self.robot_ref_t: Optional[np.ndarray] = None
        self.robot_ref_r: Optional[Rotation] = None
        self.tracker_ref_t: Optional[np.ndarray] = None
        self.tracker_ref_r: Optional[Rotation] = None

    def initialize(self, robot_reference_tcp: np.ndarray, vive_initial_pose: np.ndarray) -> None:
        robot_reference_tcp = np.asarray(robot_reference_tcp, dtype=np.float64)
        vive_initial_pose = _validate_vive_pose(vive_initial_pose)
        if robot_reference_tcp.shape != (6,):
            raise ValueError(
                "Robot reference TCP must have shape (6,), "
                f"got {robot_reference_tcp.shape}."
            )

        self.robot_ref_t = robot_reference_tcp[:3].copy()
        self.robot_ref_r = Rotation.from_rotvec(robot_reference_tcp[3:])
        vive_t = vive_initial_pose[:3, 3]
        vive_r = Rotation.from_matrix(vive_initial_pose[:3, :3])
        self.tracker_ref_t = vive_t + vive_r.apply(self.tracker_ref_to_robot_ref_t)
        self.tracker_ref_r = vive_r * self.tracker_ref_to_robot_ref_r

    def map_pose(self, vive_pose: np.ndarray) -> np.ndarray:
        if (
            self.robot_ref_t is None
            or self.robot_ref_r is None
            or self.tracker_ref_t is None
            or self.tracker_ref_r is None
        ):
            raise RuntimeError("Mapper is not initialized.")

        vive_pose = _validate_vive_pose(vive_pose)
        vive_t = vive_pose[:3, 3]
        vive_r = Rotation.from_matrix(vive_pose[:3, :3])
        vive_t = vive_t + vive_r.apply(self.tracker_ref_to_robot_ref_t)
        vive_r = vive_r * self.tracker_ref_to_robot_ref_r

        relative_t = self.tracker_ref_r.inv().apply(vive_t - self.tracker_ref_t)
        relative_r = self.tracker_ref_r.inv() * vive_r
        target_t = self.robot_ref_t + self.robot_ref_r.apply(relative_t)
        target_r = self.robot_ref_r * relative_r
        return np.concatenate((target_t, target_r.as_rotvec())).astype(np.float32)


def _validate_vive_pose(pose: np.ndarray) -> np.ndarray:
    pose = np.asarray(pose, dtype=np.float64)
    if pose.shape != (4, 4):
        raise ValueError(f"Expected VIVE pose shape (4, 4), got {pose.shape}.")
    if not np.isfinite(pose).all():
        raise ValueError("VIVE pose contains non-finite values.")
    return pose


def _to_serializable(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {key: _to_serializable(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_to_serializable(item) for item in value]
    return value


@draccus.wrap()
def main(cfg: ViveToRobotDataCollectionConfig) -> None:
    target_position_offset = np.asarray(cfg.target_position_offset, dtype=np.float32)
    if target_position_offset.shape != (3,):
        raise ValueError(
            "target_position_offset must contain exactly three values "
            f"[x, y, z], got shape {target_position_offset.shape}."
        )

    vive_buffer = ReplayBuffer.create_from_path(cfg.vive_dataset_path, mode="r")
    if "vive_pose" not in vive_buffer:
        raise KeyError(f"{cfg.vive_dataset_path} does not contain a 'vive_pose' array.")
    if not 0 <= cfg.start_episode < vive_buffer.n_episodes:
        raise ValueError(
            f"start_episode must be in [0, {vive_buffer.n_episodes}), "
            f"got {cfg.start_episode}."
        )
    stop_episode = vive_buffer.n_episodes
    if cfg.max_episodes is not None:
        if cfg.max_episodes <= 0:
            raise ValueError("max_episodes must be positive when provided.")
        stop_episode = min(stop_episode, cfg.start_episode + cfg.max_episodes)

    robot_buffer = ReplayBuffer.create_from_path(cfg.robot_dataset_path, mode="w")
    for key, value in asdict(cfg).items():
        robot_buffer.root.attrs[key] = _to_serializable(value)

    piano_env_cfg = PianoConfig(
        enable_robot_movement=True,
        constrain_motion=cfg.constrain_motion,
        control_hz=cfg.control_hz,
        scheduled_waypoint_time_offset=cfg.scheduled_waypoint_time_offset,
        use_simulated_ur=False,
        controller_frequency=300,
        rtde_controller_gain=cfg.rtde_controller_gain,
        rtde_lookahead_time=cfg.rtde_lookahead_time,
        robot_ip_real=cfg.robot_ip_real,
        camera_serial=cfg.camera_serial,
        resolution=cfg.resolution,
        img_dim=cfg.img_dim,
        camera_capture_fps=cfg.camera_capture_fps,
        init_robot_joints=cfg.init_robot_joints,
        max_pos_speed=cfg.max_pos_speed,
        max_rot_speed=cfg.max_rot_speed,
        reset_waypoint_duration_s=cfg.reset_waypoint_duration_s,
    )
    env = Piano(piano_env_cfg)
    
    try:
        input("Ensure robot is at its reference pose, then press Enter to begin robot replay.\n")
        env.current_robot_state = env.robot.get_state().copy()
        robot_reference_tcp = np.asarray(
            env.current_robot_state[RTDEReceiveKeys.ActualTCPPose], dtype=np.float32
        ).copy()
        env.robot_reference_tcp = robot_reference_tcp


        mapper = ViveToRobotMapper()

        for episode_id in range(cfg.start_episode, stop_episode):
            vive_episode = vive_buffer.get_episode(episode_id)
            vive_poses = np.asarray(vive_episode["vive_pose"])

            if episode_id == cfg.start_episode:
                mapper.initialize(robot_reference_tcp, vive_poses[0])

            if vive_poses.ndim != 3 or vive_poses.shape[1:] != (4, 4):
                raise ValueError(
                    f"Episode {episode_id} VIVE poses must have shape (T, 4, 4), "
                    f"got {vive_poses.shape}."
                )
            if len(vive_poses) == 0:
                continue

            env.reset()
            episode: list[dict] = []
            for vive_pose in vive_poses:
                target_tcp = mapper.map_pose(vive_pose)
                target_tcp[:3] += target_position_offset
                action = target_tcp[:3]
                obs, reward, done, _ = env.step(action)
                save = obs.copy()
                save["action"] = action.copy()
                save["target_tcp"] = target_tcp
                save["vive_pose"] = np.asarray(vive_pose, dtype=np.float32)
                save["reward"] = np.float32(reward)
                save["done"] = bool(done)
                episode.append(save)
                if done:
                    break

            if episode:
                episode[-1]["done"] = True
                robot_buffer.add_episode(
                    {
                        key: np.stack([step[key] for step in episode])
                        for key in episode[0]
                    },
                    compressors="disk",
                )
                print(
                    f"Saved robot episode {episode_id} ({len(episode)} steps) "
                    f"-> {cfg.robot_dataset_path}"
                )
    finally:
        env.close()


if __name__ == "__main__":
    main()
