import time
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple

import draccus
import numpy as np
from scipy.spatial.transform import Rotation

from diffusion_policy_contrib.real_world.fanuc.fanuc_interpolation_controller import (
    FanucInterpolationController,
    FanucReceiveKeys,
)
from diffusion_policy_contrib.shared_memory.shared_memory_util import SharedMemoryManager
from embodied_io.vive_tracker import ViveTrackerModule, VRTrackedDevice


@dataclass
class LiftQARealViveConfig:
    tracker_name: str = "tracker_1"
    auto_configure_steamvr: bool = False
    calibrate_on_reset: bool = True
    enable_robot_movement: bool = True
    control_hz: int = 60
    scheduled_waypoint_time_offset: float = 1 / control_hz
    robot_ip_real: str = "192.168.10.123"
    init_robot_joints: np.ndarray = field(
        default_factory=lambda: np.array(np.deg2rad([0.0, 0, 0, 0, -90, 0]))
    )
    max_target_lead_m: Optional[float] = 0.06


class TrackerCalibration:
    """Maps a Vive tracker pose into a robot TCP target pose."""

    def __init__(self):
        self.tracker_ref_to_robot_ref_t = np.zeros(3, dtype=np.float64)
        self.tracker_ref_to_robot_ref_r = Rotation.from_euler(seq="ZYX", angles=[90, 0, 0], degrees=True) # this reference works for this TCP: [0.0, 0.0, 0, 0, -90, 0]

        self.robot_ref_t: Optional[np.ndarray] = None
        self.robot_ref_r: Optional[Rotation] = None
        self.tracker_ref_t: Optional[np.ndarray] = None
        self.tracker_ref_r: Optional[Rotation] = None

    def initialize_calibration(self, robot_reference_tcp: np.ndarray, tracker_reference_pose: np.ndarray) -> None:
        robot_reference_tcp = np.asarray(robot_reference_tcp, dtype=np.float64)
        self.robot_ref_t = robot_reference_tcp[:3].copy()
        self.robot_ref_r = Rotation.from_rotvec(robot_reference_tcp[3:])

        tracker_ref_t = np.asarray(tracker_reference_pose[:3, 3], dtype=np.float64)
        tracker_ref_r = Rotation.from_matrix(tracker_reference_pose[:3, :3])
        self.tracker_ref_t = tracker_ref_t + tracker_ref_r.apply(self.tracker_ref_to_robot_ref_t)
        self.tracker_ref_r = tracker_ref_r * self.tracker_ref_to_robot_ref_r

    def map_tracker_pose_to_target(self, tracker_pose_mat: np.ndarray) -> np.ndarray:
        if self.robot_ref_t is None or self.robot_ref_r is None or self.tracker_ref_t is None or self.tracker_ref_r is None:
            raise RuntimeError("TrackerCalibration is not initialized. Call initialize_calibration first.")

        tracker_pose_t = np.asarray(tracker_pose_mat[:3, 3], dtype=np.float64)
        tracker_pose_r = Rotation.from_matrix(tracker_pose_mat[:3, :3])

        rhs_t = tracker_pose_t + tracker_pose_r.apply(self.tracker_ref_to_robot_ref_t)
        rhs_r = tracker_pose_r * self.tracker_ref_to_robot_ref_r

        inv_tracker_ref_r = self.tracker_ref_r.inv()
        inv_tracker_ref_t = -inv_tracker_ref_r.apply(self.tracker_ref_t)

        tracker_in_robot_t = inv_tracker_ref_t + inv_tracker_ref_r.apply(rhs_t)
        tracker_in_robot_r = inv_tracker_ref_r * rhs_r

        new_robot_t = self.robot_ref_t + self.robot_ref_r.apply(tracker_in_robot_t)
        new_robot_r = self.robot_ref_r * tracker_in_robot_r
        return np.concatenate([new_robot_t, new_robot_r.as_rotvec()])


class LiftQARealViveEnvironment:
    def __init__(self, cfg: LiftQARealViveConfig):
        self.cfg = cfg
        self.shm_manager = SharedMemoryManager()
        self.shm_manager.__enter__()

        self.robot = FanucInterpolationController(
            shm_manager=self.shm_manager,
            robot_ip=cfg.robot_ip_real,
            stream_motion_data_style="cartesian",
            max_pos_speed= 500 / 1000.0,
            max_pos_acceleration= 10000 / 1000.0,
            max_pos_jerk= 20000.0 / 1000.0,
            max_rot_speed=np.deg2rad(180.0),
            max_rot_acceleration=np.deg2rad(1000.0),
            max_rot_jerk=np.deg2rad(5000.0),
            joints_init=self.cfg.init_robot_joints,
            verbose=True,
            max_target_lead_m=self.cfg.max_target_lead_m,
        )
        self.robot.start(wait=True)

        time.sleep(2.0)
        ref_state = self.robot.get_state().copy()
        self.robot_reference_tcp = ref_state[FanucReceiveKeys.CommandTCPPose]

        self.vive = ViveTrackerModule(auto_configure_steamvr=self.cfg.auto_configure_steamvr)
        self.vive.print_discovered_objects()
        self.tracker: Optional[VRTrackedDevice] = self.vive.devices.get(self.cfg.tracker_name)
        if self.tracker is None:
            raise RuntimeError(f"Tracker '{self.cfg.tracker_name}' not found. Available devices: {list(self.vive.devices.keys())}")

        self.calibration = TrackerCalibration()
        self.timestep = 0
        self.current_robot_state = self.robot.get_state()

    def _get_tracker_pose(self) -> Optional[np.ndarray]:
        pose = self.tracker.get_pose_matrix()
        if pose is None:
            return None
        pose = np.asarray(pose, dtype=np.float64)
        if pose.shape != (4, 4) or not np.isfinite(pose).all():
            return None
        return pose

    def _calibrate_from_tracker(self) -> None:
        print("Move the tracker to the desired starting pose and hold it steady...")
        pose = None
        for _ in range(50):
            pose = self._get_tracker_pose()
            if pose is not None:
                break
            time.sleep(0.02)

        if pose is None:
            raise RuntimeError("No valid tracker pose received during calibration")

        self.calibration.initialize_calibration(self.robot_reference_tcp, pose)
        print("Tracker calibration initialized")

    def reset(self) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        self.timestep = 0
        self.current_robot_state = self.robot.get_state()
        if self.cfg.calibrate_on_reset:
            self._calibrate_from_tracker()
        obs = self._get_obs()
        info: Dict[str, Any] = {}
        return obs, info

    def step(self) -> Tuple[Dict[str, Any], float, bool, Dict[str, Any]]:
        self.timestep += 1
        step_start = time.time()

        tracker_pose = self._get_tracker_pose()
        if tracker_pose is None:
            target_pose = np.asarray(self.current_robot_state[FanucReceiveKeys.CommandTCPPose], dtype=np.float32).copy()
        else:
            target_pose = self.calibration.map_tracker_pose_to_target(tracker_pose)

        if self.cfg.enable_robot_movement:
            self.robot.schedule_waypoint(target_pose, time.time() + self.cfg.scheduled_waypoint_time_offset)

        step_duration = time.time() - step_start
        target_duration = 1.0 / self.cfg.control_hz
        while step_duration < target_duration:
            time.sleep(target_duration / 100)
            step_duration = time.time() - step_start

        self.current_robot_state = self.robot.get_state()
        obs = self._get_obs()
        reward = 0.0
        done = False
        info = {}

        step_dt = time.time() - step_start
        obs["real_hz"] = 1 / step_dt
        return obs, reward, done, info

    def close(self) -> None:
        try:
            self.robot.stop(wait=True)
        except Exception:
            pass
        self.shm_manager.__exit__(None, None, None)

    def _get_obs(self) -> Dict[str, Any]:
        obs: Dict[str, Any] = {}
        obs["step_count"] = self.timestep
        obs["cur_time"] = time.time()

        tcp = self.current_robot_state[FanucReceiveKeys.TargetTCPPose]
        tcp_cmd = self.current_robot_state[FanucReceiveKeys.CommandTCPPose]
        tcp_actual = self.current_robot_state[FanucReceiveKeys.ActualTCPPose]
        obs["target_tcp"] = tcp
        obs["command_tcp"] = tcp_cmd
        obs["actual_tcp"] = tcp_actual

        tracker_pose = self._get_tracker_pose()
        if tracker_pose is not None:
            obs["tracker_pose"] = tracker_pose
        return obs


@draccus.wrap()
def main(cfg: LiftQARealViveConfig):
    env = LiftQARealViveEnvironment(cfg=cfg)
    try:
        obs, info = env.reset()
        print("Teleop started. Press Ctrl+C to exit.")
        while True:
            obs, reward, done, info = env.step()
            if done:
                break
    except KeyboardInterrupt:
        print("User cancelled teleop")
    finally:
        env.close()


if __name__ == "__main__":
    main()
