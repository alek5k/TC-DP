import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Dict, Literal, Optional, Tuple

import cv2
import numpy as np

from diffusion_policy_contrib.real_world.single_realsense import SingleRealsense
from diffusion_policy_contrib.real_world.rtde_extended_interpolation_controller import (
    RTDEInterpolationController,
    RTDEReceiveKeys,
)
from diffusion_policy_contrib.shared_memory.shared_memory_util import SharedMemoryManager
from diffusion_policy_contrib.real_world.robotiq_gripper_controller import RobotiqGripperController

MAX_GOAL_VISITS = 6


class TaskPhase:
    APPROACH_START = "APPROACH_START"
    IN_CAMERA_VIEW = "IN_CAMERA_VIEW"
    APPROACH_GOAL = "APPROACH_GOAL"
    COMPLETED = "COMPLETED"


@dataclass
class LiftQARealConfig:
    enable_robot_movement: bool = True
    constrain_motion: bool = True

    control_hz: int = 15
    scheduled_waypoint_time_offset: float = 1 / 15
    render_camera_output: bool = False
    use_simulated_ur: bool = False
    robot_type: Literal["e-series", "cb3"] = "e-series"
    rtde_controller_gain: int = 100
    rtde_lookahead_time: float = 0.1
    robot_ip_sim: str = "localhost"
    robot_ip_real: str = "192.168.0.120"
    camera_serial: str = "938422072678"
    resolution: Tuple[int, int] = (320, 240)
    img_dim: int = 144
    camera_capture_fps: int = 30
    init_robot_joints: np.ndarray = field(
        default_factory=lambda: np.array(np.deg2rad([64.27, -69.99, 70, -90, -90, -25]))# np.array(np.deg2rad([84.81, -68.54, 71.3, -92.77, -90, -5.17])) # np.array(np.deg2rad([90.0, -90.0, 90.0, -90.0, -90.0, 0.0]))
    )

    goal_corner1: np.ndarray = field(default_factory=lambda: np.array([0.3442, -0.6238, 0.0500]))
    goal_corner2: np.ndarray = field(default_factory=lambda: np.array([0.6000, -0.7845, 0.2419]))

    max_pos_speed: float = 1.0
    max_rot_speed: float = 1.0

    robotiq_open_position: float = 0.0
    robotiq_closed_position: float = 1.0
    robotiq_initial_position: float = 0.0
    robotiq_gripper_frequency: int = 30
    robotiq_gripper_force: int = 125
    robotiq_gripper_velocity: int = 128 # max is 255
    robotiq_action_normalized: bool = True

    show_status_window: bool = True
    status_window_name: str = "LiftQA Goal Status"
    status_window_size: Tuple[int, int] = (640, 320)
    done_pause_seconds: float = 1.5
    reset_waypoint_duration_s: float = 2.0
    gripper_velocity_window_size: int = 3
    gripper_velocity_ema_alpha: float = 0.3

    height_to_end_episode: float = 0.29
    


    def robot_ip(self) -> str:
        return self.robot_ip_sim if self.use_simulated_ur else self.robot_ip_real

    def controller_frequency(self) -> int:
        if self.robot_type == "cb3":
            return 125
        return 300


class LiftQARealEnvironment:
    """
    Minimal Gym-like env:
      - reset() -> (obs, info)
      - step(action) -> (obs, reward, done, info)
      - close()

        action: np.ndarray shape (3,)
            [x, y, z, gripper]
            where x / y / z are absolute TCP targets and gripper is scalar in [0, 1].
            orientation is constrained by config.
    """

    def __init__(self, cfg: LiftQARealConfig):
        self.cfg = cfg
        self.robotiq_gripper: RobotiqGripperController
        self._prev_gripper_action: Optional[float] = None
        self._status_window_initialized = False
        self._done_displayed_this_episode = False
        self._gripper_pos_history = deque(maxlen=max(2, int(cfg.gripper_velocity_window_size)))
        self._gripper_ts_history = deque(maxlen=max(2, int(cfg.gripper_velocity_window_size)))
        self._gripper_velocity_estimate = 0.0

        self.shm_manager = SharedMemoryManager()
        self.shm_manager.__enter__()
        self.robotiq_gripper = RobotiqGripperController(
            shm_manager=self.shm_manager,
            hostname=cfg.robot_ip(),
            frequency=cfg.robotiq_gripper_frequency,
            gripper_force=cfg.robotiq_gripper_force,
            gripper_velocity=cfg.robotiq_gripper_velocity,
            verbose=False,
        )

        self.robot = RTDEInterpolationController(
            shm_manager=self.shm_manager,
            frequency=cfg.controller_frequency(),
            robot_ip=cfg.robot_ip(),
            max_pos_speed=self.cfg.max_pos_speed,
            max_rot_speed=self.cfg.max_rot_speed,
            launch_timeout=30,
            joints_init=cfg.init_robot_joints,
            lookahead_time=cfg.rtde_lookahead_time,
            verbose=False,
            gain=cfg.rtde_controller_gain,
        )

        self.camera = SingleRealsense(
            serial_number=cfg.camera_serial,
            shm_manager=self.shm_manager,
            resolution=cfg.resolution,
            capture_fps=cfg.camera_capture_fps,
            put_fps=cfg.camera_capture_fps,
            put_downsample=False,
            verbose=False,
        )

        self.robot.start(wait=True)
        self.camera.start(wait=True)
        self.robotiq_gripper.start(wait=True)
        self._init_status_window()

        time.sleep(2.0)
        ref_state = self.robot.get_state().copy()
        self.robot_reference_tcp = ref_state[RTDEReceiveKeys.TargetTCPPose]

    def _init_status_window(self):
        if not self.cfg.show_status_window:
            return
        try:
            cv2.namedWindow(self.cfg.status_window_name, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(self.cfg.status_window_name, *self.cfg.status_window_size)
            self._status_window_initialized = True
            self._render_status_window("IDLE", self._current_visit_timer if hasattr(self, "_current_visit_timer") else 0.0)
        except Exception:
            self._status_window_initialized = False

    def _render_status_window(self, state: str, timer_s: float):
        if not self.cfg.show_status_window or not self._status_window_initialized:
            return

        width, height = self.cfg.status_window_size
        canvas = np.zeros((height, width, 3), dtype=np.uint8)

        if state == "AT_GOAL":
            color = (0, 0, 255)  # red
            title = "AT GOAL"
        elif state == "EXIT_NOW":
            color = (0, 255, 0)  # green
            title = "EXIT NOW"
        elif state == "DONE":
            color = (255, 140, 0)  # orange
            title = "EPISODE DONE"
        else:
            color = (70, 70, 70)
            title = "APPROACH"

        canvas[:] = color
        cv2.putText(canvas, title, (16, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(canvas, f"Hold: {timer_s:.2f}s", (16, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.imshow(self.cfg.status_window_name, canvas)
        cv2.waitKey(1)

    def _gripper_to_native_position(self, position: float) -> float:
        position = float(np.clip(position, self.cfg.robotiq_open_position, self.cfg.robotiq_closed_position))
        if not self.cfg.robotiq_action_normalized:
            return position

        min_pos = float(self.robotiq_gripper.min_gripper_position)
        max_pos = float(self.robotiq_gripper.max_gripper_position)
        denom = max(self.cfg.robotiq_closed_position - self.cfg.robotiq_open_position, 1e-8)
        alpha = (position - self.cfg.robotiq_open_position) / denom
        return min_pos + alpha * (max_pos - min_pos)

    def _gripper_from_native_position(self, position: float) -> float:
        position = float(position)
        if not self.cfg.robotiq_action_normalized:
            return position

        min_pos = float(self.robotiq_gripper.min_gripper_position)
        max_pos = float(self.robotiq_gripper.max_gripper_position)
        denom = max(max_pos - min_pos, 1e-8)
        alpha = (position - min_pos) / denom
        mapped = self.cfg.robotiq_open_position + alpha * (self.cfg.robotiq_closed_position - self.cfg.robotiq_open_position)
        return float(np.clip(mapped, self.cfg.robotiq_open_position, self.cfg.robotiq_closed_position))

    def _gripper_get_position(self) -> float:
        state = self.robotiq_gripper.get_state()
        native_pos = float(np.asarray(state["GripperPose"]).reshape(-1)[0])
        return self._gripper_from_native_position(native_pos)

    def _update_smoothed_gripper_velocity(self, pos: float, ts: float) -> float:
        self._gripper_pos_history.append(float(pos))
        self._gripper_ts_history.append(float(ts))

        if len(self._gripper_pos_history) < 2:
            return self._gripper_velocity_estimate

        t = np.asarray(self._gripper_ts_history, dtype=np.float64)
        p = np.asarray(self._gripper_pos_history, dtype=np.float64)
        t = t - t[0]
        if float(t[-1]) <= 1e-6:
            return self._gripper_velocity_estimate

        # Fit slope over a short window to smooth quantized / delayed gripper position updates.
        slope = float(np.polyfit(t, p, deg=1)[0])
        alpha = float(np.clip(self.cfg.gripper_velocity_ema_alpha, 0.0, 1.0))
        self._gripper_velocity_estimate = (1.0 - alpha) * self._gripper_velocity_estimate + alpha * slope
        return self._gripper_velocity_estimate
        

    def _gripper_set_position(self, position: float) -> None:
        position = float(np.clip(position, self.cfg.robotiq_open_position, self.cfg.robotiq_closed_position))
        native_pos = self._gripper_to_native_position(position)
        self.robotiq_gripper.schedule_waypoint(native_pos, time.time() + self.cfg.scheduled_waypoint_time_offset)

    def _reset_goal_tracking(self):
        self.wait_time = 0.0
        self.wait_times_each_visit = []
        self._current_visit_timer = 0.0
        self._visit_active = False
        self.current_task_phase = TaskPhase.APPROACH_START
        goal_corners = np.vstack((self.cfg.goal_corner1, self.cfg.goal_corner2))
        self.goal_min = np.min(goal_corners, axis=0)
        self.goal_max = np.max(goal_corners, axis=0)

    def _update_goal_tracking(self, obs: Dict[str, Any]):
        tcp = obs["current_tcp"]
        outside_goal_threshold = 0.05
        inside_goal_area = ((self.goal_min <= tcp[:3]).all() and (tcp[:3] <= self.goal_max).all())
        outside_goal_area = ((tcp[:3] < self.goal_min - outside_goal_threshold).any() or (tcp[:3] > self.goal_max + outside_goal_threshold).any())
        exit_cue_active = self._visit_active and (self._current_visit_timer >= 2.0)

        if inside_goal_area and not self._visit_active:
            self._visit_active = True
            self._current_visit_timer = 0.0
            print("ENTERED GOAL")

        if self._visit_active:
            self._current_visit_timer += 1.0 / self.cfg.control_hz
            if 2.0 <= self._current_visit_timer <= 2.0 + (1 / self.cfg.control_hz):
                print("EXIT")
        exit_cue_active = self._visit_active and (self._current_visit_timer >= 2.0)

        if self._visit_active and outside_goal_area:
            self.wait_times_each_visit.append(self._current_visit_timer) 
            self._current_visit_timer = 0.0
            self._visit_active = False
            exit_cue_active = False

        if inside_goal_area:
            self.wait_time += 1.0 / self.cfg.control_hz
            self.current_task_phase = TaskPhase.IN_CAMERA_VIEW

        if outside_goal_area and self.current_task_phase == TaskPhase.IN_CAMERA_VIEW:
            self.current_task_phase = TaskPhase.APPROACH_GOAL

        if self.current_task_phase == TaskPhase.APPROACH_GOAL and tcp[2] > self.cfg.height_to_end_episode:
            self.current_task_phase = TaskPhase.COMPLETED

        if exit_cue_active:
            self._render_status_window("EXIT_NOW", self._current_visit_timer)
        elif inside_goal_area:
            self._render_status_window("AT_GOAL", self._current_visit_timer)
        else:
            self._render_status_window("APPROACH", 0.0)

        obs["wait_time"] = self.wait_time
        obs["wait_times_each_visit"] = np.pad(
            np.asarray(self.wait_times_each_visit, dtype=np.float32)[:MAX_GOAL_VISITS],
            (0, max(0, MAX_GOAL_VISITS - len(self.wait_times_each_visit))),
            constant_values=-1,
        )

    def reset(self) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        self.timestep = 0
        self._gripper_set_position(0.0) # 0 is open, 1 is closed

        # WARMUP - Get obs a couple times to initialize camera - warmup
        self.current_robot_state = self.robot.get_state()
        self.current_gripper_pos = self._gripper_get_position()
        self._get_obs()
        time.sleep(0.5)
        self.current_robot_state = self.robot.get_state()
        self.current_gripper_pos = self._gripper_get_position()
        self._get_obs()

        # RETURN ROBOT TO START POS
        curr_pose = np.asarray(self.current_robot_state[RTDEReceiveKeys.TargetTCPPose], dtype=np.float32).copy()
        # self.robot_reference_tcp = np.asarray(self.current_robot_state[RTDEReceiveKeys.TargetTCPPose], dtype=np.float32).copy()
        if self.cfg.enable_robot_movement:
            # start_pose = self.robot_reference_tcp.copy()
            target_pose = self.robot_reference_tcp.copy()
            # half_box = 0.5 * self.cfg.reset_waypoint_box_size_m
            # target_pose[:3] = self.robot_reference_tcp[:3] + np.random.uniform(-half_box, half_box, size=3).astype(np.float32)
            target_pose = self._apply_motion_constraint(target_pose)
            n_chunks = max(1, int(round(self.cfg.reset_waypoint_duration_s * self.cfg.control_hz)))
            now = time.time()
            for i in range(1, n_chunks + 1):
                alpha = i / n_chunks
                interp_pose = curr_pose.copy()
                interp_pose[:3] = (1.0 - alpha) * curr_pose[:3] + alpha * target_pose[:3]
                interp_pose[3:6] = target_pose[3:6]
                self.robot.schedule_waypoint(interp_pose.astype(np.float32), now + i * self.cfg.scheduled_waypoint_time_offset)

            time.sleep(self.cfg.reset_waypoint_duration_s)
            self.current_robot_state = self.robot.get_state()
        
        self._reset_goal_tracking()
        self._done_displayed_this_episode = False

        self.current_gripper_pos = self._gripper_get_position()
        self.prev_gripper_pos = self.current_gripper_pos
        self.current_gripper_timestamp = time.time()
        self.prev_gripper_timestamp = time.time() - (1.0 / self.cfg.control_hz)
        self._gripper_pos_history.clear()
        self._gripper_ts_history.clear()
        self._gripper_velocity_estimate = 0.0
        self._update_smoothed_gripper_velocity(self.current_gripper_pos, self.current_gripper_timestamp)

        obs = self._get_obs()
        self._update_goal_tracking(obs)
        info: Dict[str, Any] = {}
        self._prev_gripper_action = None
        return obs

    def _parse_action(self, action: np.ndarray) -> Tuple[float, float, float, float]:
        action = np.asarray(action, dtype=np.float32).reshape(-1)
        if action.shape[0] != 4:
            raise ValueError(f"Expected action with 4 values [x, y, z, gripper], got {action.shape[0]}")
        return float(action[0]), float(action[1]), float(action[2]), float(action[3])

    def _apply_motion_constraint(self, target_pose: np.ndarray) -> np.ndarray:
        constrained_pose = np.asarray(target_pose, dtype=np.float32).copy()
        if self.cfg.constrain_motion:
            # Constrain orientation only.
            # constrained_pose[3:6] = np.asarray(self.robot_reference_tcp[3:6], dtype=np.float32)
            constrained_pose[0] = np.clip(target_pose[0], -0.3873, 0.495) # x
            constrained_pose[1] = np.clip(target_pose[1], -0.800, -0.33) # y
            constrained_pose[2] = np.clip(target_pose[2], 0.0390, 0.4)
        return constrained_pose

    def step(self, action: np.ndarray):
        self.timestep += 1
        step_start = time.time()

        target_x, target_y, target_z, gripper_action = self._parse_action(action)
        current_tcp = np.asarray(self.current_robot_state[RTDEReceiveKeys.TargetTCPPose], dtype=np.float32)
        target_pose = current_tcp.copy()
        target_pose[0] = target_x
        target_pose[1] = target_y
        target_pose[2] = target_z
        target_pose = self._apply_motion_constraint(target_pose)
        self._prev_gripper_action = gripper_action

        if self.cfg.enable_robot_movement:
            self.robot.schedule_waypoint(target_pose, time.time() + self.cfg.scheduled_waypoint_time_offset)

        if gripper_action < 0.5:
            gripper_action = 0.0
        else:
            gripper_action = 1.0
            
        self._gripper_set_position(gripper_action)

        step_duration = time.time() - step_start
        target_duration = 1.0 / self.cfg.control_hz
        while step_duration < target_duration:
            time.sleep(target_duration / 100)
            step_duration = time.time() - step_start

        self.current_robot_state = self.robot.get_state()
        self.prev_gripper_pos = self.current_gripper_pos
        self.current_gripper_pos = self._gripper_get_position()
        self.prev_gripper_timestamp = self.current_gripper_timestamp
        self.current_gripper_timestamp = time.time()
        self._update_smoothed_gripper_velocity(self.current_gripper_pos, self.current_gripper_timestamp)

        obs = self._get_obs()
        self._update_goal_tracking(obs)
        reward = 0.0
        done = self.current_task_phase == TaskPhase.COMPLETED
        info = {}

        if done and not self._done_displayed_this_episode:
            self._render_status_window("DONE", self._current_visit_timer)
            self._done_displayed_this_episode = True

        step_dt = time.time() - step_start
        obs["real_hz"] = 1 / step_dt
        return obs, reward, done, info

    def close(self):
        try:
            self.camera.stop(wait=True)
        except Exception:
            pass
        try:
            self.robot.stop(wait=True)
        except Exception:
            pass
        try:
            self.robotiq_gripper.stop(wait=True)
        except Exception:
            pass
        if self._status_window_initialized:
            try:
                cv2.destroyWindow(self.cfg.status_window_name)
            except Exception:
                pass
        self.shm_manager.__exit__(None, None, None)

    def _get_obs(self) -> Dict[str, Any]:
        obs: Dict[str, Any] = {}
        obs["step_count"] = self.timestep
        obs["cur_time"] = time.time()

        tcp = self.current_robot_state[RTDEReceiveKeys.TargetTCPPose]
        tcp_speed = self.current_robot_state[RTDEReceiveKeys.TargetTCPSpeed]
        joints = self.current_robot_state[RTDEReceiveKeys.ActualQ]
        joint_velocity = self.current_robot_state[RTDEReceiveKeys.ActualQd]
        speed_scaling = self.current_robot_state[RTDEReceiveKeys.SpeedScalingCombined]

        image_data = self.camera.get()

        obs["current_tcp"] = tcp
        obs["current_tcp_speed"] = tcp_speed
        obs["current_robot_joints"] = joints
        obs["current_robot_joints_velocity"] = joint_velocity
        obs["speed_scaling"] = speed_scaling

        obs["current_gripper_joints"] = np.array([self.current_gripper_pos], dtype=np.float32)
        gripper_velocity = float(self._gripper_velocity_estimate)
        obs["current_gripper_joint_velocity"] = np.array([gripper_velocity], dtype=np.float32)

        mag_joint = np.linalg.norm(joint_velocity)
        mag_gripper = np.linalg.norm(gripper_velocity)
        mag_total = mag_joint + 0.5 * mag_gripper
        obs["agent_velocity"] = mag_total
        obs["agent_pose"] = np.concatenate([obs["current_tcp"][:3], obs["current_gripper_joints"]], axis=0)

        if isinstance(image_data, dict) and "color" in image_data:
            rgb_img = image_data["color"]
        elif isinstance(image_data, dict) and len(image_data) > 0:
            first_val = next(iter(image_data.values()))
            if isinstance(first_val, dict) and "color" in first_val:
                rgb_img = first_val["color"]
            else:
                raise ValueError("SingleRealsense output does not include a 'color' frame")
        else:
            raise ValueError("Unexpected SingleRealsense output format")

        rgb_img = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2RGB)
        rgb_img_resized = cv2.resize(
            rgb_img,
            (self.cfg.img_dim, self.cfg.img_dim),
            interpolation=cv2.INTER_AREA,
        )
        # h, w = rgb_img.shape[:2]
        # crop_size = min(144, h, w)
        # y0 = (h - crop_size) // 2
        # x0 = (w - crop_size) // 2
        # rgb_img = rgb_img[y0 : y0 + crop_size, x0 : x0 + crop_size]
        img_resnet_format = np.moveaxis(rgb_img_resized.astype(np.float32) / 255.0, -1, 0) # HWC to CHW
        obs["full_image"] = img_resnet_format
        obs["full_image_raw"] = rgb_img.astype(np.float32)

        return obs
