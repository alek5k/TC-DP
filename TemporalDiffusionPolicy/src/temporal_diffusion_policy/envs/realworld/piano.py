import time
from dataclasses import dataclass, field
from threading import Lock
from typing import Any, Dict, Literal, Optional, Tuple, Union

import cv2
import numpy as np

from diffusion_policy_contrib.real_world.single_realsense import SingleRealsense
from diffusion_policy_contrib.real_world.rtde_receive_recorder import (
    RTDEReceiveKeys,
    RTDEReceiveRecorder,
)
from diffusion_policy_contrib.shared_memory.shared_memory_util import SharedMemoryManager


class AudioRecorder:
    """Continuously capture microphone input and expose fixed-size recent frames."""

    def __init__(
        self,
        sample_rate: int,
        channels: int,
        samples_per_observation: int,
        device: Optional[Union[int, str]] = None,
    ):
        try:
            import sounddevice as sd
        except ImportError as exc:
            raise ImportError(
                "Audio capture requires sounddevice. Install the project requirements first."
            ) from exc

        if sample_rate <= 0 or channels <= 0 or samples_per_observation <= 0:
            raise ValueError("Audio sample rate, channels, and frame size must be positive.")

        self.sample_rate = sample_rate
        self.channels = channels
        self.samples_per_observation = samples_per_observation
        self._capacity = max(samples_per_observation * 4, sample_rate)
        self._samples = np.empty((0, channels), dtype=np.int16)
        self._lock = Lock()
        self._stream = sd.InputStream(
            samplerate=sample_rate,
            channels=channels,
            dtype="int16",
            device=device,
            callback=self._on_audio,
        )
        self._stream.start()

    def _on_audio(self, indata, frames, time_info, status) -> None:
        if status:
            print(f"[Piano audio] {status}")
        with self._lock:
            self._samples = np.concatenate((self._samples, indata.copy()), axis=0)[
                -self._capacity:
            ]

    def clear(self) -> None:
        with self._lock:
            self._samples = np.empty((0, self.channels), dtype=np.int16)

    def get_latest(self) -> np.ndarray:
        with self._lock:
            samples = self._samples[-self.samples_per_observation :].copy()
        if len(samples) < self.samples_per_observation:
            samples = np.pad(
                samples,
                ((self.samples_per_observation - len(samples), 0), (0, 0)),
                constant_values=0,
            )
        return samples

    def close(self) -> None:
        self._stream.stop()
        self._stream.close()


@dataclass
class PianoConfig:
    enable_robot_movement: bool = True
    constrain_motion: bool = False
    control_hz: int = 15
    scheduled_waypoint_time_offset: float = 1 / 15
    render_camera_output: bool = False
    use_simulated_ur: bool = False
    controller_frequency: int = 300 # 125 for cb3, 300 for e-series
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
    max_pos_speed: float = 1.0
    max_rot_speed: float = 1.0
    enable_audio: bool = False
    audio_device: Optional[Union[int, str]] = None
    audio_sample_rate: int = 16_000
    audio_channels: int = 1
    audio_samples_per_observation: Optional[int] = None
    reset_waypoint_duration_s: float = 2.0

    def robot_ip(self) -> str:
        return self.robot_ip_sim if self.use_simulated_ur else self.robot_ip_real

    def audio_frame_size(self) -> int:
        if self.audio_samples_per_observation is not None:
            return self.audio_samples_per_observation
        return max(1, round(self.audio_sample_rate / self.control_hz))


class Piano:
    """
    Real-world piano environment.

    Actions are absolute TCP positions with shape ``(3,)``: ``[x, y, z]``.
    The robot retains its current TCP orientation.
    """

    def __init__(self, cfg: PianoConfig):
        self.cfg = cfg
        self.audio_recorder = None

        self.shm_manager = SharedMemoryManager()
        self.shm_manager.__enter__()

        if cfg.enable_robot_movement:
            # Import lazily so recorder-only runs never load the RTDE control
            # interface or create a control connection.
            from diffusion_policy_contrib.real_world.rtde_extended_interpolation_controller import (
                RTDEInterpolationController,
            )

            self.robot = RTDEInterpolationController(
                shm_manager=self.shm_manager,
                frequency=cfg.controller_frequency,
                robot_ip=cfg.robot_ip(),
                max_pos_speed=self.cfg.max_pos_speed,
                max_rot_speed=self.cfg.max_rot_speed,
                launch_timeout=30,
                joints_init=cfg.init_robot_joints,
                lookahead_time=cfg.rtde_lookahead_time,
                verbose=False,
                gain=cfg.rtde_controller_gain,
            )
        else:
            self.robot = RTDEReceiveRecorder(
                shm_manager=self.shm_manager,
                robot_ip=cfg.robot_ip(),
                frequency=cfg.controller_frequency,
                launch_timeout=30,
                verbose=False,
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
        if cfg.enable_audio:
            self.audio_recorder = AudioRecorder(
                sample_rate=cfg.audio_sample_rate,
                channels=cfg.audio_channels,
                samples_per_observation=cfg.audio_frame_size(),
                device=cfg.audio_device,
            )
        time.sleep(2.0)
        ref_state = self.robot.get_state().copy()
        self.robot_reference_tcp = ref_state[RTDEReceiveKeys.TargetTCPPose]

    def reset(self) -> Dict[str, Any]:
        self.timestep = 0
        if self.audio_recorder is not None:
            self.audio_recorder.clear()

        # WARMUP - Get obs a couple times to initialize camera - warmup
        self.current_robot_state = self.robot.get_state()
        self._get_obs()
        time.sleep(0.5)
        self.current_robot_state = self.robot.get_state()
        self._get_obs()

        # RETURN ROBOT TO START POS
        curr_pose = np.asarray(
            self.current_robot_state[RTDEReceiveKeys.TargetTCPPose], dtype=np.float32
        ).copy()
        if self.cfg.enable_robot_movement:
            target_pose = self._apply_motion_constraint(self.robot_reference_tcp.copy())
            n_chunks = max(
                1,
                int(round(self.cfg.reset_waypoint_duration_s * self.cfg.control_hz)),
            )
            now = time.time()
            for i in range(1, n_chunks + 1):
                alpha = i / n_chunks
                interp_pose = curr_pose.copy()
                interp_pose[:3] = (
                    (1.0 - alpha) * curr_pose[:3] + alpha * target_pose[:3]
                )
                interp_pose[3:6] = target_pose[3:6]
                self.robot.schedule_waypoint(
                    interp_pose.astype(np.float32),
                    now + i * self.cfg.scheduled_waypoint_time_offset,
                )

            time.sleep(self.cfg.reset_waypoint_duration_s)
            self.current_robot_state = self.robot.get_state()

        obs = self._get_obs()
        return obs

    def _apply_motion_constraint(self, target_pose: np.ndarray) -> np.ndarray:
        constrained_pose = np.asarray(target_pose, dtype=np.float32).copy()
        if self.cfg.constrain_motion:
            constrained_pose[0] = np.clip(target_pose[0], -0.3873, 0.495) # x
            constrained_pose[1] = np.clip(target_pose[1], -0.800, -0.33) # y
            constrained_pose[2] = np.clip(target_pose[2], 0.0390, 0.4)
        return constrained_pose

    def step(self, action: np.ndarray):
        """Move to the action's absolute ``[x, y, z]`` TCP position."""
        step_start = time.time()
        position = np.asarray(action, dtype=np.float32).reshape(-1)

        self.timestep += 1
        target_pose = np.asarray(self.current_robot_state[RTDEReceiveKeys.TargetTCPPose], dtype=np.float32).copy()
        target_pose[:3] = position
        target_pose = self._apply_motion_constraint(target_pose)
        if self.cfg.enable_robot_movement:
            self.robot.schedule_waypoint(target_pose,  time.time() + self.cfg.scheduled_waypoint_time_offset)

        target_duration = 1.0 / self.cfg.control_hz
        while time.time() - step_start < target_duration:
            time.sleep(target_duration / 100)

        self.current_robot_state = self.robot.get_state()
        obs = self._get_obs()
        obs["real_hz"] = 1.0 / (time.time() - step_start)
        return obs, 0.0, False, {}

    def close(self):
        if self.audio_recorder is not None:
            try:
                self.audio_recorder.close()
            except Exception:
                pass
        try:
            self.camera.stop(wait=True)
        except Exception:
            pass
        try:
            self.robot.stop(wait=True)
        except Exception:
            pass
        self.shm_manager.__exit__(None, None, None)

    def _get_obs(self) -> Dict[str, Any]:
        obs: Dict[str, Any] = {}
        obs["step_count"] = self.timestep
        obs["cur_time"] = time.time()

        tcp = self.current_robot_state[RTDEReceiveKeys.ActualTCPPose]
        tcp_speed = self.current_robot_state[RTDEReceiveKeys.ActualTCPSpeed]
        joints = self.current_robot_state[RTDEReceiveKeys.ActualQ]
        joint_velocity = self.current_robot_state[RTDEReceiveKeys.ActualQd]
        speed_scaling = self.current_robot_state[RTDEReceiveKeys.SpeedScalingCombined]

        image_data = self.camera.get()

        obs["current_tcp"] = tcp
        obs["current_tcp_speed"] = tcp_speed
        obs["current_robot_joints"] = joints
        obs["current_robot_joints_velocity"] = joint_velocity
        obs["speed_scaling"] = speed_scaling

        mag_joint = np.linalg.norm(joint_velocity)
        obs["agent_velocity"] = mag_joint
        obs["agent_pose"] = obs["current_tcp"][:3].copy()

        if self.audio_recorder is not None:
            obs["audio"] = self.audio_recorder.get_latest()
            obs["audio_sample_rate"] = np.int32(self.cfg.audio_sample_rate)

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
        img_resnet_format = np.moveaxis(rgb_img_resized.astype(np.float32) / 255.0, -1, 0) # HWC to CHW
        obs["full_image"] = img_resnet_format
        obs["full_image_raw"] = rgb_img.astype(np.float32)

        return obs


PianoEnvironment = Piano
