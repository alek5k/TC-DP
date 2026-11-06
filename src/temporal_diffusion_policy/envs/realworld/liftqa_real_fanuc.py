import time
from dataclasses import dataclass, field
from typing import Any, Dict, Literal, Optional, Tuple

import draccus
import numpy as np
from pynput.keyboard import Key, KeyCode

from diffusion_policy_contrib.real_world.fanuc.fanuc_interpolation_controller import (
    FanucInterpolationController,
    FanucReceiveKeys,
)
from diffusion_policy_contrib.real_world.keystroke_counter import KeystrokeCounter
from diffusion_policy_contrib.shared_memory.shared_memory_util import SharedMemoryManager


@dataclass
class LiftQARealConfig:
    enable_robot_movement: bool = True
    constrain_motion: bool = False
    control_hz: int = 60
    scheduled_waypoint_time_offset: float = 1 / control_hz
    robot_ip_real: str = "192.168.10.123"
    init_robot_joints: np.ndarray = field(
        default_factory=lambda: np.array(np.deg2rad([0.0, 0.0, 0, 0, -90, 0]))# np.array(np.deg2rad([84.81, -68.54, 71.3, -92.77, -90, -5.17])) # np.array(np.deg2rad([90.0, -90.0, 90.0, -90.0, -90.0, 0.0]))
    )
    max_pos_speed: float = 0.2
    max_rot_speed: float = 0.2
    max_target_lead_m: Optional[float] = 0.03


@dataclass
class FanucManualTestConfig:
    episodes: int = 200
    max_episode_length: int = 1000
    env: LiftQARealConfig = field(default_factory=LiftQARealConfig)

    keypress_delta_x: float = 0.06
    keypress_delta_y: float = 0.06
    keypress_delta_z: float = 0.06
    max_command_lead: float = 0.06
    # "script" keeps a persistent keyboard target and clips that accumulator in
    # this process. "controller" sends unclipped targets and lets
    # FanucInterpolationController clip each request against CommandTCPPose.
    clipping_mode: Literal["script", "controller"] = "controller"
    # In controller mode, "return_to_command" recomputes the target from
    # CommandTCPPose every cycle. Releasing a key therefore removes one keypress
    # worth of offset. "hold_target" keeps the last target on release, matching
    # the accumulator behavior of script clipping.
    controller_release_behavior: Literal["return_to_command", "hold_target"] = "hold_target"



class LiftQARealEnvironment:
    def __init__(self, cfg: LiftQARealConfig):
        self.cfg = cfg
        self.shm_manager = SharedMemoryManager()
        self.shm_manager.__enter__()

        self.robot = FanucInterpolationController(
            shm_manager=self.shm_manager,
            robot_ip=cfg.robot_ip_real,
            stream_motion_data_style="cartesian",
            max_pos_speed=self.cfg.max_pos_speed,
            max_rot_speed=self.cfg.max_rot_speed,
            joints_init=self.cfg.init_robot_joints,
            verbose=True,
            max_target_lead_m=self.cfg.max_target_lead_m,
        )

        self.robot.start(wait=True)

        time.sleep(2.0)
        ref_state = self.robot.get_state().copy()
        self.robot_reference_tcp = ref_state[FanucReceiveKeys.CommandTCPPose]

    def reset(self) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        self.timestep = 0
        # WARMUP - Get obs a couple times to initialize camera - warmup
        self.current_robot_state = self.robot.get_state()
        self._get_obs()
        time.sleep(0.5)
        self.current_robot_state = self.robot.get_state()
        self._get_obs()
        obs = self._get_obs()
        info: Dict[str, Any] = {}
        return obs

    def _parse_action(self, action: np.ndarray) -> Tuple[float, float, float]:
        action = np.asarray(action, dtype=np.float32).reshape(-1)
        if action.shape[0] != 3:
            raise ValueError(f"Expected action with 3 values [x, y, z], got {action.shape[0]}")
        return float(action[0]), float(action[1]), float(action[2])

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

        target_x, target_y, target_z = self._parse_action(action)
        current_tcp = np.asarray(self.current_robot_state[FanucReceiveKeys.CommandTCPPose], dtype=np.float32)
        target_pose = current_tcp.copy()
        target_pose[0] = target_x
        target_pose[1] = target_y
        target_pose[2] = target_z
        target_pose = self._apply_motion_constraint(target_pose)

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

    def close(self):
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
        return obs


@draccus.wrap()
def main(cfg: FanucManualTestConfig):
    env_cfg = cfg.env
    env_cfg.max_target_lead_m = (
        cfg.max_command_lead if cfg.clipping_mode == "controller" else None
    )
    env = LiftQARealEnvironment(cfg=env_cfg)
    episode_data_to_save = []
    key_counter = None

    try:
        key_counter = KeystrokeCounter()
        key_counter.__enter__()

        for episode_id in range(cfg.episodes):
            print(f"Started episode {episode_id}")
            obs = env.reset()
            reward = 0.0
            done = False
            info = {}
            episode_data_to_save = []
            key_counter.clear()
            keyboard_target = np.asarray(obs["command_tcp"][:3], dtype=np.float32).copy()

            step_num = 0
            while True:
                x_dir = int(key_counter.is_pressed(Key.right)) - int(key_counter.is_pressed(Key.left))
                y_dir = int(key_counter.is_pressed(Key.up)) - int(key_counter.is_pressed(Key.down))
                z_dir = int(key_counter.is_pressed(Key.space)) - int(key_counter.is_pressed(KeyCode(char="c")))
                command_tcp = np.asarray(obs["command_tcp"][:3], dtype=np.float32)

                if cfg.clipping_mode == "script":
                    # Persistent accumulator: releasing the key adds zero, so
                    # the target stays where it was and only the script-side
                    # clip keeps it near CommandTCPPose.
                    keyboard_target[0] += cfg.keypress_delta_x * x_dir
                    keyboard_target[1] += cfg.keypress_delta_y * y_dir
                    keyboard_target[2] += cfg.keypress_delta_z * z_dir
                    keyboard_target = np.clip(
                        keyboard_target,
                        command_tcp - cfg.max_command_lead,
                        command_tcp + cfg.max_command_lead,
                    )
                else:
                    # Controller clipping test path. In return_to_command mode,
                    # no key means target == CommandTCPPose, which shows up as a
                    # one-keypress jump back on release. In hold_target mode, we
                    # only update axes with active keypresses.
                    if cfg.controller_release_behavior == "return_to_command":
                        keyboard_target = command_tcp.copy()
                    if x_dir:
                        keyboard_target[0] = command_tcp[0] + cfg.keypress_delta_x * x_dir
                    if y_dir:
                        keyboard_target[1] = command_tcp[1] + cfg.keypress_delta_y * y_dir
                    if z_dir:
                        keyboard_target[2] = command_tcp[2] + cfg.keypress_delta_z * z_dir

                action = keyboard_target.copy()
                obs, reward, done, info = env.step(action)

                if done or step_num > cfg.max_episode_length:
                    break

                step_num += 1

                episode_data_to_save.append({
                    "tgt_tcp": obs["target_tcp"],
                    "cmd_tcp": obs["command_tcp"],
                    "act_tcp": obs["actual_tcp"],
                    "target_keys": keyboard_target.copy()
                })

    except KeyboardInterrupt:
        print("User cancelled data collection")
        import matplotlib.pyplot as plt

        t = range(len(episode_data_to_save))
        cmd_x = [d["cmd_tcp"][0] for d in episode_data_to_save]
        cmd_y = [d["cmd_tcp"][1] for d in episode_data_to_save]
        cmd_z = [d["cmd_tcp"][2] for d in episode_data_to_save]

        act_x = [d["act_tcp"][0] for d in episode_data_to_save]
        act_y = [d["act_tcp"][1] for d in episode_data_to_save]
        act_z = [d["act_tcp"][2] for d in episode_data_to_save]

        tar_x = [d["tgt_tcp"][0] for d in episode_data_to_save]
        tar_y = [d["tgt_tcp"][1] for d in episode_data_to_save]
        tar_z = [d["tgt_tcp"][2] for d in episode_data_to_save]

        key_x = [d["target_keys"][0] for d in episode_data_to_save]
        key_y = [d["target_keys"][1] for d in episode_data_to_save]
        key_z = [d["target_keys"][2] for d in episode_data_to_save]

        fig, axs = plt.subplots(3, 1, figsize=(10, 8), sharex=True)

        # X
        axs[0].plot(t, cmd_x, label="Commanded")
        axs[0].plot(t, act_x, label="Actual")
        axs[0].plot(t, tar_x, label="Target")
        axs[0].plot(t, key_x, "--", label="Keyboard")
        axs[0].set_ylabel("X")
        axs[0].legend()
        axs[0].grid(True)

        # Y
        axs[1].plot(t, cmd_y, label="Commanded")
        axs[1].plot(t, act_y, label="Actual")
        axs[1].plot(t, tar_y, label="Target")
        axs[1].plot(t, key_y, "--", label="Keyboard")
        axs[1].set_ylabel("Y")
        axs[1].grid(True)

        # Z
        axs[2].plot(t, cmd_z, label="Commanded")
        axs[2].plot(t, act_z, label="Actual")
        axs[2].plot(t, tar_z, label="Target")
        axs[2].plot(t, key_z, "--", label="Keyboard")
        axs[2].set_ylabel("Z")
        axs[2].set_xlabel("Time (sample index)")
        axs[2].grid(True)

        plt.tight_layout()
        plt.show()

    finally:
        if key_counter is not None:
            key_counter.__exit__(None, None, None)
        env.close()


if __name__ == "__main__":
    main()
