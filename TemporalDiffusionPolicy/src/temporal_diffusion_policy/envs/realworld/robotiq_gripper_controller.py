import os
import time
import enum
import multiprocessing as mp
from multiprocessing.managers import SharedMemoryManager
from diffusion_policy_contrib.shared_memory.shared_memory_queue import (SharedMemoryQueue, Empty)
from diffusion_policy_contrib.shared_memory.shared_memory_ring_buffer import SharedMemoryRingBuffer
from diffusion_policy_contrib.common.precise_sleep import precise_wait
from diffusion_policy_contrib.common.pose_trajectory_interpolator import PoseTrajectoryInterpolator
from robotiq_gripper import RobotiqGripper

import numpy as np
class Command(enum.Enum):
    SHUTDOWN = 0
    SCHEDULE_WAYPOINT = 1
    RESTART_PUT = 2

class RobotiqGripperController(mp.Process):
    def __init__(self,
            shm_manager: SharedMemoryManager,
            hostname,
            port=1000,
            frequency=30,
            home_to_open=True,
            move_max_speed=64.0,
            get_max_k=None,
            command_queue_size=1024,
            launch_timeout=3,
            receive_latency=0.1,
            use_meters=False,
            verbose=False,
            gripper_force = 125,
            gripper_velocity = 64
            ):
        super().__init__(name="RobotiqGripperController")
        self.hostname = hostname
        self.port = port
        self.frequency = frequency
        self.home_to_open = home_to_open
        self.move_max_speed = move_max_speed
        self.launch_timeout = launch_timeout
        self.receive_latency = receive_latency
        self.scale = 1000.0 if use_meters else 1.0
        self.verbose = verbose

                #send gripper position
        self.gripper_force = gripper_force
        self.gripper_velocity = gripper_velocity

        rgripper = RobotiqGripper(
            robot_ip=self.hostname
        )
        self.max_gripper_position = rgripper.get_max_position_metres()
        self.min_gripper_position = rgripper.get_min_position_metres()

        if get_max_k is None:
            get_max_k = int(frequency * 10)

        # build input queue
        example = {
            'cmd': Command.SCHEDULE_WAYPOINT.value,
            'target_pos': 0.0,
            'target_time': 0.0
        }
        input_queue = SharedMemoryQueue.create_from_examples(
            shm_manager=shm_manager,
            examples=example,
            buffer_size=command_queue_size
        )

        # build ring buffer
        example = {
            'GripperPose': np.array([0.00]),
            'gripper_measure_timestamp': time.time(),
            'gripper_receive_timestamp': time.time(),
        }
        ring_buffer = SharedMemoryRingBuffer.create_from_examples(
            shm_manager=shm_manager,
            examples=example,
            get_max_k=get_max_k,
            get_time_budget=0.2,
            put_desired_frequency=frequency
        )

        self.ready_event = mp.Event()
        self.input_queue = input_queue
        self.ring_buffer = ring_buffer

    # ========= launch method ===========
    def start(self, wait=True):
        super().start()
        if wait:
            self.start_wait()
        if self.verbose:
            print(f"[WSGController] Controller process spawned at {self.pid}")

    def stop(self, wait=True):
        message = {
            'cmd': Command.SHUTDOWN.value
        }
        self.input_queue.put(message)
        if wait:
            self.stop_wait()

    def start_wait(self):
        self.ready_event.wait(self.launch_timeout)
        assert self.is_alive()

    def stop_wait(self):
        self.join()

    @property
    def is_ready(self):
        return self.ready_event.is_set()

    # ========= context manager ===========
    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()

    # ========= command methods ============
    def schedule_waypoint(self, pos: float, target_time: float):
        message = {
            'cmd': Command.SCHEDULE_WAYPOINT.value,
            'target_pos': pos,
            'target_time': target_time
        }
        self.input_queue.put(message)


    def restart_put(self, start_time):
        self.input_queue.put({
            'cmd': Command.RESTART_PUT.value,
            'target_time': start_time
        })

    # ========= receive APIs =============
    def get_state(self, k=None, out=None):
        if k is None:
            return self.ring_buffer.get(out=out)
        else:
            return self.ring_buffer.get_last_k(k=k,out=out)

    def get_all_state(self):
        data = self.ring_buffer.get_all()
        return data

    def open(self):
        message = {
            'cmd': Command.SCHEDULE_WAYPOINT.value,
            'target_pos': 255,
            'target_time': 0
        }
        self.input_queue.put(message)

    def close(self):
        message = {
            'cmd': Command.SCHEDULE_WAYPOINT.value,
            'target_pos': 0,
            'target_time': 0
        }
        self.input_queue.put(message)
    # ========= main loop in process ============
    def run(self):
        # start connection
        rgripper = RobotiqGripper(
            robot_ip=self.hostname
        )
        try:
            rgripper.connect()
            rgripper.activate()
            print("Activating gripper...",self.max_gripper_position, self.min_gripper_position)

            print(f"Pos: {str(rgripper.get_current_position_metres()): >3}  "
            f"Open: {rgripper.is_open(): <2}  "
            f"Closed: {rgripper.is_closed(): <2}  ")

            keep_running = True
            t_start = time.monotonic()
            iter_idx = 0


            # curr_pos = 100.0
            curr_pos = rgripper.get_current_position_metres()
            curr_t = time.monotonic()
            last_waypoint_time = curr_t
            pose_interp = PoseTrajectoryInterpolator(
                times=[curr_t],
                poses=[[curr_pos,0,0,0,0,0]]
            )

            while keep_running:
                # command gripper
                t_now = time.monotonic()
                dt = 1 / self.frequency

                t_target = t_now
                target_pos = pose_interp(t_target)[0]
                target_vel = (target_pos - pose_interp(t_target - dt)[0]) / dt

                # print('controller', int(target_pos), int(target_vel), int(gripper_force))
                info = rgripper.move_metres(
                    position=target_pos, speed=int(self.gripper_velocity),force=int(self.gripper_force))

                # info = rgripper.move_open_close(
                #     position=target_pos, speed=int(self.gripper_velocity),force=int(self.gripper_force))

                # get state from robot
                receive_time = time.time() - self.receive_latency
                # print(rgripper.get_current_position_metres(),time.time())
                state = {
                    'GripperPose': np.array([rgripper.get_current_position_metres()]),
                    # 'GripperPose': np.array([rgripper.get_current_position()]),
                    'gripper_receive_timestamp': receive_time,
                    'gripper_measure_timestamp': time.time(),
                }
                self.ring_buffer.put(state)


                # fetch command from queue
                try:
                    commands = self.input_queue.get_all()
                    n_cmd = len(commands['cmd'])
                except Empty:
                    n_cmd = 0

                # execute commands
                for i in range(n_cmd):
                    command = dict()
                    for key, value in commands.items():
                        command[key] = value[i]
                    cmd = command['cmd']

                    if cmd == Command.SHUTDOWN.value:
                        keep_running = False
                        # stop immediately, ignore later commands
                        break
                    elif cmd == Command.SCHEDULE_WAYPOINT.value:
                        target_pos = command['target_pos']
                        target_time = command['target_time']
                        # translate global time to monotonic time
                        target_time = time.monotonic() - time.time() + target_time
                        curr_time = t_now
                        pose_interp = pose_interp.schedule_waypoint(
                            pose=[target_pos, 0, 0, 0, 0, 0],
                            time=target_time,
                            max_pos_speed=self.move_max_speed,
                            max_rot_speed=self.move_max_speed,
                            curr_time=curr_time,
                            last_waypoint_time=last_waypoint_time
                        )
                        last_waypoint_time = target_time
                    elif cmd == Command.RESTART_PUT.value:
                        t_start = command['target_time'] - time.time() + time.monotonic()
                        iter_idx = 1
                    else:
                        keep_running = False
                        break
                # first loop successful, ready to receive command
                if iter_idx == 0:
                    self.ready_event.set()
                iter_idx += 1

                # regulate frequency
                dt = 1 / self.frequency
                t_end = t_start + dt * iter_idx
                precise_wait(t_end=t_end, time_func=time.monotonic)

        finally:
            rgripper.disconnect()
            self.ready_event.set()
            if self.verbose:
                print(f"[RobotiqGripperController] Disconnected from robot: {self.hostname}")