import socket
import struct
import threading
import time
from collections import namedtuple

import numpy as np

try:
    import hid
except ModuleNotFoundError as exc:
    raise ImportError(
        "Unable to load module hid, required to interface with SpaceMouse. "
    ) from exc

import robosuite.macros as macros
from robosuite.devices import Device
from robosuite.utils.transform_utils import rotation_matrix

AxisSpec = namedtuple("AxisSpec", ["channel", "byte1", "byte2", "scale"])

SPACE_MOUSE_SPEC = {
    "x": AxisSpec(channel=1, byte1=1, byte2=2, scale=1),
    "y": AxisSpec(channel=1, byte1=3, byte2=4, scale=-1),
    "z": AxisSpec(channel=1, byte1=5, byte2=6, scale=-1),
    "roll": AxisSpec(channel=1, byte1=7, byte2=8, scale=-1),
    "pitch": AxisSpec(channel=1, byte1=9, byte2=10, scale=-1),
    "yaw": AxisSpec(channel=1, byte1=11, byte2=12, scale=1),
}


def to_int16(y1, y2):
    """
    Convert two 8 bit bytes to a signed 16 bit integer.

    Args:
        y1 (int): 8-bit byte
        y2 (int): 8-bit byte

    Returns:
        int: 16-bit integer
    """
    x = (y1) | (y2 << 8)
    if x >= 32768:
        x = -(65536 - x)
    return x


def scale_to_control(x, axis_scale=350.0, min_v=-1.0, max_v=1.0):
    """
    Normalize raw HID readings to target range.

    Args:
        x (int): Raw reading from HID
        axis_scale (float): (Inverted) scaling factor for mapping raw input value
        min_v (float): Minimum limit after scaling
        max_v (float): Maximum limit after scaling

    Returns:
        float: Clipped, scaled input from HID
    """
    x = x / axis_scale
    x = min(max(x, min_v), max_v)
    return x


def convert(b1, b2):
    """
    Converts SpaceMouse message to commands.

    Args:
        b1 (int): 8-bit byte
        b2 (int): 8-bit byte

    Returns:
        float: Scaled value from Spacemouse message
    """
    return scale_to_control(to_int16(b1, b2))


class SpaceMouse(Device):
    """
    A minimalistic driver class for SpaceMouse with HID library.

    Note: Use hid.enumerate() to view all USB human interface devices (HID).
    Make sure SpaceMouse is detected before running the script.

    Args:
        env (RobotEnv): The environment which contains the robot(s) to control
                        using this device.
        pos_sensitivity (float): Magnitude of input position command scaling
        rot_sensitivity (float): Magnitude of scale input rotation commands scaling
    """

    def __init__(
        self,
        env,
        spacemouse_model_type="old",
        x_sensitivity=1.0,
        y_sensitivity=1.0,
        z_sensitivity=1.0,
        roll_sensitivity=1.0,
        pitch_sensitivity=1.0,
        yaw_sensitivity=1.0,
        initial_motion_speed_scaling=1.0,
        server_address="192.168.112.1",
        server_port=26211,
        latch_grip=True,
        print_received_packets=False,
        use_keyboard_for_aux_controls=False
    ):
        super().__init__(env)

        print("Opening SpaceMouse device")
        
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.server_address = server_address
        self.server_port = server_port
        
        self.spacemouse_model_type = spacemouse_model_type

        assert 0 <= x_sensitivity <= 1.0
        assert 0 <= y_sensitivity <= 1.0
        assert 0 <= z_sensitivity <= 1.0
        assert 0 <= roll_sensitivity <= 1.0
        assert 0 <= pitch_sensitivity <= 1.0
        assert 0 <= yaw_sensitivity <= 1.0

        self.x_sensitivity = x_sensitivity
        self.y_sensitivity = y_sensitivity
        self.z_sensitivity = z_sensitivity
        self.roll_sensitivity = roll_sensitivity
        self.pitch_sensitivity = pitch_sensitivity
        self.yaw_sensitivity = yaw_sensitivity
        self.set_motion_speed_scaling(initial_motion_speed_scaling)

        self.latch_grip = latch_grip

        self.print_received_packets = print_received_packets

        # 6-DOF variables
        self.x, self.y, self.z = 0, 0, 0
        self.roll, self.pitch, self.yaw = 0, 0, 0

        self._display_controls()

        self.single_click_and_hold = False

        self._control = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        self._reset_state = 0
        self.rotation = np.array([[-1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, -1.0]])
        self._enabled = False

        # launch a new listener thread to listen to SpaceMouse
        self.thread = threading.Thread(target=self.run)
        self.thread.daemon = True
        self.thread.start()

        # also add a keyboard for aux controls
        if use_keyboard_for_aux_controls:
            from pynput.keyboard import Controller, Key, Listener
            self.listener = Listener(on_press=self.on_press, on_release=self.on_release)
            self.listener.start()
    
    @staticmethod
    def _display_controls():
        """
        Method to pretty print controls.
        """

        def print_command(char, info):
            char += " " * (30 - len(char))
            print("{}\t{}".format(char, info))

        print("")
        print_command("Control", "Command")
        print_command("Right button", "reset simulation")
        print_command("Left button (hold)", "close gripper")
        print_command("Move mouse laterally", "move arm horizontally in x-y plane")
        print_command("Move mouse vertically", "move arm vertically")
        print_command("Twist mouse about an axis", "rotate arm about a corresponding axis")
        print_command("Control+C", "quit")
        print_command("b", "toggle arm/base mode (if applicable)")
        print_command("s", "switch active arm (if multi-armed robot)")
        print_command("=", "switch active robot (if multi-robot environment)")
        print("")

    def _reset_internal_state(self):
        """
        Resets internal state of controller, except for the reset signal.
        """
        super()._reset_internal_state()

        self.rotation = np.array([[-1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, -1.0]])
        # Reset 6-DOF variables
        self.x, self.y, self.z = 0, 0, 0
        self.roll, self.pitch, self.yaw = 0, 0, 0
        # Reset control
        self._control = np.zeros(6)
        # Reset grasp
        self.single_click_and_hold = False

    def start_control(self):
        """
        Method that should be called externally before controller can
        start receiving commands.
        """
        self._reset_internal_state()
        self._reset_state = 0
        self._enabled = True


    def set_motion_speed_scaling(self, scaling: float):
        '''
        Sets the motion speed scaling (both position and rotation)
        '''
        self.motion_speed_scaling = scaling

    def get_controller_state(self):
        """
        Grabs the current state of the 3D mouse.

        Returns:
            dict: A dictionary containing dpos, orn, unmodified orn, grasp, and reset
        """
        dposx = self.control[0] * 0.005 * self.motion_speed_scaling * self.x_sensitivity
        dposy = self.control[1] * 0.005 * self.motion_speed_scaling * self.y_sensitivity
        dposz = self.control[2] * 0.005 * self.motion_speed_scaling * self.z_sensitivity
        roll = self.control[3] * 0.005 * self.motion_speed_scaling * self.roll_sensitivity
        pitch = self.control[4] * 0.005 * self.motion_speed_scaling * self.pitch_sensitivity
        yaw = self.control[5] * 0.005 * self.motion_speed_scaling * self.yaw_sensitivity

        # convert RPY to an absolute orientation
        drot1 = rotation_matrix(angle=-pitch, direction=[1.0, 0, 0], point=None)[:3, :3]
        drot2 = rotation_matrix(angle=roll, direction=[0, 1.0, 0], point=None)[:3, :3]
        drot3 = rotation_matrix(angle=yaw, direction=[0, 0, 1.0], point=None)[:3, :3]

        self.rotation = self.rotation.dot(drot1.dot(drot2.dot(drot3)))

        return dict(
            dpos=np.array([dposx, dposy, dposz]),
            rotation=self.rotation,
            raw_drotation=np.array([roll, pitch, yaw]),
            grasp=self.control_gripper,
            reset=self._reset_state,
            base_mode=int(self.base_mode),
        )

    def run(self):
        """Listener method that keeps pulling new messages."""

        t_last_click = -1
        
        def keepalive(sock: socket.socket, server_addr):
            """Background thread to send keepalive packets."""
            while True:
                try:
                    sock.sendto(b"ping", server_addr)
                except OSError:
                    break
                time.sleep(10)

        self.socket.sendto(b"hello", (self.server_address, self.server_port))
        threading.Thread(target=keepalive, args=(self.socket, (self.server_address, self.server_port)), daemon=True).start()
        # time.sleep(1)
        
        while True:
            data, addr = self.socket.recvfrom(1024)  # buffer size
            if len(data) < 13:
                continue
            d = struct.unpack('13B', data)
            if d is not None and self._enabled:
                if self.print_received_packets:
                    print(d)

                if self.spacemouse_model_type == "old":
                    ## logic for older spacemouse model

                    if d[0] == 1:  ## readings from 6-DoF sensor
                        self.y = convert(d[1], d[2])
                        self.x = convert(d[3], d[4])
                        self.z = convert(d[5], d[6]) * -1.0

                    elif d[0] == 2:

                        self.roll = convert(d[1], d[2])
                        self.pitch = convert(d[3], d[4])
                        self.yaw = convert(d[5], d[6])

                        self._control = [
                            self.x,
                            self.y,
                            self.z,
                            self.roll,
                            self.pitch,
                            self.yaw,
                        ]
                else:
                    ## default logic for all other spacemouse models

                    if d[0] == 1:  ## readings from 6-DoF sensor
                        self.y = convert(d[1], d[2])
                        self.x = convert(d[3], d[4])
                        self.z = convert(d[5], d[6]) * -1.0

                        self.roll = convert(d[7], d[8])
                        self.pitch = convert(d[9], d[10])
                        self.yaw = convert(d[11], d[12])

                        self._control = [
                            self.x,
                            self.y,
                            self.z,
                            self.roll,
                            self.pitch,
                            self.yaw,
                        ]

                if d[0] == 3:  ## readings from the side buttons
                    
                    # either button released, no buttons pressed
                    if d[1] == 0:
                        if not self.latch_grip and self.single_click_and_hold:
                            self.single_click_and_hold = False

                    # press left button
                    if d[1] == 1:
                        t_click = time.time()
                        elapsed_time = t_click - t_last_click
                        t_last_click = t_click
                        
                        if self.latch_grip:
                            self.single_click_and_hold = not self.single_click_and_hold
                        else:
                            self.single_click_and_hold = True

                    # right button is for reset
                    if d[1] == 2:
                        self._reset_state = 1
                        self._enabled = False
                        self._reset_internal_state()
                    

    @property
    def control(self):
        """
        Grabs current pose of Spacemouse

        Returns:
            np.array: 6-DoF control value
        """
        return np.array(self._control)

    @property
    def control_gripper(self):
        """
        Maps internal states into gripper commands.

        Returns:
            float: Whether we're using single click and hold or not
        """
        if self.single_click_and_hold:
            return 1.0
        return 0

    def on_press(self, key):
        """
        Key handler for key presses.
        Args:
            key (str): key that was pressed
        """
        pass

    def on_release(self, key):
        """
        Key handler for key releases.
        Args:
            key (str): key that was pressed
        """
        try:
            # controls for mobile base (only applicable if mobile base present)
            if key.char == "b":
                self.base_modes[self.active_robot] = not self.base_modes[self.active_robot]  # toggle mobile base
            elif key.char == "s":
                self.active_arm_index = (self.active_arm_index + 1) % len(self.all_robot_arms[self.active_robot])
            elif key.char == "=":
                self.active_robot = (self.active_robot + 1) % self.num_robots

        except AttributeError as e:
            pass

    def _postprocess_device_outputs(self, dpos, drotation):
        drotation = drotation * 50
        dpos = dpos * 125

        dpos = np.clip(dpos, -1, 1)
        drotation = np.clip(drotation, -1, 1)

        return dpos, drotation


if __name__ == "__main__":

    space_mouse = SpaceMouse()
    for i in range(100):
        print(space_mouse.control, space_mouse.control_gripper)
        time.sleep(0.02)
