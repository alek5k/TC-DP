from r2d2.controllers.oculus_controller import VRPolicy
from r2d2.robot_env import RobotEnv
from r2d2.user_interface.data_collector import DataCollecter
from r2d2.user_interface.gui_nocalib import RobotGUI
from r2d2.evaluation.policy_wrapper_marcel import PolicyWrapper

# Make the robot env
env = RobotEnv()
controller = VRPolicy()
# Make the data collector
path = ".ckpt"
policy = torch.load(path)

policy_timestep_filtering_kwargs = {
    "action_space": "cartesian_position"
    "robot_state_keys": ["cartesian_position", "gripper_position"],
    
}

wrapped_policy = PolicyWrapper(
    policy=policy,
    timestep_filtering_kwargs=policy_timestep_filtering_kwargs,
    image_transform_kwargs=policy_image_transform_kwargs,
    eval_mode=True,
)

# Launch GUI #
data_collector = DataCollecter(
    env=env,
    controller=controller,
    policy=wrapped_policy,
    save_traj_dir=log_dir,
    save_data=True,
)

# Make the GUI
user_interface = RobotGUI(robot=data_collector)
