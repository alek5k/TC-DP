from dataclasses import dataclass, field
import numpy as np

@dataclass
class EnvSetting:
    env_name: str
    env_type: str
    img_dim: int
    resnet_type: str = "resnet18"
    agent_pose_dim: int = 2
    action_dim: int = 2
    normalization_mask: dict[str, np.ndarray] = field(default_factory=dict) # dict of key to boolean mask of shape (D,) for which dimensions to apply normalization to. If None, apply to all dimensions.
    normalization_stats_overrides: dict[str, dict] = field(default_factory=dict) # dict of key -> {'min': (D,), 'max': (D,), optional 'mask': (D,)}

class EnvType:
    PYGAME = 'pygame'
    ROBOSUITE = 'robosuite'
    REAL_WORLD = 'real_world'

class EnvName:
    MOVE_AROUND = "MoveAround"
    AVOID_OBSTACLE_ENV = "AvoidObstacleEnv"
    WAIT_AT_GOAL = "WaitAtGoal"
    LIFT_QA = "LiftQA"
    SUCTIONPICKING = "suctionpicking"
    LIFTQA_REAL = "LiftQA-Real"
    PIANO = "Piano"


def get_env_settings_from_env_name(name: EnvName):
    if name == EnvName.LIFT_QA:
        return EnvSetting(env_name=EnvName.LIFT_QA, env_type=EnvType.ROBOSUITE, img_dim=144, resnet_type='resnet18', agent_pose_dim=3, action_dim=3)
    elif name == EnvName.SUCTIONPICKING:
        return EnvSetting(env_name=EnvName.SUCTIONPICKING, env_type=EnvType.REAL_WORLD, img_dim=144, resnet_type='resnet18', agent_pose_dim=6 + 1, action_dim=6 + 1)
    elif name == EnvName.LIFTQA_REAL:
        return EnvSetting(env_name=EnvName.LIFTQA_REAL, env_type=EnvType.REAL_WORLD, img_dim=144, resnet_type='resnet18', agent_pose_dim=3 + 1, action_dim=3 + 1)
    elif name == EnvName.PIANO:
        return EnvSetting(env_name=EnvName.PIANO, env_type=EnvType.REAL_WORLD, img_dim=144, resnet_type='resnet18', agent_pose_dim=3, action_dim=3)
    elif name in {EnvName.MOVE_AROUND, EnvName.AVOID_OBSTACLE_ENV, EnvName.WAIT_AT_GOAL}:
        return EnvSetting(env_name=name, env_type=EnvType.PYGAME, resnet_type='resnet18', img_dim=144)
    else:
        raise ValueError(f'Unsupported environment/dataset name: {name}')

def get_env_settings_from_dataset_name(name: str):
    if 'lift_qa_real' in name:
        return get_env_settings_from_env_name(EnvName.LIFTQA_REAL)
    elif 'lift_qa' in name:
        return get_env_settings_from_env_name(EnvName.LIFT_QA)
    elif 'move_around' in name:
        return get_env_settings_from_env_name(EnvName.MOVE_AROUND)
    elif 'avoid_obstacle' in name:
        return get_env_settings_from_env_name(EnvName.AVOID_OBSTACLE_ENV)
    elif 'wait_at_goal' in name:
        return get_env_settings_from_env_name(EnvName.WAIT_AT_GOAL)
    elif 'SchmalzSuction' in name:
        return get_env_settings_from_env_name(EnvName.SUCTIONPICKING)
    elif 'piano' in name.lower():
        return get_env_settings_from_env_name(EnvName.PIANO)
    else:
        raise ValueError(f'Unsupported environment/dataset name: {name}')


def create_pygame_env(env_name: str, render_size: int | None = None):
    '''
    Create an environment instance based on the provided name and render size.
    Args:
        env_name (str): The name of the environment to create. Supported values are "MoveAround", "AvoidObstacleEnv", "WaitAtGoal".
        render_size (int | None): Optional render size to pass to the environment constructor. If None, the environment will use its default render size.
    Returns:
        An instance of the specified environment.
    '''
    # Late imports to allow for only having specific env dependencies installed when needed
    kwargs = {}
    if render_size is not None:
        kwargs["render_size"] = render_size

    if env_name == EnvName.MOVE_AROUND:
        from temporal_diffusion_policy.envs.pygame.movearound_env import MoveAround
        return MoveAround(**kwargs)
    elif env_name == EnvName.AVOID_OBSTACLE_ENV:
        from temporal_diffusion_policy.envs.pygame.avoid_obstacle_env import AvoidObstacleEnv
        return AvoidObstacleEnv(**kwargs)
    elif env_name == EnvName.WAIT_AT_GOAL:
        from temporal_diffusion_policy.envs.pygame.waitatgoal_env import WaitAtGoal
        return WaitAtGoal(**kwargs)
    else:
        raise ValueError(f'Unsupported environment name: {env_name}')
