"""SO-101-style simulated manipulation stack.

This package is a *simulation only* re-implementation of an SO-101 pick and
place pipeline: kinematics, a table-top scene with two synthetic camera
streams, simplified grasp physics, and a scripted Cartesian expert used to
generate behaviour-cloning demonstrations.  It is not a hardware driver.
"""

from .env import EnvConfig, GraspEnv
from .expert import TOTAL_STEPS, plan_expert_actions, rollout_expert
from .kinematics import (
    ACTION_DIM,
    BASE_XYZ,
    JOINT_LIMITS,
    N_ARM_JOINTS,
    STATE_DIM,
    can_reach,
    fk,
    gripper_fingers,
    home_q,
    ik,
    qpos,
)
from .render import Camera, Renderer
from .scene import (
    ACT_TASK,
    COLOR_RGB,
    COLOR_ZH,
    INSTRUCTION_TEMPLATE,
    SEEN_COLORS,
    UNSEEN_COLORS,
    Block,
    Box,
    CameraSetup,
    Scene,
)

__all__ = [
    "ACT_TASK",
    "ACTION_DIM",
    "BASE_XYZ",
    "Block",
    "Box",
    "COLOR_RGB",
    "COLOR_ZH",
    "Camera",
    "CameraSetup",
    "EnvConfig",
    "GraspEnv",
    "INSTRUCTION_TEMPLATE",
    "JOINT_LIMITS",
    "N_ARM_JOINTS",
    "Renderer",
    "SEEN_COLORS",
    "STATE_DIM",
    "Scene",
    "TOTAL_STEPS",
    "UNSEEN_COLORS",
    "can_reach",
    "fk",
    "gripper_fingers",
    "home_q",
    "ik",
    "plan_expert_actions",
    "qpos",
    "rollout_expert",
]
