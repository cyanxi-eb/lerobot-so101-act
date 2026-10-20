"""Launch file skeleton for the SO-101 policy node (not executed here)."""

from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description() -> LaunchDescription:
    return LaunchDescription(
        [
            Node(
                package="so101_policy_node",
                executable="policy_node",
                name="so101_policy_node",
                output="screen",
                parameters=[
                    {
                        "checkpoint": "checkpoints/act.pt",
                        "replan_interval": 5,
                        "execute_actions": True,
                        "front_topic": "/so101/front_camera/image_raw",
                        "wrist_topic": "/so101/wrist_camera/image_raw",
                        "state_topic": "/so101/joint_states",
                        "action_topic": "/so101/action",
                        "instruction_topic": "/so101/instruction",
                    }
                ],
            )
        ]
    )
