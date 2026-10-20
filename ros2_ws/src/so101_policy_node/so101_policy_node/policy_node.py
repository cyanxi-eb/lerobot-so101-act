"""ROS2 inference node skeleton for the trained SO-101 policy (unexecuted).

Environment: this package was written for ROS2 Jazzy but was **never built or
run** in this project, because the development machine has no ROS2 / no real
SO-101 hardware.  It is a deployment skeleton: the code is syntax-checked in
CI, the data flow is documented, and the actual topics / message types must be
adapted to the real robot driver.

Data flow
---------
    /so101/front_camera/image_raw   (sensor_msgs/Image)
    /so101/wrist_camera/image_raw   (sensor_msgs/Image)
    /so101/joint_states             (sensor_msgs/JointState, 5 arm joints + gripper)
              |
              v
    so101_policy.inference.load_checkpoint -> policy.act(obs)
              |
              v
    /so101/action                   (std_msgs/Float64MultiArray, 6 values)
              |
              v
    robot driver (servo target positions) -> SO-101 arm

Usage on a real robot (not possible here)::

    export PYTHONPATH=$PYTHONPATH:/path/to/lerobot-so101-act/src
    colcon build --packages-select so101_policy_node
    ros2 run so101_policy_node policy_node \
        --ros-args -p checkpoint:=/path/to/act.pt -p replan_interval:=5
"""

from __future__ import annotations

import time

import numpy as np

try:  # pragma: no cover - ROS2 is not installed on the development machine
    import rclpy
    from rclpy.node import Node
    from sensor_msgs.msg import Image, JointState
    from std_msgs.msg import Float64MultiArray, String

    _ROS2_AVAILABLE = True
except ImportError:  # pragma: no cover
    rclpy = None  # type: ignore[assignment]
    Node = object  # type: ignore[assignment,misc]
    Image = JointState = Float64MultiArray = String = object  # type: ignore[assignment,misc]
    _ROS2_AVAILABLE = False

from so101_policy.inference import load_checkpoint  # noqa: E402

ARM_JOINTS = ("q1_base_yaw", "q2_shoulder", "q3_elbow", "q4_wrist", "q5_roll")
GRIPPER_JOINT = "gripper"


def _ros2_missing() -> RuntimeError:
    return RuntimeError(
        "rclpy is not installed: this script must be run inside a ROS2 "
        "environment (ros2 run so101_policy_node policy_node). It is a "
        "deployment skeleton and has not been executed on this machine."
    )


def _image_to_numpy(msg) -> np.ndarray:
    """Convert a sensor_msgs/Image to ``(H, W, 3)`` uint8 (rgb8 / bgr8)."""
    encoding = getattr(msg, "encoding", "rgb8")
    height, width = int(msg.height), int(msg.width)
    buffer = np.frombuffer(bytes(msg.data), dtype=np.uint8)
    if encoding in ("rgb8", "bgr8"):
        channels = 3
    elif encoding in ("rgba8", "bgra8"):
        channels = 4
    elif encoding == "mono8":
        channels = 1
    else:
        raise ValueError(f"unsupported image encoding {encoding!r}")
    image = buffer.reshape(height, width, channels)[:, :, :3]
    if encoding.startswith("bgr"):
        image = image[:, :, ::-1]
    if channels == 1:
        image = np.repeat(image, 3, axis=2)
    return np.ascontiguousarray(image, dtype=np.uint8)


def _quat_to_matrix(x: float, y: float, z: float, w: float) -> np.ndarray:
    """Quaternion -> 3x3 rotation matrix (sensor_msgs/Quaternion convention)."""
    norm = np.sqrt(x * x + y * y + z * z + w * w) or 1.0
    x, y, z, w = x / norm, y / norm, z / norm, w / norm
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )


if _ROS2_AVAILABLE:  # pragma: no cover - requires ROS2

    class SO101PolicyNode(Node):
        """Closed-loop action-chunking inference node."""

        def __init__(self) -> None:
            super().__init__("so101_policy_node")
            self.declare_parameter("checkpoint", "checkpoints/act.pt")
            self.declare_parameter("replan_interval", 5)
            self.declare_parameter("execute_actions", True)
            self.declare_parameter("front_topic", "/so101/front_camera/image_raw")
            self.declare_parameter("wrist_topic", "/so101/wrist_camera/image_raw")
            self.declare_parameter("state_topic", "/so101/joint_states")
            self.declare_parameter("action_topic", "/so101/action")
            self.declare_parameter("instruction_topic", "/so101/instruction")

            checkpoint = self.get_parameter("checkpoint").value
            self.policy = load_checkpoint(checkpoint)
            self.get_logger().info(
                f"loaded {self.policy.policy_type} checkpoint {checkpoint} "
                f"({self.policy.n_parameters() / 1e6:.2f}M params)"
            )
            self.replan_interval = int(self.get_parameter("replan_interval").value)
            self.execute_actions = bool(self.get_parameter("execute_actions").value)

            self.front = None
            self.wrist = None
            self.state = None
            self.instruction = ""
            self.chunk = None
            self.step = 0
            self.inference_ms: list[float] = []

            self.create_subscription(Image, self.get_parameter("front_topic").value,
                                     self._on_front, 10)
            self.create_subscription(Image, self.get_parameter("wrist_topic").value,
                                     self._on_wrist, 10)
            self.create_subscription(JointState, self.get_parameter("state_topic").value,
                                     self._on_state, 10)
            self.create_subscription(String, self.get_parameter("instruction_topic").value,
                                     self._on_instruction, 10)
            self.action_pub = self.create_publisher(
                Float64MultiArray, self.get_parameter("action_topic").value, 10)
            # Control loop at 7.5 Hz, matching the demonstration FPS.
            self.timer = self.create_timer(1.0 / 7.5, self._on_timer)

        # -- subscriptions -------------------------------------------------
        def _on_front(self, msg) -> None:
            self.front = _image_to_numpy(msg)

        def _on_wrist(self, msg) -> None:
            self.wrist = _image_to_numpy(msg)

        def _on_state(self, msg: "JointState") -> None:
            by_name = dict(zip(msg.name, msg.position))
            arm = [float(by_name[name]) for name in ARM_JOINTS]
            gripper = float(by_name.get(GRIPPER_JOINT, 1.0))
            self.state = np.asarray(arm + [gripper], dtype=np.float32)

        def _on_instruction(self, msg: "String") -> None:
            self.instruction = str(msg.data)

        # -- control -------------------------------------------------------
        def _on_timer(self) -> None:
            if self.front is None or self.wrist is None or self.state is None:
                return
            if self.chunk is None or self.step % self.replan_interval == 0:
                obs = {
                    "front": self.front,
                    "wrist": self.wrist,
                    "state": self.state,
                    "instruction": self.instruction,
                }
                started = time.perf_counter()
                self.chunk = self.policy.act(obs, instruction=self.instruction)
                self.inference_ms.append((time.perf_counter() - started) * 1000.0)
                self.get_logger().debug(
                    f"chunk inference {self.inference_ms[-1]:.1f} ms "
                    f"(mean {np.mean(self.inference_ms):.1f} ms)"
                )
            action = self.chunk[self.step % self.replan_interval]
            self.step += 1
            if not self.execute_actions:
                return
            msg = Float64MultiArray()
            msg.data = [float(v) for v in action]
            self.action_pub.publish(msg)

    def main(args=None) -> int:
        rclpy.init(args=args)
        node = SO101PolicyNode()
        try:
            rclpy.spin(node)
        except KeyboardInterrupt:
            pass
        finally:
            node.destroy_node()
            rclpy.shutdown()
        return 0

else:  # pragma: no cover - the normal case on this machine

    class SO101PolicyNode:  # type: ignore[no-redef]
        """Placeholder that explains why the node cannot run here."""

        def __init__(self, *args, **kwargs) -> None:
            raise _ros2_missing()

    def main(args=None) -> int:
        raise SystemExit(_ros2_missing())


if __name__ == "__main__":
    raise SystemExit(main())
