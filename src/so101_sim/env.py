"""Table-top pick-and-place environment with simplified grasp physics.

The environment is a kinematic simulation: actions are absolute joint
position targets, joints move towards the target with a per-step rate limit
(the "servo tracking" model), and the grasp is decided by a small set of
geometric rules:

1. the gripper closes while the tool tip is inside ``grasp_tol`` of a block
   centre and at the block's centre height -> the block is attached;
2. an attached block follows the tool tip but can slip off with a small
   per-step probability (domain-randomised "gripper slip");
3. opening the gripper above the box releases the block, which then has to
   fall inside the box footprint.

Each episode is 60 control steps at 7.5 Hz, i.e. 8.0 s of simulated time,
which matches the "~8 s per pick-and-place cycle" figure reported by the
student project this repository re-implements.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from . import kinematics as kin
from .render import Renderer
from .scene import Scene


@dataclass
class EnvConfig:
    max_steps: int = 60
    #: Simulated duration of one pick-and-place cycle (seconds).
    cycle_seconds: float = 8.0
    max_joint_step: float = 0.25  # rad / control step
    max_gripper_step: float = 0.35
    image_size: int = 96

    @property
    def dt(self) -> float:
        return self.cycle_seconds / self.max_steps

    @property
    def fps(self) -> float:
        return self.max_steps / self.cycle_seconds


class GraspEnv:
    """A minimal Gym-style environment (no external Gym dependency)."""

    def __init__(self, config: EnvConfig | None = None, renderer: Renderer | None = None):
        self.config = config or EnvConfig()
        self.renderer = renderer or Renderer(self.config.image_size)
        self.scene: Scene | None = None
        self.q = kin.home_q()
        self.opening = 1.0
        self.step_count = 0
        self.carrying: int | None = None
        self._carry_offset = np.zeros(2)
        self._grasp_attempted = False
        self._events: list[dict] = []
        self._released_xy: np.ndarray | None = None
        self.success = False
        self.failure: str | None = None
        self.done = True
        self._rng = np.random.default_rng(0)

    # -- lifecycle --------------------------------------------------------
    def reset(self, seed: int = 0, split: str = "train", n_blocks: int = 1,
              target_color: str | None = None) -> dict:
        self.scene = Scene(seed=seed, split=split, n_blocks=n_blocks,
                           target_color=target_color)
        self._rng = np.random.default_rng((int(seed) * 7919 + 13) % (2**32))
        self.step_count = 0
        self.carrying = None
        self._carry_offset = np.zeros(2)
        self._grasp_attempted = False
        self._events = []
        self._released_xy = None
        self.success = False
        self.failure = None
        self.done = False

        # Randomised initial arm pose ("initial pose randomisation").
        noise = self._rng.normal(0.0, self.scene.start_pose_noise, size=5)
        home = kin.home_q()
        self.q = np.clip(home + noise, kin.JOINT_LIMITS[:, 0] + 0.05,
                         kin.JOINT_LIMITS[:, 1] - 0.05)
        self.opening = 1.0
        return self._obs()

    def step(self, action) -> tuple[dict, float, bool, dict]:
        if self.scene is None:
            raise RuntimeError("call reset() before step()")
        action = np.asarray(action, dtype=np.float64).reshape(-1)
        if action.shape[0] < kin.ACTION_DIM:
            raise ValueError(f"action must have {kin.ACTION_DIM} dims, got {action.shape}")

        target_q = np.clip(action[:5], kin.JOINT_LIMITS[:, 0], kin.JOINT_LIMITS[:, 1])
        target_g = float(np.clip(action[5], 0.0, 1.0))
        max_q = self.config.max_joint_step
        self.q = self.q + np.clip(target_q - self.q, -max_q, max_q)
        self.opening = self.opening + float(
            np.clip(target_g - self.opening, -self.config.max_gripper_step,
                    self.config.max_gripper_step)
        )
        self.q = np.clip(self.q, kin.JOINT_LIMITS[:, 0], kin.JOINT_LIMITS[:, 1])

        self._update_grasp()
        self.step_count += 1

        if not self.success and self._check_success():
            self.success = True
            self.done = True
            self._events.append({"step": self.step_count, "event": "success"})
        elif self.success:
            self.done = True
        elif self.step_count >= self.config.max_steps:
            self.done = True
            self.failure = self._classify_failure()

        reward = 1.0 if self.success else 0.0
        return self._obs(), reward, self.done, self.info()

    # -- physics ----------------------------------------------------------
    def _tip(self) -> np.ndarray:
        return kin.fk(self.q)["tip_pos"]

    def _update_grasp(self) -> None:
        assert self.scene is not None
        tip = self._tip()

        # --- grasp attempt on closing ------------------------------------
        if self.opening < 0.25 and self.carrying is None and not self._grasp_attempted:
            self._grasp_attempted = True
            best = None
            for idx, block in enumerate(self.scene.blocks):
                d_xy = float(np.linalg.norm(block.xy - tip[:2]))
                if d_xy < self.scene.grasp_tol and abs(tip[2] - block.center_z) < 0.030:
                    if best is None or d_xy < best[1]:
                        best = (idx, d_xy)
            if best is not None:
                self.carrying = best[0]
                self._carry_offset = self.scene.blocks[best[0]].xy - tip[:2]
                self._events.append({"step": self.step_count, "event": "grasp",
                                     "error": round(float(best[1]), 5),
                                     "block_index": int(best[0])})
            else:
                self._events.append({"step": self.step_count, "event": "grasp_miss",
                                     "tip": tip.round(4).tolist()})

        # --- release on opening ------------------------------------------
        if self.opening > 0.6:
            self._grasp_attempted = False
            if self.carrying is not None:
                block = self.scene.blocks[self.carrying]
                self._released_xy = tip[:2].copy()
                block.xy = tip[:2].copy()
                self._events.append({"step": self.step_count, "event": "release",
                                     "xy": tip[:2].round(4).tolist()})
                self.carrying = None

        # --- carry / slip -------------------------------------------------
        if self.carrying is not None:
            block = self.scene.blocks[self.carrying]
            block.xy = tip[:2] + self._carry_offset
            slip = float(np.clip(self.scene.slip_prob, 0.0, 1.0))
            # Extra slip risk when the gripper is not fully closed yet, i.e.
            # the block is held by a partial grasp.
            slip += 0.05 * max(0.0, self.opening)
            if self._rng.random() < slip:
                self._events.append({"step": self.step_count, "event": "slip",
                                     "xy": block.xy.round(4).tolist()})
                block.xy = block.xy.copy()
                self.carrying = None
                self._grasp_attempted = True

    def _check_success(self) -> bool:
        assert self.scene is not None
        if self.carrying is not None:
            return False
        target = self.scene.target
        return self.scene.box.contains_xy(target.xy)

    def _classify_failure(self) -> str:
        events = [e["event"] for e in self._events]
        if self.carrying is not None:
            return "timeout"
        if "slip" in events:
            return "slip"
        if "release" in events:
            return "misplace"
        if "grasp_miss" in events:
            return "grasp_miss"
        return "timeout"

    # -- observations -----------------------------------------------------
    def _obs(self) -> dict:
        views = self.renderer.render_obs(self.scene, self.q, self.opening)
        state = kin.qpos(self.q, self.opening)
        return {
            "front": views["front"],
            "wrist": views["wrist"],
            "state": state,
            "instruction": self.scene.instruction,
            "step": self.step_count,
        }

    def qpos(self) -> np.ndarray:
        return kin.qpos(self.q, self.opening)

    def info(self) -> dict:
        return {
            "success": self.success,
            "failure": self.failure,
            "step": self.step_count,
            "events": list(self._events),
            "instruction": None if self.scene is None else self.scene.instruction,
            "target_color": None if self.scene is None else self.scene.target.color,
            "carrying": self.carrying,
            "scene": None if self.scene is None else self.scene.meta(),
        }

    # -- misc -------------------------------------------------------------
    def expert_waypoints(self) -> dict:
        """Cartesian waypoints used by the scripted expert."""
        assert self.scene is not None
        target = self.scene.target
        box = self.scene.box
        home_xyz = kin.BASE_XYZ + np.array([0.16, 0.0, 0.12])
        return {
            "home": home_xyz,
            "approach": np.array([target.xy[0], target.xy[1], target.top_z + 0.055]),
            "grasp": target.center3d,
            "lift": np.array([target.xy[0], target.xy[1], 0.13]),
            "over_box": np.array([box.xy[0], box.xy[1], 0.13]),
            "place": np.array([box.xy[0], box.xy[1], 0.05]),
            "retreat": np.array([box.xy[0], box.xy[1], 0.13]),
        }
