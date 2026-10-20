"""Table-top scene: red/blue/green blocks, a sorting box, camera jitter.

A scene is fully determined by a seed and a split name, which is what makes
the simulated demonstrations reproducible.  Domain randomisation covers:

* block colour / size / position / yaw
* box position (always on the opposite side of the arm from the block)
* front camera extrinsics ("camera mounting error") and field of view
* wrist camera offset
* table / box colours and a global brightness jitter
* optional tall occluder block between the front camera and the target block
* gripper slip probability and grasp tolerance

The "unseen" split deliberately moves colour, size and position outside the
training distribution so evaluation can report a real generalisation gap.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from . import kinematics as kin

COLOR_RGB = {
    "red": (208, 58, 46),
    "blue": (46, 84, 200),
    "green": (56, 164, 76),
    "yellow": (226, 188, 44),
    "purple": (142, 66, 188),
}
COLOR_ZH = {
    "red": "红色",
    "blue": "蓝色",
    "green": "绿色",
    "yellow": "黄色",
    "purple": "紫色",
}
SEEN_COLORS = ("red", "blue", "green")
UNSEEN_COLORS = ("yellow", "purple")

BOX_RGB = (118, 132, 146)
TABLE_RGB = (206, 199, 186)

#: Uniform instruction template used by the language-conditioned dataset.
INSTRUCTION_TEMPLATE = "把{color}积木放进盒子里"
#: Task string used by the (unconditioned) ACT demonstrations.
ACT_TASK = "抓取积木并放进盒子"


@dataclass
class Block:
    """A graspable cuboid block resting on the table."""

    color: str
    size: float  # half extent (m)
    xy: np.ndarray
    yaw: float = 0.0
    rgb: tuple[int, int, int] = (200, 200, 200)

    @property
    def height(self) -> float:
        return 2.0 * self.size

    @property
    def center_z(self) -> float:
        return self.size

    @property
    def top_z(self) -> float:
        return 2.0 * self.size

    @property
    def center3d(self) -> np.ndarray:
        return np.array([self.xy[0], self.xy[1], self.size], dtype=np.float64)

    def contains_xy(self, xy, margin: float = 0.0) -> bool:
        d = np.asarray(xy, dtype=np.float64).reshape(-1) - self.xy
        c, s = math.cos(self.yaw), math.sin(self.yaw)
        local = np.array([c * d[0] + s * d[1], -s * d[0] + c * d[1]])
        return bool(np.all(np.abs(local) <= self.size + margin))

    def to_dict(self) -> dict:
        return {
            "color": self.color,
            "size": float(self.size),
            "xy": self.xy.round(5).tolist(),
            "yaw": float(self.yaw),
        }


@dataclass
class Box:
    """Open-top sorting box."""

    xy: np.ndarray
    inner_half: float = 0.045
    wall: float = 0.008
    wall_h: float = 0.026
    rgb: tuple[int, int, int] = BOX_RGB

    @property
    def outer_half(self) -> float:
        return self.inner_half + self.wall

    def contains_xy(self, xy) -> bool:
        d = np.asarray(xy, dtype=np.float64).reshape(-1) - self.xy
        return bool(np.all(np.abs(d) <= self.inner_half - 0.004))

    def to_dict(self) -> dict:
        return {
            "xy": self.xy.round(5).tolist(),
            "inner_half": float(self.inner_half),
            "wall_h": float(self.wall_h),
        }


@dataclass
class CameraSetup:
    """Front camera extrinsics with per-episode mounting error."""

    position: np.ndarray
    target: np.ndarray
    fov_deg: float = 70.0
    wrist_offset: np.ndarray = field(default_factory=lambda: np.zeros(3))

    def jittered(self, rng: np.random.Generator, pos_sigma=0.012, target_sigma=0.008,
                 fov_sigma=1.5, wrist_sigma=0.006) -> "CameraSetup":
        return CameraSetup(
            position=self.position + rng.normal(0.0, pos_sigma, size=3),
            target=self.target + rng.normal(0.0, target_sigma, size=3),
            fov_deg=float(self.fov_deg + rng.normal(0.0, fov_sigma)),
            wrist_offset=rng.normal(0.0, wrist_sigma, size=3),
        )

    def to_dict(self) -> dict:
        return {
            "front_position": self.position.round(5).tolist(),
            "front_target": self.target.round(5).tolist(),
            "front_fov_deg": round(float(self.fov_deg), 3),
            "wrist_offset": self.wrist_offset.round(5).tolist(),
        }


def default_camera() -> CameraSetup:
    return CameraSetup(
        position=np.array([-0.50, 0.0, 0.245]),
        target=np.array([-0.09, 0.0, 0.02]),
        fov_deg=72.0,
    )


class Scene:
    """One randomised table-top scene."""

    def __init__(
        self,
        seed: int = 0,
        split: str = "train",
        n_blocks: int = 1,
        target_color: str | None = None,
    ):
        if split not in ("train", "seen", "unseen", "lang"):
            raise ValueError(f"unknown split {split!r}")
        self.seed = int(seed)
        self.split = split
        self.n_blocks = max(1, int(n_blocks))
        # Separate RNG streams so block sampling and camera error do not shift
        # each other when we change the code.
        self.rng_scene = np.random.default_rng((self.seed * 2654435761) % (2**32))
        self.rng_aux = np.random.default_rng((self.seed * 40503 + 17) % (2**32))
        self.target_color = target_color
        self.tall_occluder: Block | None = None
        self._sample()
        for _ in range(300):
            if self._validate():
                break
            self._sample()
        else:  # pragma: no cover - extremely unlikely
            raise RuntimeError("could not sample a reachable scene")

    # -- sampling ---------------------------------------------------------
    def _sample(self) -> None:
        rng = self.rng_scene
        unseen = self.split == "unseen"
        palette = UNSEEN_COLORS if unseen else SEEN_COLORS

        if self.target_color is None:
            color = str(rng.choice(palette))
        else:
            color = self.target_color

        if unseen:
            radius = rng.uniform(0.238, 0.258)
            theta = rng.uniform(-0.85, 0.85)
            size = rng.uniform(0.023, 0.027)
        else:
            radius = rng.uniform(0.150, 0.232)
            theta = rng.uniform(-0.55, 0.55)
            size = rng.uniform(0.017, 0.023)

        target_xy = kin.BASE_XYZ[:2] + np.array(
            [radius * math.cos(theta), radius * math.sin(theta)]
        )
        self.target = Block(
            color=color,
            size=float(size),
            xy=target_xy,
            yaw=float(rng.uniform(-math.pi, math.pi)),
            rgb=COLOR_RGB[color],
        )

        self.blocks: list[Block] = [self.target]

        # Distractor blocks for the language-conditioned split.  Their colour
        # differs from the target so the instruction is unambiguous.
        if self.n_blocks > 1:
            others = [c for c in SEEN_COLORS if c != color]
            for _ in range(self.n_blocks - 1):
                d_color = str(rng.choice(others))
                for _try in range(20):
                    d_theta = rng.uniform(-0.8, 0.8)
                    d_radius = rng.uniform(0.16, 0.235)
                    d_xy = kin.BASE_XYZ[:2] + np.array(
                        [d_radius * math.cos(d_theta), d_radius * math.sin(d_theta)]
                    )
                    if np.linalg.norm(d_xy - self.target.xy) > 0.07:
                        break
                self.blocks.append(
                    Block(
                        color=d_color,
                        size=float(rng.uniform(0.017, 0.023)),
                        xy=d_xy,
                        yaw=float(rng.uniform(-math.pi, math.pi)),
                        rgb=COLOR_RGB[d_color],
                    )
                )

        # Box on the opposite side of the arm to force a transfer motion.
        box_sign = -1.0 if theta >= 0.0 else 1.0
        box_theta = box_sign * rng.uniform(0.78, 1.02)
        box_radius = rng.uniform(0.196, 0.232) if not unseen else rng.uniform(0.238, 0.252)
        box_xy = kin.BASE_XYZ[:2] + np.array(
            [box_radius * math.cos(box_theta), box_radius * math.sin(box_theta)]
        )
        self.box = Box(xy=box_xy)

        # Optional tall occluder between the front camera and the target.
        self.tall_occluder = None
        if not unseen and rng.random() < 0.35:
            cam = default_camera().position
            direction = self.target.xy - cam[:2]
            norm = float(np.linalg.norm(direction))
            if norm > 1e-6:
                direction = direction / norm
                occ_xy = self.target.xy - direction * 0.048
                # keep the occluder on the table and away from the box
                if np.linalg.norm(occ_xy - self.box.xy) > 0.06:
                    self.tall_occluder = Block(
                        color="occluder",
                        size=0.012,
                        xy=occ_xy,
                        yaw=0.0,
                        rgb=(120, 112, 104),
                    )

        # Camera mounting error and appearance jitter.
        self.camera = default_camera().jittered(self.rng_aux)
        self.brightness = float(rng.uniform(0.88, 1.12))
        self.table_rgb = tuple(
            int(np.clip(c * self.brightness, 0, 255)) for c in TABLE_RGB
        )
        self.box_rgb = tuple(int(np.clip(c * self.brightness, 0, 255)) for c in BOX_RGB)

        # Grasp / slip physics randomisation.
        self.grasp_tol = float(rng.uniform(0.010, 0.014))
        self.slip_prob = float(rng.uniform(0.0002, 0.0012)) if self.split != "unseen" else 0.0
        self.start_pose_noise = 0.10

        if self.split == "lang" or self.n_blocks > 1:
            self.instruction = INSTRUCTION_TEMPLATE.format(color=COLOR_ZH[color])
        else:
            self.instruction = INSTRUCTION_TEMPLATE.format(color=COLOR_ZH[color])
        self.task_act = ACT_TASK

    # -- validation -------------------------------------------------------
    def _validate(self) -> bool:
        for block in self.blocks:
            try:
                kin.ik(block.center3d)
            except ValueError:
                return False
            # Block must be reachable from above as well.
            if not kin.can_reach(block.center3d + np.array([0.0, 0.0, 0.10])):
                return False
        try:
            kin.ik(np.array([self.box.xy[0], self.box.xy[1], 0.05]))
        except ValueError:
            return False
        if np.linalg.norm(self.target.xy - self.box.xy) < 0.12:
            return False
        return True

    # -- helpers ----------------------------------------------------------
    @property
    def target_index(self) -> int:
        return 0

    def instruction_for(self, color: str | None = None) -> str:
        if color is None:
            return self.instruction
        return INSTRUCTION_TEMPLATE.format(color=COLOR_ZH[color])

    def meta(self) -> dict:
        return {
            "seed": self.seed,
            "split": self.split,
            "target_color": self.target.color,
            "target_size": round(float(self.target.size), 4),
            "target_xy": self.target.xy.round(4).tolist(),
            "box_xy": self.box.xy.round(4).tolist(),
            "has_occluder": self.tall_occluder is not None,
            "camera_jitter": self.camera.to_dict(),
            "brightness": round(self.brightness, 3),
            "grasp_tol": round(self.grasp_tol, 4),
            "slip_prob": round(self.slip_prob, 5),
            "instruction": self.instruction,
        }
