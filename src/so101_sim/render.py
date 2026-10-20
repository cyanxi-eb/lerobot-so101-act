"""Tiny software renderer for the SO-101-style simulated table-top scene.

The renderer is a deliberately simple painter's-algorithm rasteriser: every
solid is a set of projected polygons drawn with Pillow.  Two cameras are
rendered:

* ``front``: a fixed camera looking at the table from the -x side, with a
  per-episode mounting error (position / aim / field of view jitter).
* ``wrist``: a camera rigidly attached above the tool tip, looking down,
  with a small mounting offset jitter.

Both produce 96x96 RGB uint8 images, which is small enough to train a policy
on a 2-core CPU but still contains colour, depth ordering and occlusion cues.

Static scenery (table background, grid, contact shadows) is rendered once per
episode and cached, and back faces are culled, which keeps a full 60-step
episode rollout in the low hundreds of milliseconds.
"""

from __future__ import annotations

import math

import numpy as np
from PIL import Image, ImageDraw

from . import kinematics as kin
from .scene import Block, Box, Scene

FACE_SHADE = {
    "top": 1.00,
    "bottom": 0.45,
    "px": 0.88,
    "nx": 0.66,
    "py": 0.94,
    "ny": 0.74,
}

# Face definition: vertex indices into the local cuboid corners and the
# outward normal in the local (yaw-rotated) frame.
_FACE_DEFS = (
    ("top", (4, 5, 6, 7), (0.0, 0.0, 1.0)),
    ("bottom", (0, 1, 2, 3), (0.0, 0.0, -1.0)),
    ("ny", (0, 1, 5, 4), (0.0, -1.0, 0.0)),
    ("px", (1, 2, 6, 5), (1.0, 0.0, 0.0)),
    ("py", (2, 3, 7, 6), (0.0, 1.0, 0.0)),
    ("nx", (3, 0, 4, 7), (-1.0, 0.0, 0.0)),
)


def _shade(rgb, factor: float, brightness: float = 1.0) -> tuple[int, int, int]:
    out = []
    for channel in rgb:
        value = int(channel * factor * brightness)
        out.append(0 if value < 0 else (255 if value > 255 else value))
    return out[0], out[1], out[2]


class Camera:
    """Pinhole camera with look-at extrinsics."""

    def __init__(self, position, target, up=(0.0, 0.0, 1.0), fov_deg: float = 70.0,
                 size: int = 96):
        self.position = np.asarray(position, dtype=np.float64)
        target = np.asarray(target, dtype=np.float64)
        forward = target - self.position
        forward = forward / (np.linalg.norm(forward) + 1e-9)
        up = np.asarray(up, dtype=np.float64)
        right = np.cross(forward, up)
        if np.linalg.norm(right) < 1e-6:
            right = np.cross(forward, np.array([0.0, 1.0, 0.0]))
        right = right / (np.linalg.norm(right) + 1e-9)
        true_up = np.cross(right, forward)
        self.forward, self.right, self.up = forward, right, true_up
        self.size = int(size)
        self.focal = 0.5 * self.size / math.tan(math.radians(fov_deg) / 2.0)
        self.cx = self.cy = (self.size - 1) / 2.0

    def project(self, pts) -> tuple[np.ndarray, np.ndarray]:
        rel = np.asarray(pts, dtype=np.float64).reshape(-1, 3) - self.position
        x = rel @ self.right
        y = rel @ self.up
        z = rel @ self.forward
        zz = np.where(np.abs(z) < 1e-6, 1e-6, z)
        u = self.cx + self.focal * x / zz
        v = self.cy - self.focal * y / zz
        return np.stack([u, v], axis=-1), z

    def project_quad(self, quad) -> tuple[list[tuple[float, float]], float] | None:
        """Project a 4-point world polygon; ``None`` when it is behind the camera."""
        uv, z = self.project(quad)
        if float(np.min(z)) < 0.015:
            return None
        return [(float(u), float(v)) for u, v in uv], float(np.mean(z))


class Renderer:
    """Renders 96x96 front / wrist RGB observations."""

    def __init__(self, size: int = 96):
        self.size = int(size)

    # -- cameras ----------------------------------------------------------
    def front_camera(self, scene: Scene) -> Camera:
        cam = scene.camera
        return Camera(cam.position, cam.target, up=(0.0, 0.0, 1.0),
                      fov_deg=cam.fov_deg, size=self.size)

    def wrist_camera(self, scene: Scene, tip_pos) -> Camera:
        offset = scene.camera.wrist_offset
        pos = np.asarray(tip_pos, dtype=np.float64) + offset + np.array([0.0, 0.0, 0.155])
        target = np.asarray(tip_pos, dtype=np.float64) + 0.25 * offset
        return Camera(pos, target, up=(0.0, 1.0, 0.0), fov_deg=64.0, size=self.size)

    # -- public API -------------------------------------------------------
    def prepare(self, scene: Scene) -> None:
        """Render and cache scenery that does not move during an episode."""
        if getattr(scene, "bg_front", None) is None:
            camera = self.front_camera(scene)
            scene.bg_front = self._background(scene, camera, with_grid=True)

    def render_obs(self, scene: Scene, q, opening: float) -> dict:
        info = kin.fk(q)
        self.prepare(scene)
        front = self.render_front(scene, q, opening, info["tip_pos"])
        wrist = self.render_wrist(scene, q, opening, info["tip_pos"])
        return {"front": front, "wrist": wrist}

    def render_front(self, scene: Scene, q, opening: float, tip_pos=None) -> np.ndarray:
        camera = self.front_camera(scene)
        base = scene.bg_front.copy()
        self._draw_box_fast(base, camera, scene)
        self._draw_blocks_fast(base, camera, scene)
        self._draw_arm(base, camera, q, opening, mode="front")
        return np.asarray(base, dtype=np.uint8)

    def render_wrist(self, scene: Scene, q, opening: float, tip_pos=None) -> np.ndarray:
        info = kin.fk(q)
        camera = self.wrist_camera(scene, info["tip_pos"])
        img = Image.new("RGB", (self.size, self.size), _shade(scene.table_rgb, 1.0))
        self._draw_box_fast(img, camera, scene)
        self._draw_blocks_fast(img, camera, scene)
        self._draw_arm(img, camera, q, opening, mode="wrist")
        return np.asarray(img, dtype=np.uint8)

    # -- scenery ----------------------------------------------------------
    def _background(self, scene: Scene, camera: Camera, with_grid: bool = True) -> Image.Image:
        img = Image.new("RGB", (self.size, self.size), _shade(scene.table_rgb, 1.0))
        draw = ImageDraw.Draw(img)
        if with_grid:
            line_rgb = _shade(scene.table_rgb, 0.90)
            lo, hi, step = -0.46, 0.36, 0.075
            val = lo
            while val <= hi + 1e-9:
                for seg in (
                    np.array([[lo, val, 0.001], [hi, val, 0.001]]),
                    np.array([[val, lo, 0.001], [val, hi, 0.001]]),
                ):
                    uv, z = camera.project(seg)
                    if float(np.min(z)) < 0.02:
                        continue
                    draw.line([(float(uv[0][0]), float(uv[0][1])),
                               (float(uv[1][0]), float(uv[1][1]))],
                              fill=line_rgb, width=1)
                val += step
        shadow_rgb = _shade(scene.table_rgb, 0.80)
        for block in list(scene.blocks) + ([scene.tall_occluder] if scene.tall_occluder else []):
            hx = hy = block.size * 1.15
            c, s = math.cos(block.yaw), math.sin(block.yaw)
            local = np.array([[-hx, -hy], [hx, -hy], [hx, hy], [-hx, hy]], dtype=np.float64)
            xy = local @ np.array([[c, s], [-s, c]]) + block.xy + np.array([0.008, 0.006])
            quad = np.column_stack([xy, np.full(4, 0.0015)])
            projected = camera.project_quad(quad)
            if projected:
                draw.polygon(projected[0], fill=shadow_rgb)
        if scene.tall_occluder is not None:
            # The occluder casts a longer shadow.
            block = scene.tall_occluder
            quad = np.array(
                [
                    [block.xy[0] - block.size * 2.2, block.xy[1] - block.size * 1.5, 0.0015],
                    [block.xy[0] + block.size * 2.2, block.xy[1] - block.size * 1.5, 0.0015],
                    [block.xy[0] + block.size * 2.2, block.xy[1] + block.size * 0.5, 0.0015],
                    [block.xy[0] - block.size * 2.2, block.xy[1] + block.size * 0.5, 0.0015],
                ]
            )
            projected = camera.project_quad(quad)
            if projected:
                draw.polygon(projected[0], fill=shadow_rgb)
        return img

    # -- geometry drawing -------------------------------------------------
    @staticmethod
    def _visible_faces(center, half, yaw: float, camera: Camera):
        c, s = math.cos(yaw), math.sin(yaw)
        rot = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
        hx, hy, hz = half
        local = np.array(
            [
                [-hx, -hy, -hz], [hx, -hy, -hz], [hx, hy, -hz], [-hx, hy, -hz],
                [-hx, -hy, hz], [hx, -hy, hz], [hx, hy, hz], [-hx, hy, hz],
            ],
            dtype=np.float64,
        )
        world = local @ rot.T + np.asarray(center, dtype=np.float64)
        view = camera.position - np.asarray(center, dtype=np.float64)
        faces = []
        for name, indices, local_normal in _FACE_DEFS:
            normal = rot @ np.asarray(local_normal)
            if float(np.dot(normal, view)) <= 0.0:
                continue
            faces.append((name, world[list(indices)]))
        return faces

    def _draw_faces(self, draw: ImageDraw.ImageDraw, camera: Camera,
                    faces: list[tuple[str, np.ndarray]], rgb, brightness: float) -> None:
        ops = []
        for name, quad in faces:
            projected = camera.project_quad(quad)
            if projected is None:
                continue
            polys, depth = projected
            ops.append((depth, polys, _shade(rgb, FACE_SHADE[name], brightness)))
        # Far to near so nearer faces overdraw farther ones.
        ops.sort(key=lambda item: -item[0])
        for _, poly, color in ops:
            draw.polygon(poly, fill=color)

    def _draw_box_fast(self, img: Image.Image, camera: Camera, scene: Scene) -> None:
        draw = ImageDraw.Draw(img)
        box: Box = scene.box
        # Visible inner floor.
        h = box.inner_half
        floor = np.array(
            [[box.xy[0] - h, box.xy[1] - h, 0.002], [box.xy[0] + h, box.xy[1] - h, 0.002],
             [box.xy[0] + h, box.xy[1] + h, 0.002], [box.xy[0] - h, box.xy[1] + h, 0.002]]
        )
        projected = camera.project_quad(floor)
        if projected is not None:
            draw.polygon(projected[0], fill=_shade(scene.box_rgb, 0.58))
        oh, wall, wh = box.outer_half, box.wall, box.wall_h
        walls = (
            ((box.xy[0], box.xy[1] - oh + wall / 2), (oh, wall / 2)),
            ((box.xy[0], box.xy[1] + oh - wall / 2), (oh, wall / 2)),
            ((box.xy[0] - oh + wall / 2, box.xy[1]), (wall / 2, oh)),
            ((box.xy[0] + oh - wall / 2, box.xy[1]), (wall / 2, oh)),
        )
        for (cx, cy), (hx, hy) in walls:
            center = np.array([cx, cy, wh / 2])
            faces = self._visible_faces(center, (hx, hy, wh / 2), 0.0, camera)
            self._draw_faces(draw, camera, faces, scene.box_rgb, scene.brightness)

    def _draw_blocks_fast(self, img: Image.Image, camera: Camera, scene: Scene) -> None:
        draw = ImageDraw.Draw(img)
        blocks: list[Block] = list(scene.blocks)
        if scene.tall_occluder is not None:
            blocks.append(scene.tall_occluder)
        # Draw far objects first so near ones overdraw them.
        def depth_key(block):
            return -float(np.dot(block.xy - camera.position[:2],
                                 camera.forward[:2]))

        for block in sorted(blocks, key=depth_key, reverse=True):
            if block.color == "occluder":
                center = np.array([block.xy[0], block.xy[1], 0.055])
                half = (block.size, block.size, 0.055)
            else:
                center = block.center3d
                half = (block.size, block.size, block.size)
            faces = self._visible_faces(center, half, block.yaw, camera)
            self._draw_faces(draw, camera, faces, block.rgb, scene.brightness)

    # -- robot arm --------------------------------------------------------
    def _draw_arm(self, img: Image.Image, camera: Camera, q, opening: float,
                  mode: str = "front") -> None:
        info = kin.fk(q)
        overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
        od = ImageDraw.Draw(overlay)
        if mode == "front":
            body = (78, 88, 106, 255)
            joint = (52, 60, 74, 255)
            uv, z = camera.project(np.asarray(info["points"]))
            if float(np.min(z)) > 0.02:
                widths = [0, 9, 7, 5, 4]
                for i in range(1, 5):
                    od.line([tuple(uv[i - 1]), tuple(uv[i])], fill=body, width=widths[i])
                for i in range(1, 5):
                    r = 5 - i * 0.4
                    od.ellipse([uv[i][0] - r, uv[i][1] - r, uv[i][0] + r, uv[i][1] + r],
                               fill=joint)
            f1, f2 = kin.gripper_fingers(q, opening)
            fuv, fz = camera.project(np.stack([f1, f2, info["tip_pos"]]))
            if float(np.min(fz)) > 0.02:
                od.line([tuple(fuv[2]), tuple(fuv[0])], fill=joint, width=2)
                od.line([tuple(fuv[2]), tuple(fuv[1])], fill=joint, width=2)
        else:
            f1, f2 = kin.gripper_fingers(q, opening)
            col = (60, 70, 86, 205)
            fuv, fz = camera.project(np.stack([f1, f2, info["tip_pos"]]))
            if float(np.min(fz)) > 0.02:
                od.line([tuple(fuv[2]), tuple(fuv[0])], fill=col, width=3)
                od.line([tuple(fuv[2]), tuple(fuv[1])], fill=col, width=3)
                od.ellipse([fuv[2][0] - 2.5, fuv[2][1] - 2.5, fuv[2][0] + 2.5,
                            fuv[2][1] + 2.5], fill=col)
        img.paste(Image.alpha_composite(img.convert("RGBA"), overlay).convert("RGB"))
