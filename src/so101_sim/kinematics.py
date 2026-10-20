"""Simplified SO-101-style 5-DOF arm + 1-DOF gripper kinematics.

The real SO-101 is a 6-DOF (5 arm joints + gripper) low-cost arm with roughly
30 cm reach.  This module implements a deliberately simplified *planar*
version of that arm: base yaw + three pitch joints (shoulder, elbow, wrist)
in a vertical plane, plus a tool roll joint that does not change the tool
position.  The tool tip is always kept pointing straight down for table-top
pick-and-place, which makes analytic inverse kinematics exact and cheap.

Joint conventions
-----------------
q = [q1, q2, q3, q4, q5] (radians)

* q1 : base yaw, rotates the arm plane around the world +z axis.
* q2 : shoulder pitch, angle of link 2 measured from world +z toward the
       radial direction (sin / cos formulation, see ``fk``).
* q3 : elbow pitch, relative to link 2.
* q4 : wrist pitch, relative to link 3.
* q5 : tool roll, does not affect tip position (kept for a 5-DOF state).

The gripper opening is a scalar in [0, 1] where 1.0 = fully open and
0.0 = fully closed.  The full 6-D observation/action vector is
``[q1, q2, q3, q4, q5, gripper]``.
"""

from __future__ import annotations

import math

import numpy as np

# ---------------------------------------------------------------------------
# Geometry constants (metres).  Scaled down from a real SO-101 but with the
# same topology: a short shoulder offset, two long links and a short tool.
# ---------------------------------------------------------------------------
BASE_XYZ = np.array([-0.20, 0.0, 0.0], dtype=np.float64)
SHOULDER_HEIGHT = 0.06
LINK_SHOULDER = 0.15
LINK_ELBOW = 0.14
LINK_TOOL = 0.06

N_ARM_JOINTS = 5
STATE_DIM = 6
ACTION_DIM = 6

# Joint limits are wider than a real SO-101 (it is a simplified model), but
# they still keep the arm out of the table.
JOINT_LIMITS = np.array(
    [
        [-2.80, 2.80],   # q1 base yaw
        [-2.80, 2.80],   # q2 shoulder pitch
        [-2.60, 2.60],   # q3 elbow pitch
        [-3.20, 3.20],   # q4 wrist pitch
        [-3.1415, 3.1415],  # q5 tool roll
    ],
    dtype=np.float64,
)
GRIPPER_LIMITS = (0.0, 1.0)
MAX_REACH = LINK_SHOULDER + LINK_ELBOW + LINK_TOOL


def wrap_angle(angle: float) -> float:
    """Wrap an angle to (-pi, pi]."""
    return (float(angle) + math.pi) % (2.0 * math.pi) - math.pi


def fk(q) -> dict:
    """Forward kinematics.

    Returns a dict with world-space joint positions ``points`` (5 points:
    base, shoulder, elbow, wrist, tip), the tool direction ``tool_dir``,
    ``tip_pos``, the planar radial distance ``tip_radius`` and the absolute
    planar angle ``tool_angle``.
    """
    q = np.asarray(q, dtype=np.float64).reshape(-1)
    if q.shape[0] < N_ARM_JOINTS:
        raise ValueError(f"fk expects at least {N_ARM_JOINTS} joint values, got {q.shape}")
    q1, q2, q3, q4, q5 = q[:5]

    a2, a3, a4 = LINK_SHOULDER, LINK_ELBOW, LINK_TOOL
    r_s, z_s = 0.0, SHOULDER_HEIGHT
    r_e = r_s + a2 * math.sin(q2)
    z_e = z_s + a2 * math.cos(q2)
    a23 = q2 + q3
    r_w = r_e + a3 * math.sin(a23)
    z_w = z_e + a3 * math.cos(a23)
    a234 = a23 + q4
    r_t = r_w + a4 * math.sin(a234)
    z_t = z_w + a4 * math.cos(a234)

    cos_q1, sin_q1 = math.cos(q1), math.sin(q1)

    def to_world(r: float, z: float) -> np.ndarray:
        return BASE_XYZ + np.array([r * cos_q1, r * sin_q1, z], dtype=np.float64)

    points = [
        to_world(0.0, 0.0),      # base centre on the table
        to_world(r_s, z_s),      # shoulder
        to_world(r_e, z_e),      # elbow
        to_world(r_w, z_w),      # wrist
        to_world(r_t, z_t),      # tool tip (between the fingers)
    ]
    tool_dir = np.array(
        [math.sin(a234) * cos_q1, math.sin(a234) * sin_q1, math.cos(a234)],
        dtype=np.float64,
    )
    return {
        "points": points,
        "tip_pos": points[4],
        "tool_dir": tool_dir,
        "tool_angle": a234,
        "tip_radius": r_t,
        "joints": q[:5].copy(),
        "q5": q5,
    }


def gripper_fingers(q, opening: float) -> tuple[np.ndarray, np.ndarray]:
    """World positions of the two finger tips for a given opening."""
    info = fk(q)
    tip = info["tip_pos"]
    q1 = float(np.asarray(q).reshape(-1)[0])
    # Fingers open along the tangent (perpendicular to the radial direction).
    side = np.array([-math.sin(q1), math.cos(q1), 0.0])
    half = 0.006 + 0.013 * float(np.clip(opening, 0.0, 1.0))
    return tip + side * half, tip - side * half


def within_limits(q) -> bool:
    """True when all 5 arm joints are inside JOINT_LIMITS."""
    q = np.asarray(q, dtype=np.float64).reshape(-1)
    if q.shape[0] < N_ARM_JOINTS:
        return False
    return bool(np.all(q[:5] >= JOINT_LIMITS[:, 0]) and np.all(q[:5] <= JOINT_LIMITS[:, 1]))


def can_reach(target_xyz) -> bool:
    """Cheap reachability check for a downward tool tip at ``target_xyz``."""
    d = np.asarray(target_xyz, dtype=np.float64) - BASE_XYZ
    radius = math.hypot(d[0], d[1])
    wrist_z = float(d[2]) + LINK_TOOL
    dz = wrist_z - SHOULDER_HEIGHT
    dist = math.hypot(radius, dz)
    return abs(LINK_SHOULDER - LINK_ELBOW) + 1e-6 <= dist <= LINK_SHOULDER + LINK_ELBOW - 1e-6


def ik(target_xyz, q5: float = 0.0) -> np.ndarray:
    """Analytic inverse kinematics for a straight-down tool tip.

    Solves the 2-link planar chain for the wrist position
    (``target + LINK_TOOL * up``) and then aligns the tool with world -z.
    Raises ``ValueError`` when the target is unreachable or the solution
    violates the joint limits.
    """
    target = np.asarray(target_xyz, dtype=np.float64).reshape(-1)
    if target.shape[0] != 3:
        raise ValueError("ik expects a 3D target position")
    d = target - BASE_XYZ
    radius = math.hypot(d[0], d[1])
    q1 = math.atan2(d[1], d[0])

    wrist_z = target[2] + LINK_TOOL
    dz = wrist_z - SHOULDER_HEIGHT
    a2, a3 = LINK_SHOULDER, LINK_ELBOW
    dist2 = radius * radius + dz * dz
    cos_q3 = (dist2 - a2 * a2 - a3 * a3) / (2.0 * a2 * a3)
    if cos_q3 < -1.0 - 1e-9 or cos_q3 > 1.0 + 1e-9:
        raise ValueError(
            f"target {target.round(4).tolist()} is outside the workspace "
            f"(wrist distance {math.sqrt(dist2):.4f} m)"
        )
    cos_q3 = min(1.0, max(-1.0, cos_q3))
    base_angle = math.atan2(radius, dz)

    best: tuple[float, np.ndarray] | None = None
    for sign in (1.0, -1.0):
        q3 = sign * math.acos(cos_q3)
        q2 = base_angle - math.atan2(a3 * math.sin(q3), a2 + a3 * math.cos(q3))
        q4 = wrap_angle(math.pi - q2 - q3)
        q2 = wrap_angle(q2)
        cand = np.array([wrap_angle(q1), q2, q3, q4, float(q5)], dtype=np.float64)
        if not within_limits(cand):
            continue
        # Prefer the "elbow up" solution with the smallest joint motion.
        cost = abs(cand[1]) + abs(cand[2]) + 0.25 * abs(cand[3])
        if best is None or cost < best[0]:
            best = (cost, cand)
    if best is None:
        raise ValueError(
            f"target {target.round(4).tolist()} has a solution but violates joint limits"
        )
    return best[1]


def home_q() -> np.ndarray:
    """A neutral ready pose: tool 13 cm above the table, 16 cm in front."""
    target = BASE_XYZ + np.array([0.16, 0.0, 0.13])
    return ik(target)


def qpos(q, opening: float) -> np.ndarray:
    """Build the 6-D observation vector ``[5 arm joints, gripper]``."""
    q = np.asarray(q, dtype=np.float64).reshape(-1)
    return np.concatenate([q[:5], [float(np.clip(opening, 0.0, 1.0))]]).astype(np.float32)


def state_from_qpos(state) -> tuple[np.ndarray, float]:
    """Split a 6-D state vector into ``(arm_q, gripper_opening)``."""
    state = np.asarray(state, dtype=np.float64).reshape(-1)
    return state[:5].copy(), float(np.clip(state[5], 0.0, 1.0))
