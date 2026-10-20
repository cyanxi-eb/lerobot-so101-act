"""Forward / inverse kinematics tests for the simplified SO-101 model."""

from __future__ import annotations

import math

import numpy as np
import pytest

from so101_sim import kinematics as kin


def test_fk_zero_pose_is_straight_up():
    info = kin.fk(np.zeros(5))
    expected_tip = kin.BASE_XYZ + np.array([0.0, 0.0,
                                            kin.SHOULDER_HEIGHT + kin.LINK_SHOULDER
                                            + kin.LINK_ELBOW + kin.LINK_TOOL])
    assert np.allclose(info["tip_pos"], expected_tip, atol=1e-9)
    assert np.allclose(info["tool_dir"], np.array([0.0, 0.0, 1.0]), atol=1e-9)
    assert info["tip_radius"] == pytest.approx(0.0, abs=1e-9)


def test_fk_shoulder_90_points_forward():
    q = np.array([0.0, math.pi / 2, 0.0, 0.0, 0.0])
    info = kin.fk(q)
    expected = kin.BASE_XYZ + np.array(
        [kin.LINK_SHOULDER + kin.LINK_ELBOW + kin.LINK_TOOL, 0.0, kin.SHOULDER_HEIGHT]
    )
    assert np.allclose(info["tip_pos"], expected, atol=1e-9)


def test_tool_points_straight_down_for_top_down_grasp():
    q = np.array([0.4, 0.7, 1.2, math.pi - 0.7 - 1.2, 0.0])
    info = kin.fk(q)
    assert np.allclose(info["tool_dir"], np.array([0.0, 0.0, -1.0]), atol=1e-9)
    assert info["tool_angle"] == pytest.approx(math.pi, abs=1e-9)


@pytest.mark.parametrize("target", [
    (-0.05, 0.00, 0.030),
    (0.02, 0.10, 0.040),
    (-0.06, -0.08, 0.025),
    (0.04, 0.00, 0.120),
    (-0.10, 0.03, 0.020),
])
def test_ik_roundtrip(target):
    q = kin.ik(np.array(target))
    assert kin.within_limits(q)
    info = kin.fk(q)
    assert np.allclose(info["tip_pos"], np.array(target), atol=1e-6)
    assert np.allclose(info["tool_dir"], np.array([0.0, 0.0, -1.0]), atol=1e-6)


def test_ik_raises_when_unreachable():
    with pytest.raises(ValueError):
        kin.ik(np.array([1.5, 0.0, 0.02]))


def test_home_pose_is_valid_and_above_table():
    q = kin.home_q()
    assert kin.within_limits(q)
    tip = kin.fk(q)["tip_pos"]
    assert tip[2] > 0.10
    assert kin.can_reach(tip)


def test_gripper_fingers_open_with_opening():
    q = kin.home_q()
    closed = kin.gripper_fingers(q, 0.0)
    opened = kin.gripper_fingers(q, 1.0)
    d_closed = np.linalg.norm(closed[0] - closed[1])
    d_open = np.linalg.norm(opened[0] - opened[1])
    assert 0.0 < d_closed < d_open
    # Both fingers are symmetric around the tool tip.
    tip = kin.fk(q)["tip_pos"]
    assert np.allclose((closed[0] + closed[1]) / 2, tip, atol=1e-9)


def test_qpos_roundtrip():
    q = kin.home_q()
    state = kin.qpos(q, 0.35)
    assert state.shape == (kin.STATE_DIM,)
    q2, opening = kin.state_from_qpos(state)
    assert np.allclose(q, q2)
    assert opening == pytest.approx(0.35)
