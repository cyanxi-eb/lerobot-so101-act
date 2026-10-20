"""Environment tests: rate limits, grasp/release logic, scripted expert."""

from __future__ import annotations

import numpy as np

from so101_sim import ACTION_DIM, EnvConfig, GraspEnv
from so101_sim import kinematics as kin
from so101_sim.expert import TOTAL_STEPS, plan_expert_actions, rollout_expert


def test_reset_returns_expected_observation_shapes():
    env = GraspEnv()
    obs = env.reset(seed=0)
    assert obs["front"].shape == (96, 96, 3)
    assert obs["wrist"].shape == (96, 96, 3)
    assert obs["front"].dtype == np.uint8
    assert obs["state"].shape == (6,)
    assert isinstance(obs["instruction"], str) and obs["instruction"]


def test_actions_are_rate_limited():
    env = GraspEnv(EnvConfig(max_joint_step=0.05))
    env.reset(seed=0)
    q_before = env.q.copy()
    action = np.concatenate([q_before + 1.0, [1.0]])
    env.step(action)
    assert np.max(np.abs(env.q - q_before)) <= 0.05 + 1e-9


def test_closing_gripper_in_the_air_is_a_grasp_miss():
    env = GraspEnv()
    env.reset(seed=1)
    for _ in range(4):
        _, _, _, info = env.step(np.concatenate([env.q, [0.0]]))
    events = [e["event"] for e in info["events"]]
    assert "grasp_miss" in events
    assert env.carrying is None


def test_scripted_expert_succeeds_and_actions_are_valid():
    env = GraspEnv()
    env.reset(seed=0)
    actions = plan_expert_actions(env)
    assert actions.shape == (TOTAL_STEPS, ACTION_DIM)
    joint_limits = kin.JOINT_LIMITS
    assert np.all(actions[:, :5] >= joint_limits[:, 0] - 1e-5)
    assert np.all(actions[:, :5] <= joint_limits[:, 1] + 1e-5)
    assert actions[0, 5] == 1.0 and actions[-1, 5] == 1.0
    assert actions[:, 5].min() == 0.0


def test_expert_rollout_produces_fixed_length_episode():
    env = GraspEnv()
    env.reset(seed=3)
    result = rollout_expert(env)
    assert result["success"]
    assert result["front"].shape == (60, 96, 96, 3)
    assert result["wrist"].shape == (60, 96, 96, 3)
    assert result["state"].shape == (60, 6)
    assert result["action"].shape == (60, 6)
    assert result["failure"] is None


def test_success_requires_block_inside_box():
    env = GraspEnv()
    env.reset(seed=0)
    assert not env._check_success()
    env.scene.target.xy = env.scene.box.xy.copy()
    assert env._check_success()
