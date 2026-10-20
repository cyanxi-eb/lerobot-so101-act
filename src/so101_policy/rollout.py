"""Closed-loop policy rollouts used by evaluation, demos and training.

The policy is queried once every ``replan_interval`` steps (action chunking)
and its first actions are executed open-loop in between, which mirrors how
ACT is deployed on a real robot.  Every rollout also measures per-chunk
inference latency on the actual machine.
"""

from __future__ import annotations

import time

import numpy as np

from so101_sim import EnvConfig, GraspEnv
from so101_sim import kinematics as kin


def _percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    return float(np.percentile(np.asarray(values, dtype=np.float64), q))


def _box_occupants(env: GraspEnv) -> list[int]:
    """Indices of scene blocks currently inside the box footprint."""
    if env.scene is None:
        return []
    return [i for i, block in enumerate(env.scene.blocks)
            if env.scene.box.contains_xy(block.xy)]


def run_rollout(policy, env: GraspEnv, seed: int, split: str = "seen",
                n_blocks: int = 1, target_color: str | None = None,
                replan_interval: int = 5, max_steps: int | None = None,
                instruction: str | None = None, capture_frames: bool = False,
                capture_last: bool = False, ensemble_decay: float = 0.0) -> dict:
    """Run one closed-loop episode. ``policy`` must expose ``.act(obs)``.

    ``ensemble_decay > 0`` enables ACT-style temporal ensembling: every
    prediction for an absolute future step is stored with weight
    ``exp(-decay * age)`` and the executed action is the weighted average of
    all predictions that cover the current step.
    """
    obs = env.reset(seed=seed, split=split, n_blocks=n_blocks,
                    target_color=target_color)
    max_steps = int(max_steps or env.config.max_steps)
    wall_started = time.perf_counter()
    chunk: np.ndarray | None = None
    buffer: dict[int, list[tuple[float, np.ndarray]]] = {}
    latencies: list[float] = []
    frames: list[dict] = []
    steps = 0
    done = False
    info: dict = {}
    if capture_frames:
        frames.append({"front": obs["front"].copy(), "wrist": obs["wrist"].copy()})
    while steps < max_steps and not done:
        if chunk is None or steps % replan_interval == 0:
            started = time.perf_counter()
            chunk = policy.act(obs, instruction=instruction if instruction else obs["instruction"])
            latencies.append((time.perf_counter() - started) * 1000.0)
            if ensemble_decay > 0:
                for offset, predicted in enumerate(chunk):
                    weight = np.exp(-ensemble_decay * offset)
                    buffer.setdefault(steps + offset, []).append(
                        (float(weight), np.asarray(predicted, dtype=np.float64)))
        if ensemble_decay > 0 and buffer.get(steps):
            entries = buffer[steps]
            total = sum(weight for weight, _ in entries)
            action = sum(weight * value for weight, value in entries) / max(total, 1e-9)
        else:
            action = chunk[min(steps % replan_interval, chunk.shape[0] - 1)]
        obs, _, done, info = env.step(action)
        steps += 1
        if capture_frames:
            frames.append({"front": obs["front"].copy(), "wrist": obs["wrist"].copy()})
    success = bool(info.get("success"))
    wall_seconds = time.perf_counter() - wall_started
    occupants = _box_occupants(env)
    target_in_box = bool(env.scene is not None and env.scene.box.contains_xy(
        env.scene.target.xy))
    wrong_object_in_box = any(i != 0 for i in occupants)
    scene = env.scene
    released_xy = next((e["xy"] for e in reversed(info.get("events", []))
                        if e["event"] == "release"), None)
    release_error_m = None
    if released_xy is not None and scene is not None:
        release_error_m = round(float(np.linalg.norm(
            np.asarray(released_xy, dtype=np.float64) - scene.box.xy)), 4)
    # Language-conditioning signal that is independent of motor precision:
    # did the policy *grasp the instructed block* (index 0) or a distractor?
    grasp_events = [e for e in info.get("events", []) if e["event"] == "grasp"]
    grasped_block = int(grasp_events[0]["block_index"]) if grasp_events else None
    result = {
        "seed": int(seed),
        "split": split,
        "success": success,
        "failure": info.get("failure"),
        "steps": steps,
        "sim_seconds": round(steps * env.config.dt, 3),
        "wall_seconds": round(wall_seconds, 3),
        "latency_ms_mean": round(float(np.mean(latencies)), 3) if latencies else None,
        "latency_ms_p50": round(_percentile(latencies, 50), 3),
        "latency_ms_p95": round(_percentile(latencies, 95), 3),
        "replans": len(latencies),
        "target_in_box": target_in_box,
        "wrong_object_in_box": wrong_object_in_box,
        "instruction": None if scene is None else scene.instruction,
        "target_color": None if scene is None else scene.target.color,
        "target_xy": None if scene is None else scene.target.xy.round(4).tolist(),
        "target_size": None if scene is None else round(float(scene.target.size), 4),
        "box_xy": None if scene is None else scene.box.xy.round(4).tolist(),
        "has_occluder": None if scene is None else scene.tall_occluder is not None,
        "released_xy": released_xy,
        "release_error_m": release_error_m,
        "grasped_block": grasped_block,
        "grasp_grounding": None if grasped_block is None else grasped_block == 0,
        "events": info.get("events", []),
    }
    if capture_frames:
        result["frames"] = frames
    if capture_last:
        result["last_obs"] = {"front": obs["front"].copy(), "wrist": obs["wrist"].copy()}
    return result


def run_rollouts(policy, seeds, split: str = "seen", n_blocks: int = 1,
                 target_colors=None, replan_interval: int = 5,
                 max_steps: int | None = None, env: GraspEnv | None = None,
                 progress=None) -> list[dict]:
    """Evaluate a policy on a list of seeds and return per-trial records."""
    env = env or GraspEnv(EnvConfig())
    results = []
    for i, seed in enumerate(seeds):
        color = None if target_colors is None else target_colors[i % len(target_colors)]
        results.append(run_rollout(
            policy, env, seed=seed, split=split, n_blocks=n_blocks,
            target_color=color, replan_interval=replan_interval,
            max_steps=max_steps,
        ))
        if progress is not None:
            progress(i + 1, len(seeds), results[-1])
    return results


def summarise_rollouts(results: list[dict]) -> dict:
    """Aggregate per-trial records into the metrics reported in the README."""
    n = len(results)
    if n == 0:
        raise ValueError("no rollout results to summarise")
    successes = [bool(r["success"]) for r in results]
    failures: dict[str, int] = {}
    for r in results:
        if not r["success"]:
            key = r["failure"] or "unknown"
            failures[key] = failures.get(key, 0) + 1
    latencies = [r["latency_ms_mean"] for r in results if r["latency_ms_mean"] is not None]
    cycle = [r["sim_seconds"] for r in results]
    wall = [r["wall_seconds"] for r in results if "wall_seconds" in r]
    by_color: dict[str, dict[str, int]] = {}
    for r in results:
        color = r.get("target_color") or "unknown"
        entry = by_color.setdefault(color, {"trials": 0, "success": 0})
        entry["trials"] += 1
        entry["success"] += int(bool(r["success"]))
    grasping = [r for r in results if r.get("grasped_block") is not None]
    grounded = sum(1 for r in grasping if r["grasp_grounding"])
    return {
        "trials": n,
        "successes": int(sum(successes)),
        "success_rate": round(float(np.mean(successes)), 4),
        "failures": failures,
        "grasp_trials": len(grasping),
        "grounded_grasps": int(grounded),
        "grounding_accuracy": round(grounded / max(len(grasping), 1), 4) if grasping else None,
        "cycle_seconds_mean": round(float(np.mean(cycle)), 3),
        "cycle_seconds_min": round(float(np.min(cycle)), 3),
        "cycle_seconds_max": round(float(np.max(cycle)), 3),
        "wall_seconds_mean": round(float(np.mean(wall)), 3) if wall else None,
        "wall_seconds_min": round(float(np.min(wall)), 3) if wall else None,
        "wall_seconds_max": round(float(np.max(wall)), 3) if wall else None,
        "latency_ms_mean": round(float(np.mean(latencies)), 3) if latencies else None,
        "latency_ms_p50": round(_percentile(latencies, 50), 3),
        "latency_ms_p95": round(_percentile(latencies, 95), 3),
        "by_color": by_color,
        "wrong_object_in_box": int(sum(bool(r["wrong_object_in_box"]) for r in results)),
    }


def action_joint_limits_ok(actions: np.ndarray) -> bool:
    """True when every action joint value is inside the simulator limits."""
    actions = np.asarray(actions, dtype=np.float64)
    if actions.ndim == 1:
        actions = actions[None]
    joints = actions[:, :5]
    return bool(np.all(joints >= kin.JOINT_LIMITS[:, 0] - 1e-6)
                and np.all(joints <= kin.JOINT_LIMITS[:, 1] + 1e-6))
