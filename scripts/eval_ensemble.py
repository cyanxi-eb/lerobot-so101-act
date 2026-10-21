#!/usr/bin/env python
"""Evaluate an equal-weight ensemble of two ACT checkpoints.

The two policies predict action chunks independently; the executed chunk is
the per-step average of the two (over the shorter chunk length).  This is the
inference recipe used for the GPU re-training experiment in reports/gpu/.

Example
-------
python scripts/eval_ensemble.py \
  --checkpoint-a checkpoints/act_240_gpu.pt \
  --checkpoint-b checkpoints/act_240_aug.pt \
  --tag act_ensemble
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import eval_policy as ep  # noqa: E402
from so101_policy.inference import checkpoint_summary, load_checkpoint  # noqa: E402
from so101_policy.rollout import summarise_rollouts  # noqa: E402
from so101_sim import EnvConfig, GraspEnv  # noqa: E402


class EnsemblePolicy:
    """Average the action chunks of two loaded policies."""

    def __init__(self, policy_a, policy_b):
        self.policy_a = policy_a
        self.policy_b = policy_b

    def act(self, obs, instruction=None):
        chunk_a = self.policy_a.act(obs, instruction)
        chunk_b = self.policy_b.act(obs, instruction)
        n = min(len(chunk_a), len(chunk_b))
        return (0.5 * chunk_a[:n] + 0.5 * chunk_b[:n]).astype(np.float32)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-a", required=True)
    parser.add_argument("--checkpoint-b", required=True)
    parser.add_argument("--seen-trials", type=int, default=20)
    parser.add_argument("--unseen-trials", type=int, default=20)
    parser.add_argument("--replan", type=int, default=5)
    parser.add_argument("--max-steps", type=int, default=60)
    parser.add_argument("--out-dir", default=str(REPO_ROOT / "reports"))
    parser.add_argument("--tag", default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    a_path = Path(args.checkpoint_a)
    b_path = Path(args.checkpoint_b)
    for path in (a_path, b_path):
        if not path.exists():
            print(f"checkpoint not found: {path}")
            return 2

    policy_a = load_checkpoint(a_path)
    policy_b = load_checkpoint(b_path)
    ensemble = EnsemblePolicy(policy_a, policy_b)
    env = GraspEnv(EnvConfig(max_steps=args.max_steps, image_size=96))

    seen_seeds = [ep.SEEN_SEED_BASE + i for i in range(args.seen_trials)]
    unseen_seeds = [ep.UNSEEN_SEED_BASE + i for i in range(args.unseen_trials)]
    seen = ep.run_split(ensemble, env, seen_seeds, "seen", replan=args.replan,
                        max_steps=args.max_steps, progress_label="seen")
    unseen = ep.run_split(ensemble, env, unseen_seeds, "unseen", replan=args.replan,
                          max_steps=args.max_steps, progress_label="unseen")
    all_results = seen + unseen

    summary_a = checkpoint_summary(a_path)
    summary_b = checkpoint_summary(b_path)
    report = {
        "policy": "act_ensemble",
        "policy_type": "act",
        "checkpoint": {
            "path": f"{a_path.as_posix()} + {b_path.as_posix()}",
            "bytes": summary_a["bytes"] + summary_b["bytes"],
            "n_parameters": summary_a["n_parameters"] + summary_b["n_parameters"],
            "components": [
                {"path": a_path.as_posix(), "bytes": summary_a["bytes"],
                 "n_parameters": summary_a["n_parameters"]},
                {"path": b_path.as_posix(), "bytes": summary_b["bytes"],
                 "n_parameters": summary_b["n_parameters"]},
            ],
            "ensemble_rule": "equal-weight average of the two action chunks",
        },
        "protocol": {
            "simulator": "so101_sim (kinematic, 96x96 front + wrist cameras)",
            "control_hz": env.config.fps,
            "max_steps": args.max_steps,
            "replan_interval": args.replan,
            "seen_seeds": seen_seeds,
            "unseen_seeds": unseen_seeds,
            "success_definition": "instructed block inside the box after release",
        },
        "splits": {"seen": summarise_rollouts(seen), "unseen": summarise_rollouts(unseen)},
        "failure_mode_counts": dict(
            Counter(r["failure"] for r in all_results if not r["success"])),
        "failure_cases": [
            {"seed": r["seed"], "split": r["split"], "failure": r["failure"],
             "target_color": r["target_color"], "target_size": r["target_size"],
             "has_occluder": r["has_occluder"], "description": ep.describe_failure(r)}
            for r in all_results if not r["success"]
        ][:5],
        "images": {},
    }
    montage = Path(args.out_dir) / "eval_act_ensemble_failures.png"
    ep.save_failure_montage(all_results, montage)
    if montage.exists():
        report["images"]["failures"] = montage.relative_to(REPO_ROOT).as_posix()

    tag = args.tag or f"act_ensemble_{datetime.now().strftime('%Y%m%d')}"
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / f"eval_{tag}.json"
    md_path = out_dir / f"eval_{tag}.md"
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    md_path.write_text(ep.render_markdown(report), encoding="utf-8")
    print(f"report -> {json_path}\n         {md_path}")
    for name, summary in report["splits"].items():
        print(f"[{name}] success {summary['successes']}/{summary['trials']} "
              f"({summary['success_rate'] * 100:.0f}%) | "
              f"latency {summary['latency_ms_mean']} ms | "
              f"failures {summary['failures']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
