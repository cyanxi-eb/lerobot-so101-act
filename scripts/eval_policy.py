#!/usr/bin/env python
"""Evaluate a trained policy in the simulator and write JSON + Markdown reports.

Evaluation protocol
-------------------
* ``seen``   : 20 scenes from the *training* distribution (same colours, size
  range, position range, camera error) but unseen seeds.
* ``unseen`` : 20 scenes outside the training distribution - yellow / purple
  blocks (never seen in training), larger blocks and positions further from
  the arm.
* ``lang``   : 20 two-block scenes where a Chinese instruction selects which
  block to place; only the language-conditioned policy is expected to pass.

Success means "the instructed block ends up inside the box" (grasp ->
lift -> transfer -> release), which is exactly what the demonstrations do.
The script also measures per-chunk inference latency on this machine and the
simulated cycle time in seconds.

Examples
--------
    python scripts/eval_policy.py --policy act  --checkpoint checkpoints/act.pt
    python scripts/eval_policy.py --policy lang --checkpoint checkpoints/lang_cond.pt
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

import numpy as np
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from so101_policy.inference import checkpoint_summary, load_checkpoint  # noqa: E402
from so101_policy.rollout import run_rollout, summarise_rollouts  # noqa: E402
from so101_sim import SEEN_COLORS, EnvConfig, GraspEnv  # noqa: E402

SEEN_SEED_BASE = 3000
UNSEEN_SEED_BASE = 4100
LANG_SEED_BASE = 3200


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", choices=["act", "lang"], required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--seen-trials", type=int, default=20)
    parser.add_argument("--unseen-trials", type=int, default=20)
    parser.add_argument("--lang-trials", type=int, default=20)
    parser.add_argument("--replan", type=int, default=5,
                        help="execute this many actions from each chunk")
    parser.add_argument("--ensemble-decay", type=float, default=0.0,
                        help="temporal-ensembling decay (0 = plain chunk execution)")
    parser.add_argument("--max-steps", type=int, default=60)
    parser.add_argument("--out-dir", default=str(REPO_ROOT / "reports"))
    parser.add_argument("--tag", default=None)
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def describe_failure(result: dict) -> str:
    pieces = [
        f"seed {result['seed']}",
        f"failure={result['failure']}",
        f"target={result['target_color']} (half-size {result['target_size']} m)",
        f"occluder={result['has_occluder']}",
        f"steps={result['steps']}",
    ]
    if result.get("release_error_m") is not None:
        pieces.append(f"release point {result['release_error_m'] * 100:.1f} cm from box centre")
    if result.get("wrong_object_in_box"):
        pieces.append("a distractor block ended up in the box")
    return ", ".join(pieces)


def _rel(path) -> str:
    """Return a repo-relative path when possible (reports must not leak the host path)."""
    path = Path(path)
    try:
        return str(path.resolve().relative_to(REPO_ROOT))
    except ValueError:
        return path.name


def save_failure_montage(results: list[dict], path: Path) -> None:
    failures = [r for r in results if not r["success"] and "last_obs" in r][:3]
    if not failures:
        return
    scale = 128
    canvas = Image.new("RGB", (scale * len(failures), scale * 2), (28, 30, 34))
    for col, result in enumerate(failures):
        for row, key in enumerate(("front", "wrist")):
            img = Image.fromarray(result["last_obs"][key]).resize((scale, scale),
                                                                  Image.NEAREST)
            canvas.paste(img, (col * scale, row * scale))
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path)


def run_split(policy, env, seeds, split, n_blocks=1, target_colors=None,
              replan=5, max_steps=60, progress_label="", ensemble=0.0) -> list[dict]:
    results = []
    for i, seed in enumerate(seeds):
        color = None if target_colors is None else target_colors[i % len(target_colors)]
        result = run_rollout(policy, env, seed=seed, split=split, n_blocks=n_blocks,
                             target_color=color, replan_interval=replan,
                             max_steps=max_steps, capture_last=True,
                             ensemble_decay=ensemble)
        results.append(result)
        print(f"  [{progress_label} {i + 1}/{len(seeds)}] seed={seed} "
              f"success={result['success']} failure={result['failure']} "
              f"steps={result['steps']}", flush=True)
    return results


def main() -> int:
    args = parse_args()
    if args.smoke:
        args.seen_trials = 2
        args.unseen_trials = 1
        args.lang_trials = 2
    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.exists():
        print(f"checkpoint not found: {ckpt_path}")
        return 2
    policy = load_checkpoint(ckpt_path)
    ckpt = checkpoint_summary(ckpt_path)
    ckpt["path"] = _rel(ckpt_path)
    env = GraspEnv(EnvConfig(max_steps=args.max_steps, image_size=96))
    print(f"loaded {ckpt['policy_type']} checkpoint: {ckpt['n_parameters']/1e6:.2f}M params, "
          f"{ckpt['bytes']/1e6:.2f} MB", flush=True)

    report: dict = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "policy": args.policy,
        "policy_type": ckpt["policy_type"],
        "checkpoint": ckpt,
        "protocol": {
            "simulator": "so101_sim (kinematic, 96x96 front + wrist cameras)",
            "control_hz": env.config.fps,
            "max_steps": args.max_steps,
            "replan_interval": args.replan,
            "ensemble_decay": args.ensemble_decay,
            "seen_seeds": [SEEN_SEED_BASE + i for i in range(args.seen_trials)],
            "unseen_seeds": [UNSEEN_SEED_BASE + i for i in range(args.unseen_trials)],
            "lang_seeds": [LANG_SEED_BASE + i for i in range(args.lang_trials)],
            "success_definition": "instructed block inside the box after release",
        },
        "splits": {},
        "failure_cases": [],
        "images": {},
    }

    all_results: list[dict] = []
    if args.policy == "act":
        seen = run_split(policy, env, [SEEN_SEED_BASE + i for i in range(args.seen_trials)],
                         "seen", replan=args.replan, max_steps=args.max_steps,
                         progress_label="seen", ensemble=args.ensemble_decay)
        unseen = run_split(policy, env, [UNSEEN_SEED_BASE + i for i in range(args.unseen_trials)],
                           "unseen", replan=args.replan, max_steps=args.max_steps,
                           progress_label="unseen", ensemble=args.ensemble_decay)
        report["splits"]["seen"] = summarise_rollouts(seen)
        report["splits"]["unseen"] = summarise_rollouts(unseen)
        all_results = seen + unseen
        montage = Path(args.out_dir) / f"eval_{args.policy}_failures.png"
        save_failure_montage(all_results, montage)
        if montage.exists():
            report["images"]["failures"] = _rel(montage)
        failures = [r for r in all_results if not r["success"]]
        # Summarise *all* failure modes, but only detail a few examples.
        mode_counts = Counter(r["failure"] for r in failures)
        report["failure_mode_counts"] = dict(mode_counts)
        report["failure_cases"] = [
            {"seed": r["seed"], "split": r["split"], "failure": r["failure"],
             "target_color": r["target_color"], "target_size": r["target_size"],
             "has_occluder": r["has_occluder"],
             "release_error_m": r["release_error_m"],
             "wrong_object_in_box": r["wrong_object_in_box"],
             "description": describe_failure(r)}
            for r in failures[:5]
        ]
    else:
        colors = [SEEN_COLORS[i % len(SEEN_COLORS)] for i in range(args.lang_trials)]
        lang = run_split(policy, env, [LANG_SEED_BASE + i for i in range(args.lang_trials)],
                         "lang", n_blocks=2, target_colors=colors,
                         replan=args.replan, max_steps=args.max_steps,
                         progress_label="lang", ensemble=args.ensemble_decay)
        summary = summarise_rollouts(lang)
        summary["instruction_accuracy"] = summary["success_rate"]
        summary["wrong_object_rate"] = round(
            summary["wrong_object_in_box"] / max(summary["trials"], 1), 4)
        report["splits"]["lang"] = summary
        all_results = lang
        montage = Path(args.out_dir) / f"eval_{args.policy}_failures.png"
        save_failure_montage(all_results, montage)
        if montage.exists():
            report["images"]["failures"] = _rel(montage)
        failures = [r for r in lang if not r["success"]]
        report["failure_mode_counts"] = dict(Counter(r["failure"] for r in failures))
        report["failure_cases"] = [
            {"seed": r["seed"], "failure": r["failure"],
             "instruction": r["instruction"], "target_color": r["target_color"],
             "wrong_object_in_box": r["wrong_object_in_box"],
             "description": describe_failure(r)}
            for r in failures[:5]
        ]

    tag = args.tag or f"{args.policy}_{datetime.now().strftime('%Y%m%d')}"
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / f"eval_{tag}.json"
    md_path = out_dir / f"eval_{tag}.md"
    if args.smoke:
        json_path = out_dir / "eval_smoke.json"
        md_path = out_dir / "eval_smoke.md"
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    md_path.write_text(render_markdown(report), encoding="utf-8")
    print(f"report -> {json_path}\n         {md_path}", flush=True)
    for name, summary in report["splits"].items():
        print(f"[{name}] success {summary['successes']}/{summary['trials']} "
              f"({summary['success_rate'] * 100:.0f}%) | "
              f"latency {summary['latency_ms_mean']} ms | "
              f"cycle {summary['cycle_seconds_mean']} s | "
              f"failures {summary['failures']}", flush=True)
    return 0


def render_markdown(report: dict) -> str:
    lines = [
        f"# Evaluation - {report['policy']} policy",
        "",
        f"- Generated: {report['generated_at']}",
        f"- Checkpoint: `{report['checkpoint']['path']}` "
        f"({report['checkpoint']['bytes'] / 1e6:.2f} MB, "
        f"{report['checkpoint']['n_parameters'] / 1e6:.2f}M parameters)",
        f"- Control: {report['protocol']['control_hz']} Hz, "
        f"max {report['protocol']['max_steps']} steps "
        f"({report['protocol']['max_steps'] / report['protocol']['control_hz']:.1f} s), "
        f"replan every {report['protocol']['replan_interval']} steps",
        "- Success: the instructed block ends inside the box after release",
        "",
        "| split | trials | success | success rate | chunk latency (mean / p95) | "
        "sim cycle | wall cycle | failure modes |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for name, summary in report["splits"].items():
        failures = ", ".join(f"{k}: {v}" for k, v in summary["failures"].items()) or "-"
        lines.append(
            f"| {name} | {summary['trials']} | {summary['successes']} | "
            f"{summary['success_rate'] * 100:.1f}% | "
            f"{summary['latency_ms_mean']} / {summary['latency_ms_p95']} ms | "
            f"{summary['cycle_seconds_mean']} s | "
            f"{summary.get('wall_seconds_mean', '-')} s | {failures} |"
        )
    if "lang" in report["splits"]:
        summary = report["splits"]["lang"]
        lines += ["", f"- Instruction accuracy (full task): {summary.get('instruction_accuracy', 0) * 100:.1f}%",
                  f"- Wrong object placed rate: {summary.get('wrong_object_rate', 0) * 100:.1f}%",
                  f"- Grasp grounding (grasped the instructed block when a grasp "
                  f"happened): {summary.get('grounded_grasps', 0)}/"
                  f"{summary.get('grasp_trials', 0)} trials"]
        if summary.get("grounding_accuracy") is not None:
            lines[-1] += f" = {summary['grounding_accuracy'] * 100:.1f}%"
    lines += ["", "## Failure cases", ""]
    if report["failure_cases"]:
        for case in report["failure_cases"]:
            lines.append(f"- {case['description']}")
        if report.get("images", {}).get("failures"):
            lines.append("")
            lines.append(f"![failure cases]({Path(report['images']['failures']).name})")
    else:
        lines.append("- none")
    lines += [
        "",
        "## Notes",
        "",
        "- Simulator-only evaluation. There is no real SO-101 hardware in this project;",
        "  see `docs/real_robot.md` for the real-robot protocol.",
        "- Scene seeds are disjoint from training and from the seeds used for",
        "  training-time checkpoint selection.",
        "- `unseen` changes colour (yellow/purple never seen in training), block size",
        "  and workspace position, so the gap to `seen` is a real distribution-shift gap.",
        "- Latency is measured per action chunk on this CPU (batch size 1), including",
        "  image resize / normalisation but excluding simulator rendering.",
    ]
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    raise SystemExit(main())
