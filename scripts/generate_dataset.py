#!/usr/bin/env python
"""Generate scripted demonstrations and write a LeRobot-v3-style dataset.

Examples
--------
# 80 ACT demonstrations (single block -> box)
python scripts/generate_dataset.py --out data/demos_act --episodes 80

# 80 language-conditioned demonstrations (two blocks, instruction selects one)
python scripts/generate_dataset.py --out data/demos_lang --episodes 80 --mode lang

No real teleoperation is involved: the demonstrations come from the scripted
Cartesian expert in :mod:`so101_sim.expert` running in the simulator.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from so101_dataset import LeRobotDatasetWriter  # noqa: E402
from so101_sim import ACT_TASK, SEEN_COLORS, EnvConfig, GraspEnv  # noqa: E402
from so101_sim.expert import rollout_expert  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="data/demos_act", help="dataset output directory")
    parser.add_argument("--episodes", type=int, default=80)
    parser.add_argument("--mode", choices=["act", "lang"], default="act")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--image-size", type=int, default=96)
    parser.add_argument("--max-attempts", type=int, default=4)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--report", default=None, help="report json path")
    return parser.parse_args()


def make_montage(episodes_frames: list[np.ndarray], path: Path) -> None:
    """Save a 2 x 5 observation grid (front / wrist cameras)."""
    if not episodes_frames:
        return
    picks = [0, 12, 24, 36, 54]
    picks = [p for p in picks if p < len(episodes_frames[0][0])]
    scale = 128
    rows = len(episodes_frames)
    canvas = Image.new("RGB", (scale * len(picks), scale * rows), (32, 34, 38))
    for r, cams in enumerate(episodes_frames):
        for c, idx in enumerate(picks):
            for cam_idx, frames in enumerate(cams):
                img = Image.fromarray(frames[idx]).resize((scale // 2 - 2, scale - 4),
                                                          Image.NEAREST)
                x = c * scale + cam_idx * (scale // 2)
                canvas.paste(img, (x + 1, r * scale + 2))
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path)
    print(f"sample montage -> {path}")


def main() -> int:
    args = parse_args()
    out = Path(args.out)
    if out.exists() and any(out.iterdir()) and not args.overwrite:
        print(f"refusing to overwrite existing dataset at {out} (use --overwrite)")
        return 2
    if out.exists() and args.overwrite:
        import shutil

        shutil.rmtree(out)

    report_path = Path(args.report) if args.report else (
        REPO_ROOT / "reports" / f"dataset_{args.mode}.json"
    )
    env = GraspEnv(EnvConfig(max_steps=60, image_size=args.image_size))
    writer = LeRobotDatasetWriter(out, fps=env.config.fps, image_size=args.image_size)

    started = time.perf_counter()
    attempts = 0
    dropped = 0
    colors: Counter = Counter()
    instructions: Counter = Counter()
    grasp_errors: list[float] = []
    montage_cams: list[list[np.ndarray]] = []
    episode_summaries: list[dict] = []

    for ep in range(args.episodes):
        recorded = False
        for attempt in range(args.max_attempts):
            attempts += 1
            seed = args.seed + ep * 17 + attempt * 100003
            split = "lang" if args.mode == "lang" else "train"
            target_color = None
            if args.mode == "lang":
                target_color = SEEN_COLORS[ep % len(SEEN_COLORS)]
            env.reset(
                seed=seed,
                split=split,
                n_blocks=2 if args.mode == "lang" else 1,
                target_color=target_color,
            )
            result = rollout_expert(env)
            if not result["success"]:
                dropped += 1
                continue
            scene = result["scene"]
            task = scene["instruction"] if args.mode == "lang" else ACT_TASK
            writer.add_episode(
                front=result["front"],
                wrist=result["wrist"],
                states=result["state"],
                actions=result["action"],
                task=task,
                scene_meta=scene,
            )
            colors[scene["target_color"]] += 1
            instructions[task] += 1
            for event in result["events"]:
                if event["event"] == "grasp":
                    grasp_errors.append(float(event["error"]))
            episode_summaries.append(
                {
                    "episode_index": len(episode_summaries),
                    "seed": seed,
                    "attempt": attempt,
                    "success": True,
                    "target_color": scene["target_color"],
                    "target_size": scene["target_size"],
                    "has_occluder": scene["has_occluder"],
                    "grasp_error": next(
                        (round(float(e["error"]), 5) for e in result["events"]
                         if e["event"] == "grasp"), None
                    ),
                }
            )
            if len(montage_cams) < 2 and len(episode_summaries) in (1, args.episodes // 2):
                montage_cams.append([result["front"], result["wrist"]])
            recorded = True
            break
        if not recorded:
            print(f"episode {ep}: dropped after {args.max_attempts} failed attempts")
        if (ep + 1) % 10 == 0 or ep == args.episodes - 1:
            print(f"[{args.mode}] generated {len(episode_summaries)}/{args.episodes} "
                  f"episodes (attempts={attempts}, dropped={dropped}, "
                  f"{time.perf_counter() - started:.1f}s)")

    info = writer.finalize()
    elapsed = time.perf_counter() - started
    dataset_bytes = sum(p.stat().st_size for p in out.rglob("*") if p.is_file())
    report = {
        "mode": args.mode,
        "dataset_root": str(out.resolve()),
        "requested_episodes": args.episodes,
        "episodes": info["total_episodes"],
        "frames": info["total_frames"],
        "attempts": attempts,
        "dropped_episodes": dropped,
        "expert_success_rate": round(info["total_episodes"] / max(attempts, 1), 4),
        "episode_length": 60,
        "fps": info["fps"],
        "cycle_seconds": round(60 / info["fps"], 3),
        "image_size": args.image_size,
        "target_color_counts": dict(colors),
        "instruction_counts": dict(instructions),
        "grasp_error_mean": round(float(np.mean(grasp_errors)), 6) if grasp_errors else None,
        "grasp_error_max": round(float(np.max(grasp_errors)), 6) if grasp_errors else None,
        "dataset_bytes": int(dataset_bytes),
        "dataset_mb": round(dataset_bytes / 1e6, 2),
        "generation_seconds": round(elapsed, 2),
        "episodes_summary": episode_summaries,
        "notes": [
            "Simulator only: demonstrations are scripted, not teleoperated.",
            "Domain randomisation covers pose, colour, size, camera error, occlusion, slip.",
        ],
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    make_montage(montage_cams, REPO_ROOT / "reports" / f"dataset_{args.mode}_samples.png")
    print(f"dataset written to {out}: {info['total_episodes']} episodes, "
          f"{info['total_frames']} frames, {elapsed:.1f}s, {dataset_bytes / 1e6:.1f} MB")
    print(f"report -> {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
