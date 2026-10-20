#!/usr/bin/env python
"""Record real closed-loop rollouts of the trained policies as GIF demos.

Each GIF is a side-by-side front / wrist camera view of one successful
rollout in the simulator, with a small footer showing the policy, the
instruction (for the language policy) and the measured cycle time.
Frames come from the actual policy rollouts - nothing is hand-animated.

Examples
--------
    python scripts/record_demo.py --act-checkpoint checkpoints/act.pt
    python scripts/record_demo.py --lang-checkpoint checkpoints/lang_cond.pt
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from so101_policy.inference import load_checkpoint  # noqa: E402
from so101_policy.rollout import run_rollout  # noqa: E402
from so101_sim import SEEN_COLORS, EnvConfig, GraspEnv  # noqa: E402

FONT_CANDIDATES = (
    "C:/Windows/Fonts/msyh.ttc",
    "C:/Windows/Fonts/simhei.ttf",
    "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
)


def load_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    for candidate in FONT_CANDIDATES:
        if Path(candidate).exists():
            try:
                return ImageFont.truetype(candidate, size)
            except OSError:
                continue
    try:
        return ImageFont.truetype("DejaVuSans.ttf", size)
    except OSError:
        return ImageFont.load_default()


def annotate(frame: dict, step: int, fps: float, footer: str,
             font: ImageFont.ImageFont, scale: int = 2) -> Image.Image:
    views = []
    for key, label in (("front", "front camera"), ("wrist", "wrist camera")):
        img = Image.fromarray(frame[key]).resize((96 * scale, 96 * scale), Image.NEAREST)
        draw = ImageDraw.Draw(img)
        draw.text((4, 4), label, fill=(255, 255, 255), font=font,
                  stroke_width=1, stroke_fill=(0, 0, 0))
        views.append(img)
    canvas = Image.new("RGB", (views[0].width * 2 + 4, views[0].height + 22), (18, 20, 24))
    canvas.paste(views[0], (0, 0))
    canvas.paste(views[1], (views[0].width + 4, 0))
    draw = ImageDraw.Draw(canvas)
    draw.text((4, views[0].height + 4), f"t={step / fps:4.1f}s", fill=(190, 220, 255),
              font=font)
    draw.text((70, views[0].height + 4), footer, fill=(220, 220, 220), font=font)
    return canvas


def rollout_gif(policy, env, seeds, split, n_blocks, footer_prefix, out_path,
                fps, max_attempts=40, gif_frames=45, replan=5,
                ensemble_decay: float = 0.0, show_instruction: bool = False) -> dict:
    font = load_font(11)
    chosen = None
    for seed in seeds[:max_attempts]:
        result = run_rollout(policy, env, seed=seed, split=split, n_blocks=n_blocks,
                             replan_interval=replan, capture_frames=True,
                             ensemble_decay=ensemble_decay)
        print(f"  seed={seed} success={result['success']} "
              f"failure={result['failure']} steps={result['steps']}", flush=True)
        if result["success"]:
            chosen = result
            break
    if chosen is None:
        print(f"could not find a successful rollout for {out_path}")
        return {"ok": False}
    frames = chosen["frames"]
    footer = (f"{footer_prefix} | success | {chosen['steps'] / fps:.1f}s | "
              f"{chosen['latency_ms_mean']:.0f} ms/chunk")
    if show_instruction and chosen.get("instruction"):
        footer = f"{footer_prefix} | success | {chosen['steps'] / fps:.1f}s | {chosen['instruction']}"
    step = max(1, int(np.ceil(len(frames) / gif_frames)))
    picked = frames[::step]
    images = [annotate(frame, i * step, fps, footer, font) for i, frame in enumerate(picked)]
    # Quantise to a shared adaptive palette; much smaller GIFs than RGB frames.
    palette = images[0].quantize(colors=128, method=Image.Quantize.MEDIANCUT)
    quantised = [img.quantize(colors=128, palette=palette) for img in images]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    duration = int(round(1000.0 * step / fps))
    quantised[0].save(
        out_path,
        save_all=True,
        append_images=quantised[1:],
        duration=duration,
        loop=0,
        optimize=True,
        disposal=2,
    )
    size = out_path.stat().st_size
    try:
        display_path = str(out_path.resolve().relative_to(REPO_ROOT)).replace("\\", "/")
    except ValueError:
        display_path = out_path.name
    print(f"wrote {out_path} ({size / 1e6:.2f} MB, {len(quantised)} frames, "
          f"{duration} ms/frame)", flush=True)
    return {"ok": True, "path": display_path, "bytes": int(size),
            "frames": len(quantised), "seed": chosen["seed"],
            "steps": chosen["steps"], "latency_ms": chosen["latency_ms_mean"],
            "cycle_seconds": round(chosen["steps"] / fps, 2),
            "instruction": chosen["instruction"]}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--act-checkpoint", default=str(REPO_ROOT / "checkpoints" / "act.pt"))
    parser.add_argument("--lang-checkpoint", default=str(REPO_ROOT / "checkpoints" / "lang_cond.pt"))
    parser.add_argument("--out", default=str(REPO_ROOT / "demos"))
    parser.add_argument("--only", choices=["act", "lang", "both"], default="both")
    parser.add_argument("--replan", type=int, default=3,
                        help="actions executed per predicted chunk")
    parser.add_argument("--ensemble-decay", type=float, default=0.01,
                        help="temporal-ensembling decay (0 = plain chunk execution)")
    parser.add_argument("--max-bytes", type=float, default=3.5e6)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    env = GraspEnv(EnvConfig(max_steps=60, image_size=96))
    manifest_path = out / "demos.json"
    manifest = {"generated_at": __import__("datetime").datetime.now().isoformat(
        timespec="seconds"), "demos": {}}
    if manifest_path.exists():
        try:
            import json as _json

            previous = _json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["demos"].update(previous.get("demos", {}))
        except Exception:
            pass

    if args.only in ("act", "both"):
        ckpt = Path(args.act_checkpoint)
        if ckpt.exists():
            policy = load_checkpoint(ckpt)
            seeds = [3000 + i for i in range(40)]
            info = rollout_gif(policy, env, seeds, "seen", 1,
                               f"ACT | chunk 20 | replan {args.replan}",
                               out / "01_act_rollout.gif", env.config.fps,
                               replan=args.replan,
                               ensemble_decay=args.ensemble_decay)
            manifest["demos"]["act"] = info
        else:
            print(f"missing ACT checkpoint {ckpt}, skipping")

    if args.only in ("lang", "both"):
        ckpt = Path(args.lang_checkpoint)
        if ckpt.exists():
            policy = load_checkpoint(ckpt)
            seeds = [3200 + i for i in range(40)]
            # Record a rollout with a simple, unambiguous instruction.
            info = rollout_gif(policy, env, seeds, "lang", 2,
                               "lang policy", out / "02_lang_cond.gif",
                               env.config.fps, replan=args.replan,
                               ensemble_decay=args.ensemble_decay,
                               show_instruction=True)
            manifest["demos"]["lang"] = info
        else:
            print(f"missing language checkpoint {ckpt}, skipping")

    manifest_path = out / "demos.json"
    if manifest["demos"]:
        import json

        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2),
                                 encoding="utf-8")
        print(f"manifest -> {manifest_path}")
    for key, info in manifest["demos"].items():
        if info.get("bytes", 0) > args.max_bytes:
            print(f"WARNING: {key} gif is {info['bytes'] / 1e6:.2f} MB "
                  f"(> {args.max_bytes / 1e6:.1f} MB target)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
