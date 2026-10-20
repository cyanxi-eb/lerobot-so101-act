#!/usr/bin/env python
"""Train the lightweight language-conditioned policy ("SmolVLA-style").

The policy receives two camera streams, the 6-D joint state and a Chinese /
English instruction such as ``把红色积木放进盒子里`` and predicts an action
chunk.  Training data comes from ``data/demos_lang`` (two blocks per scene,
the instruction selects which one to place).

Example::

    python scripts/train_lang_cond.py --max-minutes 25 --epochs 120
"""

from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]

def _rel(path) -> str:
    path = Path(path)
    try:
        return str(path.resolve().relative_to(REPO_ROOT))
    except ValueError:
        return path.name

sys.path.insert(0, str(REPO_ROOT / "src"))

from so101_policy.inference import LoadedPolicy, save_checkpoint  # noqa: E402
from so101_policy.lang_cond import (  # noqa: E402
    CharTokenizer,
    LangConditionedPolicy,
    LangConfig,
    lang_loss,
)
from so101_policy.rollout import run_rollouts, summarise_rollouts  # noqa: E402
from so101_policy.training import (  # noqa: E402
    TrainingLogger,
    ensure_cache,
    make_batches,
    normalizer_for,
    prepare_arrays,
    prepare_batch,
    set_seed,
    train_val_indices,
)
from so101_sim import SEEN_COLORS, EnvConfig, GraspEnv  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default=str(REPO_ROOT / "data" / "demos_lang"))
    parser.add_argument("--cache", default=str(REPO_ROOT / ".cache" / "lang_cache.npz"))
    parser.add_argument("--out", default=str(REPO_ROOT / "checkpoints" / "lang_cond.pt"))
    parser.add_argument("--report", default=None)
    parser.add_argument("--image-size", type=int, default=64)
    parser.add_argument("--chunk-size", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--max-minutes", type=float, default=25.0)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--val-episodes", type=int, default=8)
    parser.add_argument("--eval-every", type=int, default=10)
    parser.add_argument("--eval-trials", type=int, default=6)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def tokenize_tasks(tokenizer: CharTokenizer, tasks) -> torch.Tensor:
    ids = tokenizer.encode([str(t) for t in tasks])
    return torch.from_numpy(ids).long()


def main() -> int:
    args = parse_args()
    torch.set_num_threads(max(1, torch.get_num_threads()))
    set_seed(args.seed)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if args.smoke:
        args.epochs = 2
        args.max_minutes = 1.5
        args.eval_trials = 2
        args.eval_every = 1
        args.batch_size = 16
        if args.report is None:
            args.report = str(REPO_ROOT / "reports" / "train_lang_smoke.json")
        if args.out.endswith("lang_cond.pt"):
            args.out = str(Path(args.out).with_name("lang_cond_smoke.pt"))
        if args.cache.endswith("lang_cache.npz"):
            args.cache = str(Path(args.cache).with_name("lang_smoke_cache.npz"))
    if args.report is None:
        args.report = str(REPO_ROOT / "reports" / f"train_lang_{stamp}.json")

    cache_started = time.perf_counter()
    cache = ensure_cache(args.cache, args.dataset, image_size=args.image_size,
                         chunk_size=args.chunk_size, val_episodes=args.val_episodes,
                         max_episodes=3 if args.smoke else None)
    cache_seconds = time.perf_counter() - cache_started

    tokenizer = CharTokenizer.fit([str(t) for t in np.unique(cache["task"])])
    token_ids = tokenize_tasks(tokenizer, cache["task"])
    data = prepare_arrays(cache)
    data["token_ids"] = token_ids.numpy()
    train_idx, val_idx = train_val_indices(data, args.val_episodes)
    if len(train_idx) == 0 or len(val_idx) == 0:
        raise RuntimeError("empty train/val split; generate a larger dataset first")
    normalizer = normalizer_for(cache)

    config = LangConfig(image_size=args.image_size, chunk_size=args.chunk_size)
    model = LangConditionedPolicy(tokenizer.vocab_size, config)
    n_params = model.n_parameters()
    device = torch.device("cpu")
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                  weight_decay=args.weight_decay)
    steps_per_epoch = max(1, len(train_idx) // args.batch_size)
    planned_steps = max(1, args.epochs * steps_per_epoch)
    warmup = max(20, int(0.03 * planned_steps))

    def lr_lambda(step: int) -> float:
        if step < warmup:
            return (step + 1) / warmup
        progress = min(1.0, (step - warmup) / max(1, planned_steps - warmup))
        return 0.5 * (1.0 + np.cos(np.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    logger = TrainingLogger(args.report, f"Lang-conditioned training on "
                                         f"{Path(args.dataset).name}")
    eval_env = GraspEnv(EnvConfig(max_steps=60, image_size=96))
    eval_seeds = [7000 + args.seed * 100 + i for i in range(args.eval_trials)]
    eval_colors = [SEEN_COLORS[i % len(SEEN_COLORS)] for i in range(args.eval_trials)]
    loaded = LoadedPolicy(
        policy=model,
        normalizer=normalizer,
        policy_type="lang",
        image_size=args.image_size,
        chunk_size=args.chunk_size,
        device=device,
        tokenizer=tokenizer,
    )

    best = {"score": (-1.0, -1e9), "epoch": -1, "success_rate": 0.0, "val_l1": None,
            "checkpoint": None, "bytes": None}
    rollout_history: list[dict] = []
    train_started = time.perf_counter()
    global_step = 0
    epochs_done = 0
    print(f"model: {n_params/1e6:.2f}M params | vocab {tokenizer.vocab_size} | "
          f"train frames {len(train_idx)} | val frames {len(val_idx)} | "
          f"steps/epoch {steps_per_epoch}", flush=True)

    for epoch in range(1, args.epochs + 1):
        model.train()
        rng = np.random.default_rng(args.seed * 1000 + epoch)
        sums = {"loss": 0.0, "l1": 0.0}
        n_batches = 0
        epoch_started = time.perf_counter()
        for batch_idx in make_batches(train_idx, args.batch_size, rng):
            images, state, target, mask = prepare_batch(data, batch_idx, device)
            ids = torch.from_numpy(data["token_ids"][batch_idx]).to(device)
            pred = model(images, state, ids)
            loss, metrics = lang_loss(pred, target, mask)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            global_step += 1
            for key in sums:
                sums[key] += metrics[key]
            n_batches += 1
        record = {
            "epoch": epoch,
            "train_loss": round(sums["loss"] / max(n_batches, 1), 5),
            "train_l1": round(sums["l1"] / max(n_batches, 1), 5),
            "lr": round(float(optimizer.param_groups[0]["lr"]), 7),
            "epoch_seconds": round(time.perf_counter() - epoch_started, 2),
        }
        # Validation L1 (deterministic, teacher-forced chunk targets).
        model.eval()
        val_total, val_count = 0.0, 0.0
        with torch.no_grad():
            for start in range(0, len(val_idx), 64):
                batch_idx = val_idx[start:start + 64]
                images, state, target, mask = prepare_batch(data, batch_idx, device)
                ids = torch.from_numpy(data["token_ids"][batch_idx]).to(device)
                pred = model(images, state, ids)
                err = (pred - target).abs().mean(dim=-1)
                val_total += float((err * mask).sum())
                val_count += float(mask.sum())
        val_l1 = val_total / max(val_count, 1.0)
        record["val_l1"] = round(val_l1, 5)
        epochs_done = epoch

        if (epoch % args.eval_every == 0) or (epoch == args.epochs):
            results = run_rollouts(loaded, eval_seeds, split="lang", n_blocks=2,
                                   target_colors=eval_colors, replan_interval=5,
                                   env=eval_env)
            summary = summarise_rollouts(results)
            record["success_rate"] = summary["success_rate"]
            record["failure_modes"] = summary["failures"]
            record["latency_ms_mean"] = summary["latency_ms_mean"]
            record["wrong_object"] = summary["wrong_object_in_box"]
            rollout_history.append({"epoch": epoch, **summary})
            score = (float(summary["success_rate"]), -val_l1)
            if score > best["score"]:
                info = save_checkpoint(
                    args.out, model, normalizer, policy_type="lang",
                    image_size=args.image_size, chunk_size=args.chunk_size,
                    tokenizer=tokenizer, config=config.to_dict(),
                    metadata={
                        "epoch": epoch,
                        "val_l1": round(val_l1, 5),
                        "rollout_success_rate": summary["success_rate"],
                        "trained_on": "simulated scripted demonstrations (no teleop)",
                        "instructions": sorted(set(str(t) for t in np.unique(cache["task"]))),
                        "train_frames": int(len(train_idx)),
                    },
                )
                best.update(score=score, epoch=epoch,
                            success_rate=summary["success_rate"],
                            val_l1=round(val_l1, 5), checkpoint=info["path"],
                            bytes=info["bytes"])
                record["best"] = True
        logger.log_epoch(record)
        elapsed_minutes = (time.perf_counter() - train_started) / 60.0
        if elapsed_minutes >= args.max_minutes:
            print(f"time budget reached ({elapsed_minutes:.1f} min), stopping", flush=True)
            break

    train_seconds = time.perf_counter() - train_started
    logger.finalise(
        mode="smoke" if args.smoke else "full",
        config=config.to_dict(),
        cli=vars(args),
        dataset=_rel(args.dataset),
        cache=_rel(args.cache),
        cache_build_seconds=round(cache_seconds, 2),
        parameter_count=int(n_params),
        vocab_size=tokenizer.vocab_size,
        vocabulary=tokenizer.itos,
        checkpoint=best["checkpoint"],
        checkpoint_bytes=best["bytes"],
        checkpoint_mb=round((best["bytes"] or 0) / 1e6, 3),
        best_epoch=best["epoch"],
        best_success_rate=best["success_rate"],
        best_val_l1=best["val_l1"],
        epochs_done=epochs_done,
        total_steps=global_step,
        train_seconds=round(train_seconds, 2),
        train_minutes=round(train_seconds / 60.0, 2),
        steps_per_second=round(global_step / max(train_seconds, 1e-9), 3),
        torch_threads=int(torch.get_num_threads()),
        device="cpu",
        rollout_history=rollout_history,
    )
    print(f"done: {epochs_done} epochs, {train_seconds/60:.1f} min, "
          f"best epoch {best['epoch']} success {best['success_rate']} -> "
          f"{best['checkpoint']}", flush=True)
    print(f"report -> {args.report}", flush=True)
    return 0 if best["checkpoint"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
