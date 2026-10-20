#!/usr/bin/env python
"""Train the ACT policy on the simulated SO-101 demonstrations.

CPU-only by design.  Examples
-----------------------------
# CI smoke run (<= 2 minutes, tiny cache + 2 epochs)
python scripts/train_act.py --smoke

# Full run (stops after --max-minutes even if epochs remain)
python scripts/train_act.py --max-minutes 55 --epochs 200

The script writes ``reports/train_act_<timestamp>.json`` (config, losses,
rollout success per epoch, throughput) and ``reports/train_act_*.png`` curves.
The best checkpoint is selected by closed-loop success rate on held-out scene
seeds, with validation L1 as tie-breaker.
"""

from __future__ import annotations

import argparse
import os
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

from so101_policy.act import ACTConfig, ACTPolicy, act_loss  # noqa: E402
from so101_policy.inference import LoadedPolicy, save_checkpoint  # noqa: E402
from so101_policy.training import (  # noqa: E402
    TrainingLogger,
    ensure_cache,
    make_batches,
    normalizer_for,
    prepare_arrays,
    prepare_batch,
    set_seed,
    train_val_indices,
    validate_l1,
)
from so101_policy.rollout import run_rollouts, summarise_rollouts  # noqa: E402
from so101_sim import EnvConfig, GraspEnv  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default=str(REPO_ROOT / "data" / "demos_act"))
    parser.add_argument("--cache", default=str(REPO_ROOT / ".cache" / "act_cache.npz"))
    parser.add_argument("--out", default=str(REPO_ROOT / "checkpoints" / "act.pt"))
    parser.add_argument("--report", default=None)
    parser.add_argument("--image-size", type=int, default=64)
    parser.add_argument("--chunk-size", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--max-minutes", type=float, default=50.0)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--val-episodes", type=int, default=8)
    parser.add_argument("--eval-every", type=int, default=5)
    parser.add_argument("--eval-trials", type=int, default=8)
    parser.add_argument("--kl-weight", type=float, default=10.0)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--threads", type=int, default=max(1, os.cpu_count() or 1))
    parser.add_argument("--init-from", default=None,
                        help="optional checkpoint whose weights initialise this run "
                             "(used for a short fine-tune with a fresh LR schedule)")
    parser.add_argument("--lr-min", type=float, default=0.0,
                        help="final LR of the cosine schedule (fraction of --lr)")
    parser.add_argument("--smoke", action="store_true",
                        help="tiny run for CI: 3 episodes, 2 epochs, <=2 minutes")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    torch.set_num_threads(args.threads)
    set_seed(args.seed)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if args.smoke:
        args.epochs = min(args.epochs, 2)
        args.max_minutes = min(args.max_minutes, 1.8)
        args.eval_trials = min(args.eval_trials, 2)
        args.eval_every = 1
        args.batch_size = min(args.batch_size, 16)
        if args.report is None:
            args.report = str(REPO_ROOT / "reports" / "train_act_smoke.json")
        if args.out.endswith("act.pt"):
            args.out = str(Path(args.out).with_name("act_smoke.pt"))
        if args.cache.endswith("act_cache.npz"):
            args.cache = str(Path(args.cache).with_name("act_smoke_cache.npz"))
    if args.report is None:
        args.report = str(REPO_ROOT / "reports" / f"train_act_{stamp}.json")

    cache_started = time.perf_counter()
    cache = ensure_cache(
        args.cache, args.dataset, image_size=args.image_size,
        chunk_size=args.chunk_size, val_episodes=args.val_episodes,
        max_episodes=3 if args.smoke else None,
    )
    cache_seconds = time.perf_counter() - cache_started
    data = prepare_arrays(cache)
    train_idx, val_idx = train_val_indices(data, args.val_episodes)
    if len(train_idx) == 0 or len(val_idx) == 0:
        raise RuntimeError("empty train/val split; generate a larger dataset first")
    normalizer = normalizer_for(cache)

    config = ACTConfig(
        image_size=args.image_size,
        chunk_size=args.chunk_size,
        kl_weight=args.kl_weight,
    )
    model = ACTPolicy(config)
    if args.init_from:
        from so101_policy.inference import load_checkpoint as _load_ckpt

        init = _load_ckpt(args.init_from)
        model.load_state_dict(init.policy.state_dict())
        print(f"initialised weights from {args.init_from}", flush=True)
    n_params = model.n_parameters()
    device = torch.device("cpu")
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                  weight_decay=args.weight_decay)
    steps_per_epoch = max(1, len(train_idx) // args.batch_size)
    planned_steps = args.epochs * steps_per_epoch
    warmup = max(20, int(0.03 * planned_steps))
    lr_floor = float(args.lr_min)

    def lr_lambda(step: int) -> float:
        if step < warmup:
            return (step + 1) / warmup
        progress = min(1.0, (step - warmup) / max(1, planned_steps - warmup))
        return lr_floor + (1.0 - lr_floor) * 0.5 * (1.0 + np.cos(np.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    logger = TrainingLogger(args.report, f"ACT training on {Path(args.dataset).name}")
    eval_env = GraspEnv(EnvConfig(max_steps=60, image_size=96))
    eval_seeds = [5000 + args.seed * 100 + i for i in range(args.eval_trials)]
    loaded = LoadedPolicy(
        policy=model,
        normalizer=normalizer,
        policy_type="act",
        image_size=args.image_size,
        chunk_size=args.chunk_size,
        device=device,
    )

    best = {"score": (-1.0, -1e9), "epoch": -1, "success_rate": 0.0, "val_l1": None,
            "checkpoint": None, "bytes": None}
    rollout_history: list[dict] = []
    train_started = time.perf_counter()
    global_step = 0
    epochs_done = 0

    print(f"model: {n_params/1e6:.2f}M params | train frames {len(train_idx)} | "
          f"val frames {len(val_idx)} | steps/epoch {steps_per_epoch} | "
          f"threads {torch.get_num_threads()}", flush=True)

    for epoch in range(1, args.epochs + 1):
        model.train()
        rng = np.random.default_rng(args.seed * 1000 + epoch)
        sums = {"loss": 0.0, "l1": 0.0, "kl": 0.0}
        n_batches = 0
        epoch_started = time.perf_counter()
        for batch_idx in make_batches(train_idx, args.batch_size, rng):
            images, state, target, mask = prepare_batch(data, batch_idx, device)
            pred, mu, logvar = model(images, state, target)
            loss, metrics = act_loss(pred, target, mask, mu, logvar,
                                     kl_weight=args.kl_weight)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
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
            "train_kl": round(sums["kl"] / max(n_batches, 1), 5),
            "lr": round(float(optimizer.param_groups[0]["lr"]), 7),
            "epoch_seconds": round(time.perf_counter() - epoch_started, 2),
        }
        val_l1 = validate_l1(model, data, val_idx, device)
        record["val_l1"] = round(val_l1, 5)
        epochs_done = epoch

        should_eval = (epoch % args.eval_every == 0) or (epoch == args.epochs)
        success_rate = None
        if should_eval:
            results = run_rollouts(loaded, eval_seeds, split="seen", replan_interval=5,
                                   env=eval_env)
            summary = summarise_rollouts(results)
            success_rate = summary["success_rate"]
            record["success_rate"] = success_rate
            record["failure_modes"] = summary["failures"]
            record["latency_ms_mean"] = summary["latency_ms_mean"]
            rollout_history.append({"epoch": epoch, **summary})
            score = (float(success_rate), -val_l1)
            if score > best["score"]:
                info = save_checkpoint(
                    args.out, model, normalizer, policy_type="act",
                    image_size=args.image_size, chunk_size=args.chunk_size,
                    config=config.to_dict(),
                    metadata={
                        "epoch": epoch,
                        "val_l1": round(val_l1, 5),
                        "rollout_success_rate": success_rate,
                        "trained_on": "simulated scripted demonstrations (no teleop)",
                        "train_frames": int(len(train_idx)),
                        "val_frames": int(len(val_idx)),
                        "elapsed_seconds": round(time.perf_counter() - train_started, 1),
                    },
                )
                best.update(
                    score=score, epoch=epoch, success_rate=success_rate,
                    val_l1=round(val_l1, 5), checkpoint=info["path"],
                    bytes=info["bytes"],
                )
                record["best"] = True
        logger.log_epoch(record)
        elapsed_minutes = (time.perf_counter() - train_started) / 60.0
        if elapsed_minutes >= args.max_minutes:
            print(f"time budget reached ({elapsed_minutes:.1f} min), stopping", flush=True)
            break

    train_seconds = time.perf_counter() - train_started
    summary = logger.finalise(
        mode="smoke" if args.smoke else "full",
        config=config.to_dict(),
        cli=vars(args),
        dataset=_rel(args.dataset),
        cache=_rel(args.cache),
        cache_build_seconds=round(cache_seconds, 2),
        parameter_count=int(n_params),
        checkpoint=best["checkpoint"],
        checkpoint_bytes=best["bytes"],
        checkpoint_mb=round((best["bytes"] or 0) / 1e6, 3),
        best_epoch=best["epoch"],
        best_success_rate=best["success_rate"],
        best_val_l1=best["val_l1"],
        epochs_done=epochs_done,
        batches_per_epoch=steps_per_epoch,
        total_steps=global_step,
        train_seconds=round(train_seconds, 2),
        train_minutes=round(train_seconds / 60.0, 2),
        steps_per_second=round(global_step / max(train_seconds, 1e-9), 3),
        torch_threads=int(torch.get_num_threads()),
        device="cpu",
        rollout_history=rollout_history,
    )
    print(f"done: {epochs_done} epochs, {train_seconds/60:.1f} min, "
          f"best epoch {best['epoch']} success {best['success_rate']}, "
          f"checkpoint {best['bytes'] and best['bytes']/1e6:.2f} MB -> {best['checkpoint']}",
          flush=True)
    print(f"report -> {args.report}", flush=True)
    if best["checkpoint"] is None:
        print("WARNING: no checkpoint saved", flush=True)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
