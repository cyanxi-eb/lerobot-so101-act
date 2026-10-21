"""Shared training utilities (batch building, validation, curves, reporting).

Kept separate from the training scripts so that the ACT and the
language-conditioned trainer can share the exact same data path, and so that
tests can exercise the batch/mask logic without spawning training runs.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch

from so101_dataset import build_cache, load_cache
from .normalization import Normalizer


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)


def ensure_cache(cache_path, dataset_root, image_size: int, chunk_size: int,
                 val_episodes: int, max_episodes: int | None = None,
                 log=print) -> dict:
    """Load the ``.npz`` cache, building it from the dataset when absent."""
    cache_path = Path(cache_path)
    if cache_path.exists():
        cache = load_cache(cache_path)
        if cache["meta"] and int(cache["meta"].get("image_size", image_size)) != image_size:
            raise ValueError(
                f"cache {cache_path} was built for image_size "
                f"{cache['meta'].get('image_size')} but {image_size} was requested"
            )
        log(f"loaded cache {cache_path} ({int(cache['episode_offsets'][-1])} frames)")
        return cache
    meta = build_cache(dataset_root, cache_path, image_size=image_size,
                       chunk_size=chunk_size, val_episodes=val_episodes,
                       max_episodes=max_episodes, log=log)
    cache = load_cache(cache_path)
    cache["meta"] = meta
    return cache


def normalizer_for(cache: dict) -> Normalizer:
    return Normalizer.from_cache_meta(cache["meta"])


def prepare_arrays(cache: dict) -> dict:
    """Normalise states / action chunks once, in place, into float32 arrays."""
    normalizer = normalizer_for(cache)
    data = {
        "front": cache["front"],
        "wrist": cache["wrist"],
        "state": normalizer.normalize_state(cache["state"]).astype(np.float32),
        "chunks": normalizer.normalize_action(cache["action_chunks"]).astype(np.float32),
        "mask": cache["action_mask"].astype(np.float32),
        "episode_index": cache["episode_index"],
        "task": cache["task"],
        "episode_offsets": cache["episode_offsets"],
    }
    return data


def train_val_indices(data: dict, val_episodes: int) -> tuple[np.ndarray, np.ndarray]:
    offsets = data["episode_offsets"]
    n_eps = len(offsets) - 1
    # Clamp so tiny datasets (CI smoke runs) still leave a training split.
    val_episodes = max(1, min(int(val_episodes), max(1, n_eps - 1))) if n_eps > 1 else 0
    val_eps = set(range(n_eps - val_episodes, n_eps)) if val_episodes else set()
    all_idx = np.arange(int(offsets[-1]), dtype=np.int64)
    is_val = np.isin(data["episode_index"], list(val_eps))
    return all_idx[~is_val], all_idx[is_val]


def make_batches(indices: np.ndarray, batch_size: int, rng: np.random.Generator | None = None):
    """Yield arrays of indices; shuffle when ``rng`` is given."""
    idx = np.asarray(indices, dtype=np.int64)
    if rng is not None:
        idx = idx[rng.permutation(len(idx))]
    for start in range(0, len(idx) - batch_size + 1, batch_size):
        yield idx[start:start + batch_size]


def augment_batch(images: torch.Tensor) -> torch.Tensor:
    """Photometric + small-translation augmentation for training batches.

    Kept intentionally mild: per-channel colour gain, brightness / contrast,
    a small pixel shift and a little Gaussian noise.  Validation and
    evaluation never take this path (``augment`` defaults to False).
    """
    b = images.shape[0]
    dev = images.device
    gain = torch.rand(b, 1, 3, 1, 1, device=dev) * 0.35 + 0.825
    images = images * gain
    bright = torch.rand(b, 1, 1, 1, 1, device=dev) * 0.3 + 0.85
    images = images * bright
    mean = images.mean(dim=(2, 3, 4), keepdim=True)
    contrast = torch.rand(b, 1, 1, 1, 1, device=dev) * 0.3 + 0.85
    images = (images - mean) * contrast + mean
    dx = int(torch.randint(-3, 4, (1,)).item())
    dy = int(torch.randint(-3, 4, (1,)).item())
    if dx or dy:
        images = torch.roll(images, shifts=(dy, dx), dims=(3, 4))
    images = images + torch.randn_like(images) * 0.015
    return images.clamp_(0.0, 1.0)


def prepare_batch(data: dict, batch_idx, device,
                  augment: bool = False) -> tuple[torch.Tensor, torch.Tensor,
                                                  torch.Tensor, torch.Tensor]:
    """Build ``(images, state, target_chunk, mask)`` tensors for one batch."""
    batch_idx = np.asarray(batch_idx, dtype=np.int64)
    front = data["front"][batch_idx]
    wrist = data["wrist"][batch_idx]
    imgs = np.stack([front, wrist], axis=1)  # (B, 2, H, W, 3)
    images = torch.from_numpy(imgs).to(device).permute(0, 1, 4, 2, 3).float().div_(255.0)
    if augment:
        images = augment_batch(images)
    state = torch.from_numpy(data["state"][batch_idx]).to(device)
    target = torch.from_numpy(data["chunks"][batch_idx]).to(device)
    mask = torch.from_numpy(data["mask"][batch_idx]).to(device)
    return images, state, target, mask


def masked_l1(pred, target, mask) -> torch.Tensor:
    err = (pred - target).abs().mean(dim=-1)
    return (err * mask).sum() / mask.sum().clamp_min(1.0)


@torch.no_grad()
def validate_l1(model, data: dict, indices: np.ndarray, device,
                batch_size: int = 64) -> float:
    """Deterministic (z = 0) masked L1 in normalised action space."""
    model.eval()
    total, count = 0.0, 0.0
    for start in range(0, len(indices), batch_size):
        batch_idx = indices[start:start + batch_size]
        images, state, target, mask = prepare_batch(data, batch_idx, device)
        model_type = getattr(model, "policy_type", None)
        if model_type == "lang":
            continue
        pred = model.predict(images, state)
        err = (pred - target).abs().mean(dim=-1)
        total += float((err * mask).sum())
        count += float(mask.sum())
    return total / max(count, 1.0)


class TrainingLogger:
    """Collects per-epoch metrics and writes the JSON report + curve figure."""

    def __init__(self, path, title: str):
        self.path = Path(path)
        self.title = title
        self.history: list[dict] = []
        self.extra: dict = {}
        self.started = time.perf_counter()

    def log_epoch(self, record: dict) -> None:
        self.history.append(record)
        line = " ".join(f"{k}={v}" for k, v in record.items())
        print(f"  {line}", flush=True)

    def finalise(self, **extra) -> dict:
        payload = {
            "title": self.title,
            "wall_seconds": round(time.perf_counter() - self.started, 2),
            "history": self.history,
            **extra,
        }
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(payload, ensure_ascii=False, indent=2),
                             encoding="utf-8")
        save_curves(self.history, self.path.with_suffix(".png"), self.title)
        return payload


def save_curves(history: list[dict], path, title: str) -> None:
    """Plot loss / success curves without requiring an interactive backend."""
    if not history:
        return
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:  # pragma: no cover - matplotlib is optional
        return
    epochs = [row.get("epoch", i) for i, row in enumerate(history)]
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.2))
    if "train_l1" in history[0]:
        axes[0].plot(epochs, [row.get("train_l1", np.nan) for row in history],
                     label="train L1")
    if "val_l1" in history[0]:
        axes[0].plot(epochs, [row.get("val_l1", np.nan) for row in history],
                     label="val L1")
    axes[0].set_xlabel("epoch")
    axes[0].set_ylabel("masked L1 (normalised)")
    axes[0].legend()
    axes[0].grid(alpha=0.3)
    if "success_rate" in history[0]:
        axes[1].plot(epochs, [row.get("success_rate", np.nan) for row in history],
                     color="tab:green", label="rollout success")
        axes[1].set_ylim(-0.02, 1.02)
    axes[1].set_xlabel("epoch")
    axes[1].set_ylabel("success rate")
    axes[1].legend()
    axes[1].grid(alpha=0.3)
    fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=110)
    plt.close(fig)
