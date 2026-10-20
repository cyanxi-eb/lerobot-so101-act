"""Training cache: decoded, resized observations + action chunks.

The dataset stores 96x96 videos (nice for humans / demos).  Policy training
uses 64x64 images because the whole project must train on a CPU-only budget.  This
module reads the dataset once, decodes + resizes every frame, builds padded
action-chunk targets and normalisation statistics, and writes a single
``.npz`` cache that training / evaluation can load in a second.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
from PIL import Image

from .reader import LeRobotDataset


def resize_frames(frames: np.ndarray, size: int) -> np.ndarray:
    """Resize ``(T,H,W,3)`` uint8 frames with bilinear interpolation."""
    if frames.shape[1] == size and frames.shape[2] == size:
        return frames.astype(np.uint8)
    out = np.empty((frames.shape[0], size, size, 3), dtype=np.uint8)
    for i, frame in enumerate(frames):
        img = Image.fromarray(frame).resize((size, size), Image.BILINEAR)
        out[i] = np.asarray(img, dtype=np.uint8)
    return out


def build_action_chunks(actions: np.ndarray, episode_offsets: np.ndarray,
                        chunk_size: int) -> tuple[np.ndarray, np.ndarray]:
    """Build padded action chunks for every frame.

    ``actions`` is the concatenated ``(N, 6)`` action array, ``episode_offsets``
    are the per-episode start offsets (length ``E + 1``).  For frame ``i`` the
    chunk is ``actions[i:i + chunk_size]`` truncated at the episode boundary
    and right-padded by repeating the last valid action.
    """
    n = actions.shape[0]
    dim = actions.shape[1]
    chunks = np.zeros((n, chunk_size, dim), dtype=np.float32)
    mask = np.zeros((n, chunk_size), dtype=np.float32)
    for ep in range(len(episode_offsets) - 1):
        start = int(episode_offsets[ep])
        end = int(episode_offsets[ep + 1])
        for i in range(start, end):
            avail = min(chunk_size, end - i)
            chunks[i, :avail] = actions[i:i + avail]
            chunks[i, avail:] = actions[end - 1]
            mask[i, :avail] = 1.0
    return chunks, mask


def compute_norm_stats(values: np.ndarray) -> dict[str, list[float]]:
    """Per-dimension mean / std / min / max."""
    values = np.asarray(values, dtype=np.float64)
    return {
        "mean": values.mean(axis=0).round(6).tolist(),
        "std": np.maximum(values.std(axis=0), 1e-6).round(6).tolist(),
        "min": values.min(axis=0).round(6).tolist(),
        "max": values.max(axis=0).round(6).tolist(),
    }


def build_cache(dataset_root, out_path, image_size: int = 64, chunk_size: int = 20,
                val_episodes: int = 8, max_episodes: int | None = None,
                log=print) -> dict:
    """Decode a dataset into an ``.npz`` training cache and return its metadata."""
    t0 = time.perf_counter()
    dataset = LeRobotDataset.load(dataset_root)
    total = dataset.num_episodes if max_episodes is None else min(max_episodes,
                                                                 dataset.num_episodes)
    fronts, wrists, states, actions, ep_ids, frame_ids, tasks = [], [], [], [], [], [], []
    ep_offsets = [0]
    for ep in range(total):
        data = dataset.get_episode(ep)
        fronts.append(resize_frames(data["front"], image_size))
        wrists.append(resize_frames(data["wrist"], image_size))
        states.append(data["state"])
        actions.append(data["action"])
        length = int(data["state"].shape[0])
        ep_ids.append(np.full(length, ep, dtype=np.int32))
        frame_ids.append(np.arange(length, dtype=np.int32))
        tasks.append(np.full(length, data["task"], dtype=object))
        ep_offsets.append(ep_offsets[-1] + length)
        if (ep + 1) % 10 == 0:
            log(f"  cached {ep + 1}/{total} episodes")

    front = np.concatenate(fronts).astype(np.uint8)
    wrist = np.concatenate(wrists).astype(np.uint8)
    state = np.concatenate(states).astype(np.float32)
    action = np.concatenate(actions).astype(np.float32)
    episode_index = np.concatenate(ep_ids)
    frame_index = np.concatenate(frame_ids)
    task = np.concatenate(tasks).astype(str)
    offsets = np.asarray(ep_offsets, dtype=np.int64)

    chunks, mask = build_action_chunks(action, offsets, chunk_size)

    # Normalisation statistics are computed on the training split only, so the
    # validation episodes stay a fair proxy for unseen data.  With tiny
    # datasets (CI smoke runs) clamp the validation split so the training set
    # is never empty.
    if total > 1:
        val_episodes = max(1, min(int(val_episodes), total - 1))
    else:
        val_episodes = 0
    if val_episodes > 0:
        train_mask = episode_index < (total - val_episodes)
    else:
        train_mask = np.ones_like(episode_index, dtype=bool)
    state_stats = compute_norm_stats(state[train_mask])
    action_stats = compute_norm_stats(action[train_mask])

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        out_path,
        front=front,
        wrist=wrist,
        state=state,
        action=action,
        action_chunks=chunks,
        action_mask=mask,
        episode_index=episode_index,
        frame_index=frame_index,
        task=task,
        episode_offsets=offsets,
    )
    meta = {
        "dataset_root": str(Path(dataset_root).resolve()),
        "cache_path": str(out_path.resolve()),
        "num_episodes": int(total),
        "num_frames": int(state.shape[0]),
        "val_episodes": int(val_episodes),
        "image_size": int(image_size),
        "chunk_size": int(chunk_size),
        "state_stats": state_stats,
        "action_stats": action_stats,
        "cache_bytes": int(out_path.stat().st_size),
        "build_seconds": round(time.perf_counter() - t0, 2),
        "fps": float(dataset.fps),
    }
    out_path.with_suffix(".json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    log(f"cache written: {out_path} ({meta['cache_bytes'] / 1e6:.1f} MB, "
        f"{meta['num_frames']} frames, {meta['build_seconds']}s)")
    return meta


def load_cache(path) -> dict:
    """Load an ``.npz`` cache plus its JSON metadata (``path`` -> dict)."""
    path = Path(path)
    arrays = dict(np.load(path, allow_pickle=False))
    meta_path = path.with_suffix(".json")
    meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
    arrays["meta"] = meta
    return arrays


def episode_slice(cache: dict, episode: int) -> slice:
    offsets = cache["episode_offsets"]
    return slice(int(offsets[episode]), int(offsets[episode + 1]))


def train_val_split(cache: dict, val_episodes: int) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(train_indices, val_indices)`` global frame indices."""
    offsets = cache["episode_offsets"]
    n_eps = len(offsets) - 1
    val_eps = set(range(n_eps - val_episodes, n_eps))
    all_idx = np.arange(int(offsets[-1]))
    ep_index = cache["episode_index"]
    is_val = np.isin(ep_index, list(val_eps))
    return all_idx[~is_val], all_idx[is_val]
