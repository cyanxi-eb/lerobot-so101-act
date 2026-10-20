"""Feature schema and statistics helpers for the dataset.

The feature descriptions intentionally mirror LeRobot v3 ``meta/info.json``
field names so that tooling written for LeRobot can at least understand the
schema, even though this project does not depend on the ``lerobot`` package.
"""

from __future__ import annotations

from typing import Any

import numpy as np

STATE_NAMES = ("q1_base_yaw", "q2_shoulder", "q3_elbow", "q4_wrist", "q5_roll", "gripper")
STATE_SHAPE = [6]
ACTION_SHAPE = [6]

IMAGE_KEYS = ("observation.images.front", "observation.images.wrist")


def default_features(image_size: int = 96, fps: float = 7.5) -> dict[str, dict[str, Any]]:
    """Return the feature schema written into ``meta/info.json``."""
    return {
        "observation.state": {
            "dtype": "float32",
            "shape": STATE_SHAPE,
            "names": list(STATE_NAMES),
        },
        "observation.images.front": {
            "dtype": "video",
            "shape": [image_size, image_size, 3],
            "names": ["height", "width", "channels"],
            "info": {
                "video.fps": fps,
                "video.codec": "libx264",
                "video.pix_fmt": "yuv420p",
                "video.is_depth_map": False,
                "has_audio": False,
            },
        },
        "observation.images.wrist": {
            "dtype": "video",
            "shape": [image_size, image_size, 3],
            "names": ["height", "width", "channels"],
            "info": {
                "video.fps": fps,
                "video.codec": "libx264",
                "video.pix_fmt": "yuv420p",
                "video.is_depth_map": False,
                "has_audio": False,
            },
        },
        "action": {
            "dtype": "float32",
            "shape": ACTION_SHAPE,
            "names": list(STATE_NAMES),
        },
    }


class StatsAccumulator:
    """Streaming mean / std / min / max for vector features.

    Every feature keeps its own sample count: the state and action streams are
    accumulated independently, so their means must not share a global count.
    """

    def __init__(self) -> None:
        self.counts: dict[str, int] = {}
        self._sum: dict[str, np.ndarray] = {}
        self._sumsq: dict[str, np.ndarray] = {}
        self._min: dict[str, np.ndarray] = {}
        self._max: dict[str, np.ndarray] = {}

    def update(self, key: str, values) -> None:
        arr = np.asarray(values, dtype=np.float64)
        if arr.ndim == 1:
            arr = arr[None, :]
        self.counts[key] = self.counts.get(key, 0) + int(arr.shape[0])
        s = arr.sum(axis=0)
        ss = (arr * arr).sum(axis=0)
        if key not in self._sum:
            self._sum[key] = s
            self._sumsq[key] = ss
            self._min[key] = arr.min(axis=0)
            self._max[key] = arr.max(axis=0)
        else:
            self._sum[key] += s
            self._sumsq[key] += ss
            self._min[key] = np.minimum(self._min[key], arr.min(axis=0))
            self._max[key] = np.maximum(self._max[key], arr.max(axis=0))

    def as_dict(self) -> dict[str, dict[str, list[float]]]:
        out: dict[str, dict[str, list[float]]] = {}
        for key in self._sum:
            n = max(int(self.counts[key]), 1)
            mean = self._sum[key] / n
            var = np.maximum(self._sumsq[key] / n - mean * mean, 0.0)
            out[key] = {
                "mean": mean.round(6).tolist(),
                "std": np.sqrt(var).round(6).tolist(),
                "min": self._min[key].round(6).tolist(),
                "max": self._max[key].round(6).tolist(),
                "count": [int(self.counts[key])],
            }
        return out


def compute_stats(states, actions, front=None, wrist=None) -> dict:
    """Compute ``meta/stats.json`` style statistics for one dataset."""
    acc = StatsAccumulator()
    acc.update("observation.state", states)
    acc.update("action", actions)
    stats = acc.as_dict()
    # Image stats are per-channel scalars; LeRobot stores richer shapes but a
    # 3-value mean/std is enough for normalisation experiments here.
    for name, frames in (("observation.images.front", front),
                         ("observation.images.wrist", wrist)):
        if frames is None:
            continue
        arr = np.asarray(frames, dtype=np.float64) / 255.0
        mean = arr.reshape(-1, 3).mean(axis=0)
        std = arr.reshape(-1, 3).std(axis=0)
        stats[name] = {
            "mean": mean.round(6).tolist(),
            "std": std.round(6).tolist(),
            "min": [0.0, 0.0, 0.0],
            "max": [1.0, 1.0, 1.0],
            "count": [int(arr.shape[0])],
        }
    return stats
