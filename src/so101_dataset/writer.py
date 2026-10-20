"""LeRobot-v3-style dataset writer (subset implementation).

Layout written by this writer::

    <root>/
      meta/info.json
      meta/episodes.jsonl
      meta/episodes_meta.jsonl      # extra scene metadata (subset extension)
      meta/tasks.jsonl
      meta/stats.json
      data/chunk-000/episode_000000.parquet
      videos/chunk-000/observation.images.front/episode_000000.mp4
      videos/chunk-000/observation.images.wrist/episode_000000.mp4

Differences from the official LeRobot v3 dataset are documented in
``docs/dataset_format.md``.
"""

from __future__ import annotations

import json
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from . import layout
from .stats import StatsAccumulator, default_features

VIDEO_KEYS = ("observation.images.front", "observation.images.wrist")


class LeRobotDatasetWriter:
    """Streaming writer for fixed-fps, fixed-length robot episodes."""

    def __init__(self, root, fps: float = layout.DEFAULT_FPS,
                 image_size: int = layout.IMAGE_SIZE, robot_type: str = layout.ROBOT_TYPE,
                 chunk_size: int = 1000):
        self.root = Path(root)
        self.fps = float(fps)
        self.dt = 1.0 / self.fps
        self.image_size = int(image_size)
        self.robot_type = robot_type
        self.chunk_size = int(chunk_size)
        self.features = default_features(self.image_size, self.fps)
        self.episodes: list[dict] = []
        self.episodes_meta: list[dict] = []
        self.tasks: list[str] = []
        self._task_index: dict[str, int] = {}
        self._stats = StatsAccumulator()
        self._image_acc = {
            "observation.images.front": [np.zeros(3), np.zeros(3), 0],
            "observation.images.wrist": [np.zeros(3), np.zeros(3), 0],
        }
        self.total_frames = 0
        self._closed = False
        self.root.mkdir(parents=True, exist_ok=True)
        (self.root / "meta").mkdir(parents=True, exist_ok=True)

    # -- task bookkeeping -------------------------------------------------
    def task_index(self, task: str) -> int:
        if task not in self._task_index:
            self._task_index[task] = len(self.tasks)
            self.tasks.append(task)
        return self._task_index[task]

    # -- writing ----------------------------------------------------------
    def add_episode(self, front, wrist, states, actions, task: str,
                    scene_meta: dict | None = None) -> int:
        """Append one episode. Returns the episode index."""
        front = np.asarray(front, dtype=np.uint8)
        wrist = np.asarray(wrist, dtype=np.uint8)
        states = np.asarray(states, dtype=np.float32)
        actions = np.asarray(actions, dtype=np.float32)
        length = int(states.shape[0])
        if states.shape != (length, 6) or actions.shape != (length, 6):
            raise ValueError(f"state/action must be (T, 6), got {states.shape} / {actions.shape}")
        expected = (length, self.image_size, self.image_size, 3)
        if front.shape != expected or wrist.shape != expected:
            raise ValueError(f"images must be {expected}, got {front.shape} / {wrist.shape}")
        if length <= 0:
            raise ValueError("empty episode")

        episode_index = len(self.episodes)
        task_idx = self.task_index(task)
        data_path = layout.episode_data_path(self.root, episode_index)
        data_path.parent.mkdir(parents=True, exist_ok=True)

        frame_index = np.arange(length, dtype=np.int64)
        index = np.arange(self.total_frames, self.total_frames + length, dtype=np.int64)
        timestamps = (frame_index / self.fps).astype(np.float32)
        next_done = np.zeros(length, dtype=bool)
        next_done[-1] = True

        table = pa.table(
            {
                "observation.state": pa.FixedSizeListArray.from_arrays(
                    pa.array(states.reshape(-1), type=pa.float32()), 6),
                "action": pa.FixedSizeListArray.from_arrays(
                    pa.array(actions.reshape(-1), type=pa.float32()), 6),
                "timestamp": pa.array(timestamps, type=pa.float32()),
                "frame_index": pa.array(frame_index, type=pa.int64()),
                "episode_index": pa.array(np.full(length, episode_index, dtype=np.int64)),
                "index": pa.array(index, type=pa.int64()),
                "task_index": pa.array(np.full(length, task_idx, dtype=np.int64)),
                "next.done": pa.array(next_done),
            }
        )
        pq.write_table(table, data_path, compression="snappy")

        for video_key, frames in zip(VIDEO_KEYS, (front, wrist)):
            video_path = layout.episode_video_path(self.root, video_key, episode_index)
            video_path.parent.mkdir(parents=True, exist_ok=True)
            imageio.mimwrite(
                str(video_path),
                list(frames),
                fps=self.fps,
                codec="libx264",
                macro_block_size=1,
                ffmpeg_params=["-crf", "18", "-preset", "veryfast"],
            )
            acc = self._image_acc[video_key]
            norm = frames.astype(np.float64) / 255.0
            flat = norm.reshape(-1, 3)
            acc[0] += flat.sum(axis=0)
            acc[1] += (flat * flat).sum(axis=0)
            acc[2] += int(flat.shape[0])

        self._stats.update("observation.state", states)
        self._stats.update("action", actions)
        self.episodes.append(
            {
                "episode_index": episode_index,
                "tasks": [task],
                "length": length,
                "dataset_from_index": int(index[0]),
                "dataset_to_index": int(index[-1]) + 1,
                "data/chunk_index": 0,
                "data/file_index": episode_index,
                "meta/episodes_index": episode_index,
            }
        )
        meta = {"episode_index": episode_index, "task_index": task_idx, "task": task}
        if scene_meta:
            meta["scene"] = scene_meta
        self.episodes_meta.append(meta)
        self.total_frames += length
        return episode_index

    # -- finalisation -----------------------------------------------------
    def finalize(self) -> dict:
        if self._closed:
            raise RuntimeError("dataset already finalised")
        if not self.episodes:
            raise RuntimeError("no episodes were added")
        stats = self._stats.as_dict()
        for video_key, (s, ss, count) in self._image_acc.items():
            n = max(int(count), 1)
            mean = s / n
            var = np.maximum(ss / n - mean * mean, 0.0)
            stats[video_key] = {
                "mean": mean.round(6).tolist(),
                "std": np.sqrt(var).round(6).tolist(),
                "min": [0.0, 0.0, 0.0],
                "max": [1.0, 1.0, 1.0],
                "count": [int(count)],
            }

        info = {
            "codebase_version": layout.CODEBASE_VERSION,
            "robot_type": self.robot_type,
            "total_episodes": len(self.episodes),
            "total_frames": int(self.total_frames),
            "total_tasks": len(self.tasks),
            "total_videos": len(self.episodes) * len(VIDEO_KEYS),
            "total_chunks": 1,
            "chunks_size": self.chunk_size,
            "fps": self.fps,
            "dt": round(self.dt, 6),
            "image_size": self.image_size,
            "splits": {"train": f"0:{len(self.episodes)}"},
            "data_path": layout.DATA_PATH,
            "video_path": layout.VIDEO_PATH,
            "features": self.features,
            "generation": {
                "simulator": "so101_sim (custom, no hardware)",
                "expert": "scripted Cartesian state machine with analytic IK",
                "domain_randomization": [
                    "initial arm pose",
                    "block colour / size / pose / yaw",
                    "box placement",
                    "front camera extrinsics + fov ('mounting error')",
                    "wrist camera offset",
                    "table / box colour and brightness",
                    "optional tall occluder between front camera and target",
                    "gripper slip probability and grasp tolerance",
                ],
                "hardware_teleoperation": False,
            },
        }
        self._write_json(layout.INFO_FILE, info)
        self._write_jsonl("meta/episodes.jsonl", self.episodes)
        self._write_jsonl("meta/episodes_meta.jsonl", self.episodes_meta)
        self._write_jsonl("meta/tasks.jsonl",
                          [{"task_index": i, "task": t} for i, t in enumerate(self.tasks)])
        self._write_json(layout.STATS_FILE, stats)
        self._closed = True
        return info

    # -- helpers ----------------------------------------------------------
    def _write_json(self, rel: str, payload) -> None:
        path = self.root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    def _write_jsonl(self, rel: str, rows) -> None:
        path = self.root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
