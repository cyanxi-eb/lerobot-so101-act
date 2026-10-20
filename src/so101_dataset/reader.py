"""Reader for the LeRobot-v3-style dataset written by :mod:`so101_dataset`."""

from __future__ import annotations

import json
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import pyarrow.parquet as pq

from . import layout


def read_jsonl(path: Path) -> list[dict]:
    """Read a JSON-lines file (UTF-8), returning ``[]`` when it is absent."""
    path = Path(path)
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


class LeRobotDataset:
    """Small, dependency-light reader for the dataset subset used here."""

    def __init__(self, root):
        self.root = Path(root)
        info_path = self.root / layout.INFO_FILE
        if not info_path.exists():
            raise FileNotFoundError(f"{info_path} not found; is this a dataset root?")
        self.info = json.loads(info_path.read_text(encoding="utf-8"))
        self.fps = float(self.info["fps"])
        self.features = self.info["features"]
        self.episodes = read_jsonl(self.root / layout.EPISODES_FILE)
        self.tasks = read_jsonl(self.root / layout.TASKS_FILE)
        self._task_by_index = {int(t["task_index"]): t["task"] for t in self.tasks}
        self._lengths = [int(ep["length"]) for ep in self.episodes]
        self._offsets = np.concatenate([[0], np.cumsum(self._lengths)]).astype(np.int64)
        self._cache: dict[int, dict] = {}

    # -- construction -----------------------------------------------------
    @classmethod
    def load(cls, root) -> "LeRobotDataset":
        return cls(root)

    # -- basic properties -------------------------------------------------
    @property
    def num_episodes(self) -> int:
        return len(self.episodes)

    @property
    def total_frames(self) -> int:
        return int(self._offsets[-1])

    def __len__(self) -> int:
        return self.total_frames

    @property
    def episode_lengths(self) -> list[int]:
        return list(self._lengths)

    # -- access -----------------------------------------------------------
    def get_episode(self, episode_index: int) -> dict:
        if not 0 <= episode_index < self.num_episodes:
            raise IndexError(episode_index)
        if episode_index in self._cache:
            return self._cache[episode_index]
        ep = self.episodes[episode_index]
        data_path = self.root / layout.DATA_PATH.format(chunk=0, episode=episode_index)
        table = pq.read_table(data_path).to_pydict()
        states = np.asarray(table["observation.state"], dtype=np.float32)
        actions = np.asarray(table["action"], dtype=np.float32)
        front = self._read_video("observation.images.front", episode_index)
        wrist = self._read_video("observation.images.wrist", episode_index)
        expected = int(ep["length"])
        if front.shape[0] != expected or wrist.shape[0] != expected:
            raise ValueError(
                f"episode {episode_index}: video length "
                f"{front.shape[0]}/{wrist.shape[0]} != metadata length {expected}"
            )
        task = self._task_by_index.get(int(table["task_index"][0]), "")
        result = {
            "episode_index": episode_index,
            "length": expected,
            "front": front,
            "wrist": wrist,
            "state": states,
            "action": actions,
            "task": task,
            "timestamp": np.asarray(table["timestamp"], dtype=np.float32),
            "frame_index": np.asarray(table["frame_index"], dtype=np.int64),
            "index": np.asarray(table["index"], dtype=np.int64),
            "next_done": np.asarray(table["next.done"], dtype=bool),
        }
        self._cache[episode_index] = result
        return result

    def get_frame(self, global_index: int) -> dict:
        if not 0 <= global_index < self.total_frames:
            raise IndexError(global_index)
        episode = int(np.searchsorted(self._offsets, global_index, side="right") - 1)
        offset = int(self._offsets[episode])
        frame = int(global_index - offset)
        data = self.get_episode(episode)
        return {
            "episode_index": episode,
            "frame_index": frame,
            "front": data["front"][frame],
            "wrist": data["wrist"][frame],
            "state": data["state"][frame],
            "action": data["action"][frame],
            "task": data["task"],
            "timestamp": float(data["timestamp"][frame]),
        }

    def iter_episodes(self):
        for i in range(self.num_episodes):
            yield self.get_episode(i)

    # -- internals --------------------------------------------------------
    def _read_video(self, video_key: str, episode_index: int) -> np.ndarray:
        path = self.root / layout.VIDEO_PATH.format(
            chunk=0, video_key=video_key, episode=episode_index
        )
        reader = imageio.get_reader(str(path), "ffmpeg")
        try:
            frames = [np.asarray(frame) for frame in reader]
        finally:
            reader.close()
        arr = np.stack(frames).astype(np.uint8)
        if arr.ndim != 4 or arr.shape[-1] != 3:
            raise ValueError(f"{path}: expected (T,H,W,3), got {arr.shape}")
        return arr
