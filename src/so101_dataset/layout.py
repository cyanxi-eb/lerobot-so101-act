"""On-disk layout of the LeRobot-v3-style dataset used by this project."""

from __future__ import annotations

from pathlib import Path

CODEBASE_VERSION = "v3.0"
ROBOT_TYPE = "so101_sim"
DEFAULT_FPS = 7.5
IMAGE_SIZE = 96

DATA_PATH = "data/chunk-{chunk:03d}/episode_{episode:06d}.parquet"
VIDEO_PATH = "videos/chunk-{chunk:03d}/{video_key}/episode_{episode:06d}.mp4"
META_DIR = "meta"
INFO_FILE = "meta/info.json"
EPISODES_FILE = "meta/episodes.jsonl"
TASKS_FILE = "meta/tasks.jsonl"
STATS_FILE = "meta/stats.json"
EPISODE_META_FILE = "meta/episodes_meta.jsonl"


def chunk_dir(root: Path, chunk: int = 0) -> Path:
    return Path(root) / "data" / f"chunk-{chunk:03d}"


def episode_data_path(root: Path, episode: int, chunk: int = 0) -> Path:
    return Path(root) / DATA_PATH.format(chunk=chunk, episode=episode)


def episode_video_path(root: Path, video_key: str, episode: int, chunk: int = 0) -> Path:
    return Path(root) / VIDEO_PATH.format(chunk=chunk, video_key=video_key, episode=episode)


def video_keys(features: dict) -> list[str]:
    return [k for k, v in features.items() if v.get("dtype") == "video"]


def state_keys(features: dict) -> list[str]:
    return [k for k, v in features.items() if v.get("dtype") in ("float32", "int64")
            and k.startswith(("observation.state", "action"))]
