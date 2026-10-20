"""Pytest configuration and shared fixtures.

The tests run against the source tree directly (``src`` layout) so that no
editable install is required in CI.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


@pytest.fixture(scope="session")
def mini_dataset(tmp_path_factory):
    """A 2-episode dataset written with the real writer (videos included)."""
    from so101_dataset import LeRobotDatasetWriter
    from so101_sim import EnvConfig, GraspEnv
    from so101_sim.expert import rollout_expert

    root = tmp_path_factory.mktemp("mini_dataset")
    env = GraspEnv(EnvConfig(max_steps=60, image_size=96))
    writer = LeRobotDatasetWriter(root, fps=env.config.fps, image_size=96)
    for episode in range(2):
        env.reset(seed=200 + episode)
        result = rollout_expert(env)
        assert result["success"], f"scripted expert failed for fixture seed {episode}"
        writer.add_episode(
            front=result["front"],
            wrist=result["wrist"],
            states=result["state"],
            actions=result["action"],
            task="抓取积木并放进盒子",
            scene_meta=result["scene"],
        )
    writer.finalize()
    return root


@pytest.fixture(scope="session")
def mini_cache(mini_dataset, tmp_path_factory):
    """Training cache built from :func:`mini_dataset` at 64x64."""
    from so101_dataset import build_cache, load_cache

    path = tmp_path_factory.mktemp("cache") / "mini_cache.npz"
    build_cache(mini_dataset, path, image_size=64, chunk_size=8, val_episodes=1,
                log=lambda *a, **k: None)
    return load_cache(path)
