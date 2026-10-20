"""Dataset writer / reader / cache roundtrip tests."""

from __future__ import annotations

import json

import numpy as np

from so101_dataset import (
    LeRobotDataset,
    LeRobotDatasetWriter,
    build_action_chunks,
    layout,
    resize_frames,
)


def test_metadata_files_are_written(mini_dataset):
    root = mini_dataset
    info = json.loads((root / layout.INFO_FILE).read_text(encoding="utf-8"))
    assert info["codebase_version"] == "v3.0"
    assert info["total_episodes"] == 2
    assert info["total_frames"] == 120
    assert info["fps"] == 7.5
    assert info["features"]["observation.images.front"]["dtype"] == "video"
    episodes = [json.loads(line) for line in
                (root / layout.EPISODES_FILE).read_text(encoding="utf-8").splitlines()]
    assert len(episodes) == 2
    assert episodes[0]["length"] == 60
    assert (root / layout.STATS_FILE).exists()
    assert (root / "data" / "chunk-000" / "episode_000000.parquet").exists()
    assert (root / "videos" / "chunk-000" / "observation.images.front"
            / "episode_000000.mp4").exists()


def test_reader_roundtrip_states_actions(mini_dataset):
    dataset = LeRobotDataset.load(mini_dataset)
    assert dataset.num_episodes == 2
    assert dataset.fps == 7.5
    assert len(dataset) == 120
    episode = dataset.get_episode(0)
    assert episode["state"].shape == (60, 6)
    assert episode["action"].shape == (60, 6)
    assert episode["task"] == "抓取积木并放进盒子"
    assert episode["front"].shape == (60, 96, 96, 3)
    assert episode["next_done"][-1]
    assert not episode["next_done"][0]
    assert np.allclose(episode["timestamp"][1] - episode["timestamp"][0], 1 / 7.5,
                       atol=1e-5)


def test_reader_global_frame_indexing(mini_dataset):
    dataset = LeRobotDataset.load(mini_dataset)
    frame = dataset.get_frame(60)  # first frame of episode 1
    assert frame["episode_index"] == 1
    assert frame["frame_index"] == 0
    last = dataset.get_frame(119)
    assert last["episode_index"] == 1
    assert last["frame_index"] == 59


def test_video_roundtrip_is_visually_close(tmp_path):
    """Encode smooth synthetic frames and check the decoded video is close."""
    writer = LeRobotDatasetWriter(tmp_path / "ds", fps=7.5, image_size=96)
    frames = np.zeros((6, 96, 96, 3), dtype=np.uint8)
    for t in range(6):
        frames[t, :, :, 0] = 20 * t
        frames[t, :, :, 1] = np.linspace(0, 200, 96)[None, :]
        frames[t, :, :, 2] = np.linspace(0, 120, 96)[:, None]
    states = np.zeros((6, 6), dtype=np.float32)
    actions = np.ones((6, 6), dtype=np.float32) * 0.5
    writer.add_episode(frames, frames, states, actions, task="t")
    writer.finalize()
    dataset = LeRobotDataset.load(tmp_path / "ds")
    decoded = dataset.get_episode(0)["front"]
    assert decoded.shape == frames.shape
    assert np.mean(np.abs(decoded.astype(int) - frames.astype(int))) < 12


def test_stats_are_consistent(mini_dataset):
    stats = json.loads((mini_dataset / layout.STATS_FILE).read_text(encoding="utf-8"))
    for key in ("observation.state", "action",
                "observation.images.front", "observation.images.wrist"):
        entry = stats[key]
        expected_dims = 6 if key in ("observation.state", "action") else 3
        assert len(entry["mean"]) == expected_dims
        assert len(entry["std"]) == expected_dims
        assert all(s >= 0 for s in entry["std"])
        assert all(lo <= mean <= hi for lo, mean, hi in
                   zip(entry["min"], entry["mean"], entry["max"]))


def test_resize_frames_shape():
    frames = np.random.default_rng(0).integers(0, 255, (3, 96, 96, 3), dtype=np.uint8)
    out = resize_frames(frames, 64)
    assert out.shape == (3, 64, 64, 3)
    assert out.dtype == np.uint8


def test_action_chunk_padding_and_mask():
    offsets = np.array([0, 5, 10])
    actions = np.arange(60, dtype=np.float32).reshape(10, 6)
    chunks, mask = build_action_chunks(actions, offsets, chunk_size=4)
    assert chunks.shape == (10, 4, 6)
    # First frame of episode 0 has 5 frames available -> 4 valid entries.
    assert mask[0].sum() == 4
    assert np.allclose(chunks[0, 0], actions[0])
    # Last frame of an episode: only one valid action, padded with itself.
    assert mask[4].sum() == 1
    assert np.allclose(chunks[4], np.repeat(actions[4][None], 4, axis=0))
    # Chunks never cross the episode boundary.
    assert np.allclose(chunks[4, :, :], actions[4])


def test_cache_roundtrip(mini_cache):
    cache = mini_cache
    assert cache["front"].shape[1:] == (64, 64, 3)
    assert cache["state"].shape[1] == 6
    assert cache["action_chunks"].shape[1] == 8
    assert cache["episode_offsets"][-1] == 120
    assert cache["meta"]["image_size"] == 64
    assert cache["meta"]["val_episodes"] == 1
    stats = cache["meta"]["action_stats"]
    assert len(stats["mean"]) == 6


def test_writer_rejects_bad_shapes(tmp_path):
    writer = LeRobotDatasetWriter(tmp_path / "bad", fps=7.5, image_size=96)
    with np.testing.assert_raises(ValueError):
        writer.add_episode(
            front=np.zeros((3, 96, 96, 3), dtype=np.uint8),
            wrist=np.zeros((3, 96, 96, 3), dtype=np.uint8),
            states=np.zeros((3, 5), dtype=np.float32),
            actions=np.zeros((3, 6), dtype=np.float32),
            task="bad",
        )
