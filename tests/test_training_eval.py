"""Training-loop / evaluation smoke tests (CPU, tiny models)."""

from __future__ import annotations

import numpy as np
import torch

from so101_policy.act import ACTConfig, ACTPolicy, act_loss
from so101_policy.rollout import action_joint_limits_ok, run_rollout, summarise_rollouts
from so101_policy.training import (
    make_batches,
    normalizer_for,
    prepare_arrays,
    prepare_batch,
    train_val_indices,
    validate_l1,
)
from so101_sim import EnvConfig, GraspEnv


def tiny_config() -> ACTConfig:
    return ACTConfig(image_size=64, d_model=32, n_heads=2, n_enc_layers=1,
                     n_dec_layers=1, chunk_size=8, latent_dim=8, dropout=0.0)


def test_prepare_arrays_and_split(mini_cache):
    data = prepare_arrays(mini_cache)
    train_idx, val_idx = train_val_indices(data, val_episodes=1)
    assert len(train_idx) == 60 and len(val_idx) == 60
    assert set(train_idx).isdisjoint(set(val_idx))
    batch = list(make_batches(train_idx, batch_size=4, rng=np.random.default_rng(0)))
    assert len(batch) == 15
    images, state, target, mask = prepare_batch(data, batch[0], torch.device("cpu"))
    assert images.shape == (4, 2, 3, 64, 64)
    assert state.shape == (4, 6)
    assert target.shape == (4, 8, 6)
    assert mask.shape == (4, 8)
    assert float(images.max()) <= 1.0


def test_one_training_step_is_finite(mini_cache):
    data = prepare_arrays(mini_cache)
    train_idx, val_idx = train_val_indices(data, val_episodes=1)
    model = ACTPolicy(tiny_config())
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    for batch_idx in list(make_batches(train_idx, 8))[:2]:
        images, state, target, mask = prepare_batch(data, batch_idx, torch.device("cpu"))
        pred, mu, logvar = model(images, state, target)
        loss, metrics = act_loss(pred, target, mask, mu, logvar)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        assert np.isfinite(metrics["loss"])
    val_l1 = validate_l1(model, data, val_idx, torch.device("cpu"))
    assert np.isfinite(val_l1) and val_l1 > 0


def test_normalizer_from_cache_meta(mini_cache):
    normalizer = normalizer_for(mini_cache)
    assert normalizer.action_mean.shape == (6,)
    assert np.all(normalizer.action_std > 0)


class ZeroPolicy:
    """A deliberately bad policy used to smoke-test the rollout plumbing."""

    def __init__(self, chunk_size: int = 8):
        self.chunk_size = chunk_size

    def act(self, obs, instruction=None):
        return np.zeros((self.chunk_size, 6), dtype=np.float32)


def test_rollout_and_summary_smoke():
    env = GraspEnv(EnvConfig(max_steps=6))
    result = run_rollout(ZeroPolicy(), env, seed=11, split="seen", max_steps=6,
                         capture_last=True)
    assert result["steps"] == 6
    assert result["success"] is False
    assert result["failure"] in ("timeout", "grasp_miss")
    assert "last_obs" in result
    assert action_joint_limits_ok(np.zeros((4, 6), dtype=np.float32))
    summary = summarise_rollouts([result])
    assert summary["trials"] == 1
    assert summary["success_rate"] == 0.0
    assert summary["latency_ms_mean"] is not None
