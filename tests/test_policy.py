"""Policy / tokenizer / normalisation unit tests."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from so101_policy.act import ACTConfig, ACTPolicy, act_loss
from so101_policy.inference import LoadedPolicy, load_checkpoint, save_checkpoint
from so101_policy.lang_cond import (
    CharTokenizer,
    LangConditionedPolicy,
    LangConfig,
    lang_loss,
)
from so101_policy.normalization import Normalizer


def tiny_act_config() -> ACTConfig:
    return ACTConfig(image_size=64, d_model=64, n_heads=2, n_enc_layers=1,
                     n_dec_layers=1, chunk_size=8, latent_dim=16, dropout=0.0)


def random_obs(batch: int = 2, seed: int = 0):
    rng = np.random.default_rng(seed)
    images = torch.from_numpy(
        rng.integers(0, 255, (batch, 2, 3, 64, 64), dtype=np.uint8).astype(np.float32)
        / 255.0
    )
    state = torch.from_numpy(rng.normal(size=(batch, 6)).astype(np.float32))
    return images, state


def test_act_forward_shapes_and_parameter_budget():
    model = ACTPolicy(tiny_act_config())
    images, state = random_obs()
    target = torch.zeros(2, 8, 6)
    pred, mu, logvar = model(images, state, target)
    assert pred.shape == (2, 8, 6)
    assert mu.shape == (2, 16) and logvar.shape == (2, 16)
    assert model.n_parameters() < 5_000_000


def test_act_loss_is_finite_and_kl_zero_for_prior():
    model = ACTPolicy(tiny_act_config())
    images, state = random_obs()
    target = torch.zeros(2, 8, 6)
    pred, mu, logvar = model(images, state, target)
    mask = torch.ones(2, 8)
    loss, metrics = act_loss(pred, target, mask, mu, logvar, kl_weight=10.0)
    assert torch.isfinite(loss)
    assert metrics["l1"] >= 0.0
    zero = torch.zeros(2, 16)
    _, metrics_prior = act_loss(pred, target, mask, zero, zero, kl_weight=10.0)
    assert metrics_prior["kl"] == pytest.approx(0.0, abs=1e-6)


def test_act_predict_is_deterministic_and_chunked():
    model = ACTPolicy(tiny_act_config())
    model.eval()
    images, state = random_obs()
    a = model.predict(images, state)
    b = model.predict(images, state)
    assert a.shape == (2, 8, 6)
    assert torch.allclose(a, b)


def test_tokenizer_handles_chinese_instructions():
    tokenizer = CharTokenizer.fit(["把红色积木放进盒子里", "把蓝色积木放进盒子里"])
    assert tokenizer.vocab_size > 5
    ids = tokenizer.encode(["把红色积木放进盒子里", "把绿色积木放进盒子里"])
    assert ids.shape == (2, tokenizer.max_len)
    assert ids[0, 0] == tokenizer.stoi["把"]
    # Padding is 0 and the first row has no padding.
    assert ids[0].shape == (tokenizer.max_len,)
    assert (ids[0] == 0).sum() == tokenizer.max_len - 10
    # A colour that never appeared in the corpus falls back to <unk>.
    assert tokenizer.stoi.get("黄") is None
    unknown = tokenizer.encode(["把黄色积木放进盒子里"])
    assert unknown[0, 1] == 1
    assert tokenizer.itos[0] == "<pad>"


def test_lang_policy_forward_shapes():
    tokenizer = CharTokenizer.fit(["把红色积木放进盒子里", "把蓝色积木放进盒子里"])
    config = LangConfig(image_size=64, d_model=64, n_heads=2, n_layers=1,
                        chunk_size=8, dropout=0.0)
    model = LangConditionedPolicy(tokenizer.vocab_size, config)
    images, state = random_obs()
    ids = torch.from_numpy(tokenizer.encode(["把红色积木放进盒子里"] * 2)).long()
    pred = model(images, state, ids)
    assert pred.shape == (2, 8, 6)
    assert model.n_parameters() < 5_000_000
    loss, metrics = lang_loss(pred, torch.zeros(2, 8, 6), torch.ones(2, 8))
    assert torch.isfinite(loss) and metrics["l1"] >= 0


def test_language_changes_the_prediction():
    tokenizer = CharTokenizer.fit(["把红色积木放进盒子里", "把蓝色积木放进盒子里"])
    config = LangConfig(image_size=64, d_model=64, n_heads=2, n_layers=1,
                        chunk_size=8, dropout=0.0)
    model = LangConditionedPolicy(tokenizer.vocab_size, config).eval()
    images, state = random_obs()
    red = torch.from_numpy(tokenizer.encode(["把红色积木放进盒子里"] * 2)).long()
    blue = torch.from_numpy(tokenizer.encode(["把蓝色积木放进盒子里"] * 2)).long()
    with torch.no_grad():
        pred_red = model(images, state, red)
        pred_blue = model(images, state, blue)
    assert not torch.allclose(pred_red, pred_blue, atol=1e-7)


def test_normalizer_roundtrip():
    rng = np.random.default_rng(0)
    state = rng.normal(size=(10, 6)).astype(np.float32)
    action = rng.normal(size=(10, 6)).astype(np.float32)
    norm = Normalizer(state.mean(0), state.std(0), action.mean(0), action.std(0))
    assert np.allclose(norm.denormalize_action(norm.normalize_action(action)), action,
                       atol=1e-5)
    payload = norm.to_dict()
    restored = Normalizer.from_dict(payload)
    assert np.allclose(restored.normalize_state(state), norm.normalize_state(state))


def test_checkpoint_save_load_roundtrip(tmp_path):
    model = ACTPolicy(tiny_act_config())
    rng = np.random.default_rng(0)
    normalizer = Normalizer(rng.normal(size=6), np.ones(6) * 0.5,
                            rng.normal(size=6), np.ones(6) * 0.4)
    path = tmp_path / "act.pt"
    info = save_checkpoint(path, model, normalizer, "act", 64, 8,
                           config=tiny_act_config().to_dict(), metadata={"epoch": 1})
    assert info["bytes"] < 40e6
    loaded = load_checkpoint(path)
    assert loaded.policy_type == "act"
    assert loaded.chunk_size == 8
    obs = {
        "front": np.zeros((96, 96, 3), dtype=np.uint8),
        "wrist": np.zeros((96, 96, 3), dtype=np.uint8),
        "state": np.zeros(6, dtype=np.float32),
    }
    chunk = loaded.act(obs)
    assert chunk.shape == (8, 6)
    assert np.all(np.isfinite(chunk))
