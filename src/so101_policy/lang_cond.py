"""Lightweight language-conditioned policy ("SmolVLA-style", scaled down).

This is *not* SmolVLA.  SmolVLA is a pretrained 450M-parameter VLA.  This
module implements the same high-level idea at a size that trains from scratch
on a CPU-only budget:

* the same small CNN vision backbone as the ACT policy,
* a small trainable **character-level** token embedding (works for Chinese
  instructions such as "把红色积木放进盒子里" without any external model),
  mean-pooled and projected into the transformer token space,
* a transformer encoder over ``[readout, vision tokens, state token, language
  token]`` predicting a chunk of future actions.

The language embedding is deliberately trainable rather than hash-based so
the model can learn which colour token matters; the vocabulary is built from
the instruction corpus and stored inside the checkpoint.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import torch
import torch.nn as nn

from .act import VisionBackbone, _transformer_layer

PAD_ID = 0
UNK_ID = 1
MAX_TEXT_LEN = 16


class CharTokenizer:
    """Character-level tokenizer (CJK-friendly) with a fixed vocabulary.

    Whitespace is used only as a separator; every other character becomes a
    token.  This is intentionally naive: it needs no download and handles
    mixed Chinese / English instruction strings.
    """

    def __init__(self, itos: list[str], max_len: int = MAX_TEXT_LEN):
        if itos[:2] != ["<pad>", "<unk>"]:
            raise ValueError("itos must start with <pad>, <unk>")
        self.itos = list(itos)
        self.stoi = {token: i for i, token in enumerate(self.itos)}
        self.max_len = int(max_len)

    # -- construction -----------------------------------------------------
    @classmethod
    def fit(cls, texts, max_len: int = MAX_TEXT_LEN) -> "CharTokenizer":
        vocab: set[str] = set()
        for text in texts:
            vocab.update(t for t in str(text) if not t.isspace())
        return cls(["<pad>", "<unk>"] + sorted(vocab), max_len=max_len)

    @property
    def vocab_size(self) -> int:
        return len(self.itos)

    # -- encoding ---------------------------------------------------------
    def encode(self, texts, max_len: int | None = None) -> np.ndarray:
        if isinstance(texts, str):
            texts = [texts]
        length = int(max_len or self.max_len)
        out = np.zeros((len(texts), length), dtype=np.int64)
        for row, text in enumerate(texts):
            tokens = [t for t in str(text) if not t.isspace()][:length]
            for col, token in enumerate(tokens):
                out[row, col] = self.stoi.get(token, UNK_ID)
        return out

    def decode(self, ids) -> str:
        return "".join(self.itos[int(i)] for i in ids if int(i) > UNK_ID)

    # -- serialisation ----------------------------------------------------
    def to_dict(self) -> dict:
        return {"itos": self.itos, "max_len": self.max_len}

    @classmethod
    def from_dict(cls, payload: dict) -> "CharTokenizer":
        return cls(payload["itos"], max_len=int(payload["max_len"]))


@dataclass
class LangConfig:
    image_size: int = 64
    n_cameras: int = 2
    state_dim: int = 6
    action_dim: int = 6
    chunk_size: int = 20
    d_model: int = 128
    n_heads: int = 4
    n_layers: int = 2
    text_dim: int = 32
    dropout: float = 0.1
    max_text_len: int = MAX_TEXT_LEN
    cnn_channels: tuple[int, int, int, int] = (32, 64, 64, 64)

    def to_dict(self) -> dict:
        return asdict(self)


class LangConditionedPolicy(nn.Module):
    """Vision + language + proprioception -> action chunk."""

    def __init__(self, vocab_size: int, config: LangConfig | None = None):
        super().__init__()
        self.config = config or LangConfig()
        cfg = self.config
        tokens_per_camera = (cfg.image_size // 16) ** 2
        self.tokens_per_camera = tokens_per_camera
        self.backbone = VisionBackbone(
            in_channels=3,
            channels=cfg.cnn_channels,
            d_model=cfg.d_model,
            tokens_per_camera=tokens_per_camera,
        )
        self.camera_embed = nn.Parameter(torch.zeros(1, cfg.n_cameras, 1, cfg.d_model))
        nn.init.normal_(self.camera_embed, std=0.02)
        self.n_vis = cfg.n_cameras * tokens_per_camera

        self.state_proj = nn.Sequential(
            nn.Linear(cfg.state_dim, cfg.d_model),
            nn.LayerNorm(cfg.d_model),
        )
        self.text_embed = nn.Embedding(vocab_size, cfg.text_dim, padding_idx=PAD_ID)
        self.text_proj = nn.Sequential(
            nn.Linear(cfg.text_dim, cfg.d_model),
            nn.Tanh(),
            nn.Linear(cfg.d_model, cfg.d_model),
        )
        self.readout = nn.Parameter(torch.zeros(1, 1, cfg.d_model))
        nn.init.normal_(self.readout, std=0.02)
        layer = _transformer_layer(cfg.d_model, cfg.n_heads, cfg.dropout)
        self.encoder = nn.TransformerEncoder(layer, num_layers=cfg.n_layers,
                                             enable_nested_tensor=False)
        self.action_head = nn.Sequential(
            nn.LayerNorm(cfg.d_model),
            nn.Linear(cfg.d_model, cfg.chunk_size * cfg.action_dim),
        )

    def encode_obs(self, images: torch.Tensor, state: torch.Tensor,
                   token_ids: torch.Tensor) -> torch.Tensor:
        b, n_cam, c, h, w = images.shape
        flat = images.reshape(b * n_cam, c, h, w)
        tokens = self.backbone(flat).reshape(b, n_cam, self.tokens_per_camera, -1)
        tokens = tokens + self.camera_embed[:, :n_cam]
        vis = tokens.reshape(b, self.n_vis, -1)
        state_token = self.state_proj(state).unsqueeze(1)
        mask = (token_ids != PAD_ID).float().unsqueeze(-1)
        embedded = self.text_embed(token_ids)
        pooled = (embedded * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
        text_token = self.text_proj(pooled).unsqueeze(1)
        readout = self.readout.expand(b, -1, -1)
        return torch.cat([readout, vis, state_token, text_token], dim=1)

    def forward(self, images, state, token_ids) -> torch.Tensor:
        tokens = self.encode_obs(images, state, token_ids)
        encoded = self.encoder(tokens)
        flat = self.action_head(encoded[:, 0])
        return flat.reshape(-1, self.config.chunk_size, self.config.action_dim)

    @torch.no_grad()
    def predict(self, images, state, token_ids) -> torch.Tensor:
        self.eval()
        return self.forward(images, state, token_ids)

    def n_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())


def lang_loss(pred, target, mask) -> tuple[torch.Tensor, dict]:
    """Masked L1 loss for deterministic chunk prediction."""
    abs_err = (pred - target).abs().mean(dim=-1)
    loss = (abs_err * mask).sum() / mask.sum().clamp_min(1.0)
    return loss, {"loss": float(loss.detach()), "l1": float(loss.detach())}
