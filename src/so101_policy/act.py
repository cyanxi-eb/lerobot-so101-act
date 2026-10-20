"""ACT-style action-chunking transformer policy (PyTorch, CPU friendly).

This is a compact re-implementation of the ACT recipe from
`Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware`
(Zhao et al., RSS 2023):

* a small CNN vision backbone encodes the two 96x96 camera streams,
* a transformer encoder consumes the vision / proprioception tokens,
* a CVAE-style latent variable ``z`` is inferred from the ground-truth action
  chunk during training and set to zero at inference,
* a transformer decoder (implemented here as extra self-attention blocks plus
  a readout token) predicts a chunk of future actions in one forward pass.

It is *not* the official LeRobot implementation: the encoder/decoder are
single-stack transformers with a learned readout token instead of ACT's
cross-attention decoder.  The essential ingredients (action chunking, CVAE
latent, visual backbone) are kept, and the parameter count is small enough to
train on a CPU-only budget.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class ACTConfig:
    image_size: int = 64
    n_cameras: int = 2
    state_dim: int = 6
    action_dim: int = 6
    chunk_size: int = 20
    d_model: int = 128
    n_heads: int = 4
    n_enc_layers: int = 2
    n_dec_layers: int = 2
    latent_dim: int = 32
    dropout: float = 0.1
    cnn_channels: tuple[int, int, int, int] = (32, 64, 64, 64)
    kl_weight: float = 10.0

    def to_dict(self) -> dict:
        return asdict(self)


def _transformer_layer(d_model: int, n_heads: int, dropout: float) -> nn.TransformerEncoderLayer:
    return nn.TransformerEncoderLayer(
        d_model=d_model,
        nhead=n_heads,
        dim_feedforward=2 * d_model,
        dropout=dropout,
        activation="gelu",
        batch_first=True,
        norm_first=True,
    )


class VisionBackbone(nn.Module):
    """Shared small CNN: 2 cameras x 64x64x3 -> 2 x 4x4 spatial tokens."""

    def __init__(self, in_channels: int = 3, channels=(32, 64, 64, 64),
                 d_model: int = 128, tokens_per_camera: int = 16):
        super().__init__()
        c1, c2, c3, c4 = channels
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, c1, 5, stride=2, padding=2, bias=False),
            nn.BatchNorm2d(c1),
            nn.ReLU(inplace=True),
            nn.Conv2d(c1, c2, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(c2),
            nn.ReLU(inplace=True),
            nn.Conv2d(c2, c3, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(c3),
            nn.ReLU(inplace=True),
            nn.Conv2d(c3, c4, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(c4),
            nn.ReLU(inplace=True),
        )
        self.project = nn.Linear(c4, d_model)
        self.spatial_pos = nn.Parameter(torch.zeros(1, tokens_per_camera, d_model))
        nn.init.normal_(self.spatial_pos, std=0.02)
        self.tokens_per_camera = tokens_per_camera

    @property
    def num_tokens(self) -> int:
        return self.tokens_per_camera

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """``images`` (B, C, H, W) -> tokens (B, tokens, d_model)."""
        features = self.conv(images)  # (B, c4, 4, 4)
        tokens = features.flatten(2).transpose(1, 2)  # (B, 16, c4)
        return self.project(tokens) + self.spatial_pos


class ACTPolicy(nn.Module):
    """Action-chunking transformer with a CVAE latent."""

    def __init__(self, config: ACTConfig | None = None):
        super().__init__()
        self.config = config or ACTConfig()
        cfg = self.config
        # 64x64 -> 4x4 feature map -> 16 tokens per camera.
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
        n_vis = cfg.n_cameras * tokens_per_camera
        self.n_vis = n_vis
        self.state_proj = nn.Sequential(
            nn.Linear(cfg.state_dim, cfg.d_model),
            nn.LayerNorm(cfg.d_model),
        )
        self.readout = nn.Parameter(torch.zeros(1, 1, cfg.d_model))
        nn.init.normal_(self.readout, std=0.02)

        # --- CVAE encoder (training only) ---------------------------------
        self.z_token = nn.Parameter(torch.zeros(1, 1, cfg.d_model))
        self.action_proj = nn.Linear(cfg.chunk_size * cfg.action_dim, cfg.d_model)
        enc_layer = _transformer_layer(cfg.d_model, cfg.n_heads, cfg.dropout)
        self.cvae_encoder = nn.TransformerEncoder(enc_layer, num_layers=cfg.n_enc_layers,
                                                  enable_nested_tensor=False)
        self.mu_head = nn.Linear(cfg.d_model, cfg.latent_dim)
        self.logvar_head = nn.Linear(cfg.d_model, cfg.latent_dim)
        self.z_proj = nn.Linear(cfg.latent_dim, cfg.d_model)

        # --- policy decoder ------------------------------------------------
        dec_layer = _transformer_layer(cfg.d_model, cfg.n_heads, cfg.dropout)
        self.decoder = nn.TransformerEncoder(dec_layer, num_layers=cfg.n_dec_layers,
                                             enable_nested_tensor=False)
        self.action_head = nn.Sequential(
            nn.LayerNorm(cfg.d_model),
            nn.Linear(cfg.d_model, cfg.chunk_size * cfg.action_dim),
        )

    # -- observation encoding ---------------------------------------------
    def encode_obs(self, images: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        """images (B, n_cameras, 3, H, W), state (B, state_dim) -> tokens."""
        b, n_cam, c, h, w = images.shape
        flat = images.reshape(b * n_cam, c, h, w)
        tokens = self.backbone(flat).reshape(b, n_cam, self.backbone.num_tokens, -1)
        tokens = tokens + self.camera_embed[:, :n_cam]
        vis = tokens.reshape(b, self.n_vis, -1)
        state_token = self.state_proj(state).unsqueeze(1)
        return torch.cat([vis, state_token], dim=1)  # (B, n_vis + 1, d)

    # -- CVAE --------------------------------------------------------------
    def encode_latent(self, obs_tokens: torch.Tensor, action_chunk: torch.Tensor):
        b = obs_tokens.shape[0]
        action_token = self.action_proj(
            action_chunk.reshape(b, self.config.chunk_size * self.config.action_dim)
        ).unsqueeze(1)
        z_tok = self.z_token.expand(b, -1, -1)
        tokens = torch.cat([z_tok, obs_tokens, action_token], dim=1)
        encoded = self.cvae_encoder(tokens)
        h = encoded[:, 0]
        mu = self.mu_head(h)
        logvar = self.logvar_head(h).clamp(-8.0, 8.0)
        return mu, logvar

    @staticmethod
    def reparameterize(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    # -- decoder -----------------------------------------------------------
    def decode(self, obs_tokens: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        b = obs_tokens.shape[0]
        z_token = self.z_proj(z).unsqueeze(1)
        readout = self.readout.expand(b, -1, -1)
        tokens = torch.cat([readout, z_token, obs_tokens], dim=1)
        out = self.decoder(tokens)
        flat = self.action_head(out[:, 0])
        return flat.reshape(b, self.config.chunk_size, self.config.action_dim)

    # -- public API --------------------------------------------------------
    def forward(self, images, state, action_chunk=None):
        """Training forward: returns ``(pred, mu, logvar)``."""
        obs_tokens = self.encode_obs(images, state)
        if action_chunk is None:
            z = torch.zeros(obs_tokens.shape[0], self.config.latent_dim,
                            device=obs_tokens.device, dtype=obs_tokens.dtype)
            mu = logvar = z
        else:
            mu, logvar = self.encode_latent(obs_tokens, action_chunk)
            z = self.reparameterize(mu, logvar)
        pred = self.decode(obs_tokens, z)
        return pred, mu, logvar

    @torch.no_grad()
    def predict(self, images, state) -> torch.Tensor:
        """Inference: CVAE prior is N(0, I) so ``z = 0`` (no sampling)."""
        self.eval()
        obs_tokens = self.encode_obs(images, state)
        z = torch.zeros(obs_tokens.shape[0], self.config.latent_dim,
                        device=obs_tokens.device, dtype=obs_tokens.dtype)
        return self.decode(obs_tokens, z)

    def n_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())


def act_loss(pred, target, mask, mu, logvar, kl_weight: float = 10.0) -> tuple[torch.Tensor, dict]:
    """Masked L1 reconstruction loss + KL divergence to N(0, I)."""
    abs_err = (pred - target).abs().mean(dim=-1)  # (B, chunk)
    rec = (abs_err * mask).sum() / mask.sum().clamp_min(1.0)
    kl = -0.5 * (1.0 + logvar - mu.pow(2) - logvar.exp()).mean()
    loss = rec + kl_weight * kl
    return loss, {
        "loss": float(loss.detach()),
        "l1": float(rec.detach()),
        "kl": float(kl.detach()),
    }


def build_act(config: ACTConfig | None = None) -> ACTPolicy:
    return ACTPolicy(config)
