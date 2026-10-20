"""Policy inference utilities: checkpoint loading, observation preprocessing.

Every checkpoint stores the model weights, the config, the *training* action /
state normalisation statistics and (for the language policy) the instruction
tokenizer.  Loading them together is what keeps the evaluation environment
consistent with training.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from .act import ACTConfig, ACTPolicy
from .lang_cond import CharTokenizer, LangConditionedPolicy, LangConfig
from .normalization import Normalizer


def resolve_device(device: str | None = None) -> torch.device:
    if device:
        return torch.device(device)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@dataclass
class LoadedPolicy:
    """A policy plus everything needed to run it on an observation."""

    policy: torch.nn.Module
    normalizer: Normalizer
    policy_type: str
    image_size: int
    chunk_size: int
    device: torch.device
    tokenizer: CharTokenizer | None = None
    config: dict = field(default_factory=dict)
    metadata: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.policy.eval()
        self.normalizer.to_torch(self.device)

    # -- preprocessing ----------------------------------------------------
    def _obs_to_tensors(self, obs: dict) -> tuple[torch.Tensor, torch.Tensor]:
        front = np.asarray(obs["front"], dtype=np.uint8)
        wrist = np.asarray(obs["wrist"], dtype=np.uint8)
        frames = np.stack([front, wrist], axis=0)  # (2, H, W, 3)
        x = torch.from_numpy(frames).to(self.device).float().permute(0, 3, 1, 2) / 255.0
        if x.shape[-1] != self.image_size or x.shape[-2] != self.image_size:
            x = F.interpolate(x, size=(self.image_size, self.image_size),
                              mode="bilinear", align_corners=False)
        images = x.unsqueeze(0)  # (1, 2, 3, H, W)
        state = torch.from_numpy(np.asarray(obs["state"], dtype=np.float32)).to(self.device)
        state = self.normalizer.torch_normalize_state(state).unsqueeze(0)
        return images, state

    # -- inference --------------------------------------------------------
    @torch.no_grad()
    def act(self, obs: dict, instruction: str | None = None) -> np.ndarray:
        """Return one denormalised action chunk of shape ``(chunk, 6)``."""
        images, state = self._obs_to_tensors(obs)
        if self.policy_type == "lang":
            text = instruction if instruction is not None else obs.get("instruction", "")
            if self.tokenizer is None:
                raise RuntimeError("language checkpoint is missing its tokenizer")
            ids = self.tokenizer.encode([text])
            pred = self.policy.predict(images, state,
                                       torch.from_numpy(ids).to(self.device))
        else:
            pred = self.policy.predict(images, state)
        pred = self.normalizer.torch_denormalize_action(pred[0])
        return pred.detach().cpu().numpy().astype(np.float32)

    def n_parameters(self) -> int:
        return sum(p.numel() for p in self.policy.parameters())


# ---------------------------------------------------------------------------
# Checkpoint IO
# ---------------------------------------------------------------------------
def save_checkpoint(path, policy: torch.nn.Module, normalizer: Normalizer,
                    policy_type: str, image_size: int, chunk_size: int,
                    tokenizer: CharTokenizer | None = None,
                    config: dict | None = None, metadata: dict | None = None) -> dict:
    """Save a self-describing checkpoint; returns the payload metadata."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "format_version": 1,
        "policy_type": policy_type,
        "config": dict(config or getattr(policy, "config").__dict__),
        "image_size": int(image_size),
        "chunk_size": int(chunk_size),
        "model_state": policy.state_dict(),
        "normalizer": normalizer.to_dict(),
        "tokenizer": tokenizer.to_dict() if tokenizer is not None else None,
        "metadata": dict(metadata or {}),
        "n_parameters": int(sum(p.numel() for p in policy.parameters())),
    }
    torch.save(payload, path)
    return {
        "path": str(path.resolve()),
        "bytes": int(path.stat().st_size),
        "n_parameters": payload["n_parameters"],
    }


def load_checkpoint(path, device: str | None = None) -> LoadedPolicy:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    payload = torch.load(path, map_location="cpu", weights_only=False)
    device = resolve_device(device)
    policy_type = payload["policy_type"]
    config = payload["config"]
    if policy_type == "act":
        cfg = ACTConfig(**{k: tuple(v) if isinstance(v, list) else v
                           for k, v in config.items()})
        policy = ACTPolicy(cfg)
    elif policy_type == "lang":
        tokenizer = CharTokenizer.from_dict(payload["tokenizer"])
        cfg = LangConfig(**{k: tuple(v) if isinstance(v, list) else v
                            for k, v in config.items()})
        policy = LangConditionedPolicy(tokenizer.vocab_size, cfg)
    else:  # pragma: no cover - defensive
        raise ValueError(f"unknown policy_type {policy_type!r}")
    policy.load_state_dict(payload["model_state"])
    policy.to(device)
    policy.eval()
    return LoadedPolicy(
        policy=policy,
        normalizer=Normalizer.from_dict(payload["normalizer"]),
        policy_type=policy_type,
        image_size=int(payload["image_size"]),
        chunk_size=int(payload["chunk_size"]),
        device=device,
        tokenizer=CharTokenizer.from_dict(payload["tokenizer"])
        if payload.get("tokenizer") else None,
        config=config,
        metadata=payload.get("metadata", {}),
    )


def checkpoint_summary(path) -> dict:
    """Cheap metadata read (no model construction)."""
    path = Path(path)
    payload = torch.load(path, map_location="cpu", weights_only=False)
    return {
        "path": str(path.resolve()),
        "policy_type": payload.get("policy_type"),
        "n_parameters": payload.get("n_parameters"),
        "bytes": int(path.stat().st_size),
        "metadata": payload.get("metadata", {}),
    }


def write_json(path, payload) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
