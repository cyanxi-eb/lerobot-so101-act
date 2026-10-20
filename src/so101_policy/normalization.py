"""Root-mean-square normalisation for proprioception and actions.

Statistics are computed on the *training split only* by
:func:`so101_dataset.cache.build_cache` and stored inside every checkpoint so
that training and inference always see the same transform ("aligning the
inference environment with training").
"""

from __future__ import annotations

import numpy as np

try:  # torch is optional for pure-numpy tests
    import torch
except Exception:  # pragma: no cover - torch is a hard dependency in practice
    torch = None  # type: ignore[assignment]


class Normalizer:
    """Per-dimension ``(x - mean) / std`` transform with its inverse."""

    def __init__(self, state_mean, state_std, action_mean, action_std, eps: float = 1e-6):
        self.state_mean = np.asarray(state_mean, dtype=np.float32)
        self.state_std = np.maximum(np.asarray(state_std, dtype=np.float32), eps)
        self.action_mean = np.asarray(action_mean, dtype=np.float32)
        self.action_std = np.maximum(np.asarray(action_std, dtype=np.float32), eps)
        if self.state_mean.shape != self.state_std.shape:
            raise ValueError("state mean/std shape mismatch")
        if self.action_mean.shape != self.action_std.shape:
            raise ValueError("action mean/std shape mismatch")

    # -- numpy -----------------------------------------------------------
    def normalize_state(self, state):
        return (np.asarray(state, dtype=np.float32) - self.state_mean) / self.state_std

    def normalize_action(self, action):
        return (np.asarray(action, dtype=np.float32) - self.action_mean) / self.action_std

    def denormalize_action(self, action):
        return np.asarray(action, dtype=np.float32) * self.action_std + self.action_mean

    def denormalize_state(self, state):
        return np.asarray(state, dtype=np.float32) * self.state_std + self.state_mean

    # -- torch -----------------------------------------------------------
    def to_torch(self, device="cpu") -> "Normalizer":
        if torch is None:  # pragma: no cover
            raise RuntimeError("torch is not installed")
        self._t = {
            "state_mean": torch.as_tensor(self.state_mean, device=device),
            "state_std": torch.as_tensor(self.state_std, device=device),
            "action_mean": torch.as_tensor(self.action_mean, device=device),
            "action_std": torch.as_tensor(self.action_std, device=device),
        }
        return self

    def torch_normalize_state(self, state: "torch.Tensor") -> "torch.Tensor":
        return (state - self._t["state_mean"]) / self._t["state_std"]

    def torch_normalize_action(self, action: "torch.Tensor") -> "torch.Tensor":
        return (action - self._t["action_mean"]) / self._t["action_std"]

    def torch_denormalize_action(self, action: "torch.Tensor") -> "torch.Tensor":
        return action * self._t["action_std"] + self._t["action_mean"]

    # -- serialisation ---------------------------------------------------
    def to_dict(self) -> dict:
        return {
            "state_mean": self.state_mean.tolist(),
            "state_std": self.state_std.tolist(),
            "action_mean": self.action_mean.tolist(),
            "action_std": self.action_std.tolist(),
        }

    @classmethod
    def from_dict(cls, payload: dict) -> "Normalizer":
        return cls(
            payload["state_mean"],
            payload["state_std"],
            payload["action_mean"],
            payload["action_std"],
        )

    @classmethod
    def from_cache_meta(cls, meta: dict) -> "Normalizer":
        state = meta["state_stats"]
        action = meta["action_stats"]
        return cls(state["mean"], state["std"], action["mean"], action["std"])
