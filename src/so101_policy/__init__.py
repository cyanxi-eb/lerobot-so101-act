"""Policies for the SO-101 simulated imitation-learning project.

* :class:`so101_policy.act.ACTPolicy` - ACT-style action-chunking transformer
  with a CVAE latent.
* :class:`so101_policy.lang_cond.LangConditionedPolicy` - lightweight
  language-conditioned action chunking policy (the "SmolVLA-style" baseline).
"""

from .act import ACTConfig, ACTPolicy, act_loss, build_act
from .inference import (
    LoadedPolicy,
    load_checkpoint,
    resolve_device,
    save_checkpoint,
)
from .lang_cond import (
    CharTokenizer,
    LangConditionedPolicy,
    LangConfig,
    lang_loss,
)
from .normalization import Normalizer
from .rollout import run_rollout, run_rollouts, summarise_rollouts

__all__ = [
    "ACTConfig",
    "ACTPolicy",
    "CharTokenizer",
    "LangConditionedPolicy",
    "LangConfig",
    "LoadedPolicy",
    "Normalizer",
    "act_loss",
    "build_act",
    "lang_loss",
    "load_checkpoint",
    "resolve_device",
    "run_rollout",
    "run_rollouts",
    "save_checkpoint",
    "summarise_rollouts",
]
