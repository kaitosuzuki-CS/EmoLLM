"""Trainable pooling and heads on top of emotion2vec+ features."""
import torch
import torch.nn as nn

from emotion.data import EMOTIONS
from emotion.encoder import mean_pool


class MeanPool(nn.Module):
    """Mean over valid frames (emotion2vec's own utterance embedding). No parameters."""

    def forward(self, x, frame_pad):
        return mean_pool(x, frame_pad)


class AttentionPool(nn.Module):
    """Additive attention pooling over frames:
        a_t = softmax_t(w^T tanh(W h_t + b)),  z = sum_t a_t h_t   (padded frames masked out).
    The score layer starts at zero, so training starts from exactly mean pooling."""

    def __init__(self, dim: int, hidden_dim: int = 128):
        super().__init__()
        self.proj = nn.Linear(dim, hidden_dim)
        self.score = nn.Linear(hidden_dim, 1)
        nn.init.zeros_(self.score.weight)
        nn.init.zeros_(self.score.bias)

    def forward(self, x, frame_pad):
        logits = self.score(torch.tanh(self.proj(x))).squeeze(-1).float()  # (B, T)
        weights = logits.masked_fill(frame_pad, float("-inf")).softmax(dim=1)
        return torch.einsum("bt,btc->bc", weights.to(x.dtype), x)


def build_pooler(cfg: dict) -> nn.Module:
    pool = cfg.get("pooling") or {"type": "mean"}
    if pool["type"] == "mean":
        return MeanPool()
    if pool["type"] == "attention":
        return AttentionPool(cfg["encoder"]["embed_dim"], pool.get("hidden_dim", 128))
    raise ValueError(f"unknown pooling type {pool['type']!r}")


class EmotionProjector(nn.Module):
    """z (embed_dim) -> Linear -> GELU -> Linear -> D-dim emotion representation."""

    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, dropout: float = 0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, z):
        return self.net(z)


class EmotionClassifier(nn.Module):
    """Projector followed by a linear classification head over the D-dim representation."""

    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, num_classes: int,
                 dropout: float = 0.0, head_dropout: float = 0.0):
        super().__init__()
        self.projector = EmotionProjector(in_dim, hidden_dim, out_dim, dropout)
        self.head = nn.Sequential(nn.Dropout(head_dropout), nn.Linear(out_dim, num_classes))

    def forward(self, z):
        h = self.projector(z)
        return self.head(h), h


class ZSource:
    """Turns a batch into utterance embeddings z. Cached z passes through; cached frame
    features, or frames from the encoder on raw audio, go through the pooler.
    A plain object holding references, so it adds nothing to any state dict."""

    def __init__(self, encoder=None, pooler=None):
        self.encoder = encoder
        self.pooler = pooler

    def __call__(self, batch: dict, device) -> torch.Tensor:
        if "z" in batch:
            return batch["z"].to(device, non_blocking=True)
        if "frames" in batch:
            x = batch["frames"].to(device, non_blocking=True).float()
            pad = batch["frame_mask"].to(device, non_blocking=True)
        else:
            x, pad = self.encoder.encode_frames(batch["wav"].to(device, non_blocking=True),
                                                batch["padding_mask"].to(device, non_blocking=True))
        return self.pooler(x, pad)


def get_z(batch: dict, source: ZSource, device) -> torch.Tensor:
    return source(batch, device)


def build_head(cfg: dict, objective: str | None = None):
    """Projector (+ classifier) for `objective` (default cfg["objective"], else cfg["task"])."""
    objective = objective or cfg.get("objective") or cfg["task"]
    m, in_dim = cfg["model"], cfg["encoder"]["embed_dim"]
    if objective == "classification":
        return EmotionClassifier(in_dim, m["hidden_dim"], m["out_dim"], len(EMOTIONS),
                                 m["dropout"], m["head_dropout"])
    if objective == "supcon":
        return EmotionProjector(in_dim, m["hidden_dim"], m["out_dim"], m["dropout"])
    raise ValueError(f"unknown objective {objective!r}")


class FinetuneModel(nn.Module):
    """Encoder (trainable or frozen) + pooler + head, as trained by scripts/emotion/train_finetune.py."""

    def __init__(self, encoder, head, pooler=None):
        super().__init__()
        self.encoder = encoder
        self.pooler = pooler if pooler is not None else build_pooler({})
        self.head = head

    @property
    def source(self) -> ZSource:
        return ZSource(self.encoder, self.pooler)

    def forward(self, batch, device):
        return self.head(get_z(batch, self.source, device))
