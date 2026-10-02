"""The frozen emotion model as the LLM sees it: audio -> (z, h, emotion probabilities).

Used by scripts/llm/extract_features.py (whole splits, cached) and scripts/llm/chat.py (live
audio)."""
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from emotion.checkpoint import frozen_encoder, load_checkpoint
from emotion.data import EMOTIONS
from emotion.model import EmotionClassifier


@torch.no_grad()
def emotion_head(head, ckpt_path: Path, device):
    """-> fn(z) returning (class logits, latent h). Classification checkpoints use their own
    head; SupCon checkpoints (projector only) use the linear probe saved next to them,
    applied to normalize(h) as in training."""
    if isinstance(head, EmotionClassifier):
        return lambda z: head(z)
    probe_path = Path(ckpt_path).parent / "linear_probe.pt"
    if not probe_path.exists():
        raise FileNotFoundError(f"{ckpt_path} has no classifier head and no {probe_path.name} "
                                "for predicted emotions (train it with eval.save_probe=true)")
    p = torch.load(probe_path, map_location="cpu", weights_only=False)
    probe = torch.nn.Linear(p["in_features"], p["num_classes"]).to(device)
    probe.load_state_dict(p["state_dict"])
    print(f"predicted emotions from {probe_path} (dev weighted F1 {p['metrics']['weighted_f1']:.3f})")

    def fn(z):
        h = head(z)
        return probe(F.normalize(h.float(), dim=1)), h
    fn.probe = probe
    return fn


class EmotionFeaturizer:
    """Loads an emotion-model checkpoint (with its frozen or fine-tuned encoder) and maps a
    16 kHz mono waveform to the features the LLM policy was trained on."""

    def __init__(self, ckpt_path, device):
        self.device = device
        self.cfg, head, encoder, self.pooler, self.ckpt = load_checkpoint(Path(ckpt_path), device)
        if encoder is None:   # trained on cached features: the encoder is the frozen emotion2vec+
            encoder = frozen_encoder(self.cfg).to(device).eval()
        self.encoder, self.head = encoder.eval(), head.eval()
        self.pooler.eval()
        self.head_fn = emotion_head(head, Path(ckpt_path), device)
        probe = getattr(self.head_fn, "probe", None)
        self.num_params = {name: sum(p.numel() for p in m.parameters())
                           for name, m in [("emotion2vec+ encoder", self.encoder), ("pooling", self.pooler),
                                           ("emotion projector + head", self.head)]
                           + ([("linear probe", probe)] if probe is not None else [])}
        data = self.cfg["data"]
        self.sample_rate = data["sample_rate"]
        self.max_seconds = data.get("eval_max_seconds") or data["max_seconds"]

    @torch.no_grad()
    def __call__(self, wav: np.ndarray) -> dict:
        """wav: float32 mono at self.sample_rate. Returns z, h (1-d tensors) and
        {emotion: probability}."""
        wav = wav[: int(self.max_seconds * self.sample_rate)]
        x = torch.from_numpy(np.ascontiguousarray(wav, dtype=np.float32))[None].to(self.device)
        frames, pad = self.encoder.encode_frames(x, torch.zeros_like(x, dtype=torch.bool))
        z = self.pooler(frames, pad)
        logits, h = self.head_fn(z)
        probs = logits.softmax(-1)[0].float().cpu()
        return {"z": z[0].float().cpu(), "h": h[0].float().cpu(), "frames": tuple(frames.shape),
                "probs": dict(zip(EMOTIONS, probs.tolist())), "emotion": EMOTIONS[int(probs.argmax())]}
