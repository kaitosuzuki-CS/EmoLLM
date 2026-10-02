"""Model and run statistics saved with checkpoints: parameter counts, forward FLOPs,
wall-clock timings, throughput, and peak GPU memory."""
import time

import torch
from torch.utils.flop_counter import FlopCounterMode


def count_params(module) -> dict:
    if module is None:
        return {"total": 0, "trainable": 0}
    params = list(module.parameters())
    return {"total": sum(p.numel() for p in params),
            "trainable": sum(p.numel() for p in params if p.requires_grad)}


def encoder_params(encoder) -> dict:
    """Backbone params, split into those used for embeddings and the unused
    pretraining decoder / emotion2vec+ 9-class head."""
    counts = count_params(encoder)
    audio = encoder.backbone.modality_encoders["AUDIO"]
    unused = sum(p.numel() for m in (audio.decoder, encoder.backbone.proj) if m is not None
                 for p in m.parameters())
    return {**counts, "used": counts["total"] - unused}


def _flops(fn) -> int:
    with FlopCounterMode(display=False) as counter:
        fn()
    return counter.get_total_flops()


@torch.no_grad()
def forward_flops(encoder, pooler, head, seconds: float, device, sample_rate: int = 16000) -> dict:
    """Forward FLOPs (FlopCounterMode, 1 multiply-add = 2 FLOPs) for one clip of
    `seconds` audio: encoder, pooler, and head (projector [+ classifier])."""
    wav = torch.randn(1, int(seconds * sample_rate), device=device)
    was_trainable = encoder.trainable
    encoder.set_trainable(False)  # count in eval mode (no layerdrop)
    try:
        enc = _flops(lambda: encoder.encode_frames(wav))
        x, pad = encoder.encode_frames(wav)
    finally:
        encoder.set_trainable(was_trainable)
    pool = _flops(lambda: pooler(x, pad))
    z = pooler(x, pad)
    head_flops = _flops(lambda: head(z))
    return {"clip_seconds": seconds, "encoder": enc, "pooler": pool, "head": head_flops,
            "total": enc + pool + head_flops}


def model_stats(cfg: dict, encoder, pooler, head, clip_seconds: float, device) -> dict:
    """Params and FLOPs of the full inference model (encoder + pooler + head). When the
    run trains on cached features the encoder isn't loaded, so a frozen copy is
    built here just for counting."""
    tmp = encoder is None
    if tmp:
        from emotion.encoder import Emotion2vecEncoder
        enc = cfg["encoder"]
        encoder = Emotion2vecEncoder(enc["model"], enc["hub"], enc["autocast_bf16"]).to(device)
    params = {"encoder": encoder_params(encoder), "pooler": count_params(pooler),
              "head": count_params(head)}
    if tmp:  # frozen pipeline: encoder never trains
        params["encoder"]["trainable"] = 0
    params["total"] = sum(params[k]["total"] for k in ("encoder", "pooler", "head"))
    params["trainable"] = sum(params[k]["trainable"] for k in ("encoder", "pooler", "head"))
    flops = {"per_clip": forward_flops(encoder, pooler, head, clip_seconds, device),
             "per_second": forward_flops(encoder, pooler, head, 1.0, device)}
    if tmp:
        del encoder
        torch.cuda.empty_cache()
    return {"params": params, "forward_flops": flops}


class RunTimer:
    """Wall-clock for the whole run, training steps, and evaluation passes."""

    def __init__(self, device):
        self.device = device
        self.start = time.time()
        self.train_s = 0.0
        self.eval_s = 0.0
        self.train_clips = 0
        self.epochs = 0
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)

    def summary(self) -> dict:
        out = {
            "wall_min": (time.time() - self.start) / 60,
            "train_min": self.train_s / 60,
            "eval_min": self.eval_s / 60,
            "epochs": self.epochs,
            "train_min_per_epoch": self.train_s / 60 / max(1, self.epochs),
            "train_clips_per_s": self.train_clips / max(1e-9, self.train_s),
        }
        if self.device.type == "cuda":
            out["peak_gpu_mem_gb"] = torch.cuda.max_memory_allocated(self.device) / 1e9
            out["gpu"] = torch.cuda.get_device_name(self.device)
        return out
