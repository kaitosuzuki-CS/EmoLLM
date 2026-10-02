"""Load trained emotion-model checkpoints (any pipeline) for evaluation and feature extraction."""
from pathlib import Path

import torch

from emotion.encoder import Emotion2vecEncoder
from emotion.model import EmotionClassifier, FinetuneModel, build_head, build_pooler


def load_checkpoint(path: Path, device):
    """Returns (cfg, head, encoder, pooler, ckpt); encoder is None when the frozen run
    used cached features."""
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    cfg = ckpt["config"]
    if cfg["task"] == "finetune":
        cfg["data"]["use_cache"] = False
        enc = cfg["encoder"]
        model = FinetuneModel(Emotion2vecEncoder(enc["model"], enc["hub"], enc["autocast_bf16"]),
                              build_head(cfg), build_pooler(cfg))
        model.load_state_dict(ckpt["model"])
        head, encoder, pooler = model.head, model.encoder, model.pooler
    else:
        # frozen pipelines: the checkpoint holds the classifier / projector (+ pooler)
        head = build_head(cfg, cfg["task"])
        head.load_state_dict(ckpt["model"])
        pooler = build_pooler(cfg)
        if "pooler" in ckpt:
            pooler.load_state_dict(ckpt["pooler"])
        encoder = None if cfg["data"]["use_cache"] else frozen_encoder(cfg)
    if encoder is not None:
        encoder = encoder.to(device).eval()
    return cfg, head.to(device).eval(), encoder, pooler.to(device).eval(), ckpt


def frozen_encoder(cfg):
    enc = cfg["encoder"]
    return Emotion2vecEncoder(enc["model"], enc["hub"], enc["autocast_bf16"])


def projector_of(head):
    return head.projector if isinstance(head, EmotionClassifier) else head
