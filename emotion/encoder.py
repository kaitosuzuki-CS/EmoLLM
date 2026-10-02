"""emotion2vec+ encoder producing utterance-level embeddings z (frozen or trainable)."""
import contextlib

import torch
import torch.nn as nn

# (kernel, stride) of the conv feature extractor: waveform samples -> frames
_CONV_LAYERS = [(10, 5)] + [(3, 2)] * 4 + [(2, 2)] * 2


def frame_lengths(sample_lengths: torch.Tensor) -> torch.Tensor:
    n = sample_lengths.clone()
    for k, s in _CONV_LAYERS:
        n = torch.div(n - k, s, rounding_mode="floor") + 1
    return n


class Emotion2vecEncoder(nn.Module):
    """Reproduces FunASR's emotion2vec utterance embedding
    (`generate(..., granularity="utterance")`): waveform layer norm -> backbone
    -> mean over frames, on zero-padded batches.

    Padding is made exact: the conv feature extractor is valid-mode with per-frame
    norms and attention masks padded keys, but the convolutional relative positional
    encoder does not see the padding mask, so pad frames would bleed into the tail
    of shorter clips. Forward pre-hooks zero the pad frames before each of its conv
    layers, which is what zero-padding at the boundary of an unpadded clip does.

    trainable=False: all weights frozen, backbone always in eval mode, no grad.
    trainable=True: backbone weights train (except the CNN feature extractor when
    freeze_feature_extractor), dropout/layerdrop active in train mode.
    """

    def __init__(self, model: str, hub: str = "ms", autocast_bf16: bool = True,
                 trainable: bool = False, freeze_feature_extractor: bool = True):
        super().__init__()
        from funasr import AutoModel

        auto = AutoModel(model=model, hub=hub, device="cpu", disable_update=True,
                         disable_pbar=True, log_level="ERROR")
        self.backbone = auto.model
        self.normalize = bool(self.backbone.cfg.normalize)
        self.autocast_bf16 = autocast_bf16
        self.trainable = trainable

        audio = self.backbone.modality_encoders["AUDIO"]
        self.backbone.requires_grad_(trainable)
        if freeze_feature_extractor:
            audio.local_encoder.requires_grad_(False)
        # Pretraining decoder and emotion2vec+'s own 9-class head are unused here
        for unused in (audio.decoder, self.backbone.proj):
            if unused is not None:
                unused.requires_grad_(False)
        if not trainable:
            self.backbone.eval()

        self._frame_lens = None
        for layer in audio.relative_positional_encoder:
            if isinstance(layer, nn.Sequential):  # conv -> LN -> GELU, input (B, C, T)
                layer.register_forward_pre_hook(self._zero_pad_frames)

    def _zero_pad_frames(self, module, inputs):
        if self._frame_lens is None:
            return None
        x = inputs[0]
        valid = torch.arange(x.shape[-1], device=x.device)[None, :] < self._frame_lens[:, None]
        return (x * valid.unsqueeze(1).to(x.dtype),)

    def train(self, mode: bool = True):
        super().train(mode)
        if not self.trainable:
            self.backbone.eval()  # never enable dropout/layerdrop in a frozen backbone
        return self

    def set_trainable(self, trainable: bool):
        """Temporarily run as frozen (no grad, eval mode) or trainable, without
        changing which parameters requires_grad (e.g. to warm up heads first)."""
        self.trainable = trainable
        return self.train(self.training)

    def forward(self, wav: torch.Tensor, padding_mask: torch.Tensor | None = None) -> torch.Tensor:
        """wav: (B, T) float at 16 kHz; padding_mask: (B, T) bool, True = pad.
        Returns the mean-pooled utterance embedding z: (B, embed_dim) float32."""
        return mean_pool(*self.encode_frames(wav, padding_mask))

    def encode_frames(self, wav: torch.Tensor, padding_mask: torch.Tensor | None = None):
        """Returns frame features (B, frames, embed_dim) float32 and the frame padding
        mask (B, frames) bool, True = pad."""
        # trainable: follow the caller's grad mode (no graph under an eval no_grad)
        grad = contextlib.nullcontext() if self.trainable else torch.no_grad()
        ctx = (torch.autocast("cuda", dtype=torch.bfloat16)
               if self.autocast_bf16 and wav.is_cuda else contextlib.nullcontext())
        with grad, ctx:
            x, frame_pad = self._encode(wav, padding_mask)
        if not torch.isfinite(x).all():
            raise FloatingPointError(
                f"non-finite emotion2vec features on {wav.device}. If the batch looks fine, "
                "suspect the GPU (a faulty card can return NaNs from cuDNN convolutions); "
                "retry on another device.")
        return x, frame_pad

    def _encode(self, wav, padding_mask):
        if padding_mask is None or not padding_mask.any():
            if self.normalize:
                wav = _masked_layer_norm(wav.float(), None)
            x = self.backbone.extract_features(wav, padding_mask=None)["x"].float()
            return x, torch.zeros(x.shape[:2], dtype=torch.bool, device=x.device)

        sample_lens = (~padding_mask).sum(1)
        if self.normalize:
            wav = _masked_layer_norm(wav.float(), padding_mask)
        self._frame_lens = frame_lengths(sample_lens)
        try:
            out = self.backbone.extract_features(wav, padding_mask=padding_mask)
        finally:
            self._frame_lens = None
        x, frame_pad = out["x"].float(), out["padding_mask"]
        if frame_pad is None:  # FunASR drops the mask if no frame ends up padded
            frame_pad = torch.zeros(x.shape[:2], dtype=torch.bool, device=x.device)
        return x, frame_pad


def mean_pool(x: torch.Tensor, frame_pad: torch.Tensor) -> torch.Tensor:
    """Mean over valid frames: x (B, T, C), frame_pad (B, T) True = pad -> (B, C)."""
    valid = (~frame_pad).unsqueeze(-1).to(x.dtype)
    return (x * valid).sum(1) / valid.sum(1).clamp(min=1.0)


def _masked_layer_norm(wav, padding_mask, eps=1e-5):
    """F.layer_norm(wav_i, wav_i.shape) on each utterance's unpadded samples; pads stay 0."""
    if padding_mask is None:
        mean = wav.mean(1, keepdim=True)
        var = wav.var(1, unbiased=False, keepdim=True)
        return (wav - mean) / torch.sqrt(var + eps)
    valid = (~padding_mask).to(wav.dtype)
    n = valid.sum(1, keepdim=True).clamp(min=1.0)
    mean = (wav * valid).sum(1, keepdim=True) / n
    var = (((wav - mean) * valid) ** 2).sum(1, keepdim=True) / n
    return (wav - mean) / torch.sqrt(var + eps) * valid
