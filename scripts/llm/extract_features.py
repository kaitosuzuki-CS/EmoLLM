"""Run the frozen emotion model over each transcript's audio and cache, per split:
  features_<split>.pt     ids, h (projector latent), z (pooled embedding), class probs,
                          predicted emotion, ground-truth label
  utterances_<split>.jsonl  transcript, speaker, dialogue context, ground-truth emotion,
                          has_audio (utterances without audio remain as context)

MELD stores each utterance as its own clip, dia<Dialogue_ID>_utt<Utterance_ID>.mp4, so a
transcript is matched to its audio by those ids and the whole clip is encoded (capped at
the emotion model's evaluation length, 20 s, like everywhere else).

    python scripts/llm/extract_features.py --config configs/llm.yaml
"""
import argparse
import json
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from emollm.data import add_context, cache_paths, load_transcripts
from emollm.emotion_features import emotion_head
from emotion.checkpoint import frozen_encoder, load_checkpoint
from emotion.config import load_config
from emotion.data import EMOTIONS, SortedBatchSampler, collate_waveforms, load_manifest, split_paths
from emotion.model import EmotionClassifier
from emotion.trainer import classification_metrics


class ClipAudio(Dataset):
    """The utterance's whole clip (up to max_seconds)."""

    def __init__(self, rows, sample_rate, max_seconds):
        self.rows, self.sr, self.max_s = rows, sample_rate, max_seconds
        self.durations = [min(r["duration"], max_seconds) for r in rows]

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        wav, sr = sf.read(self.rows[i]["path"], dtype="float32")
        assert sr == self.sr
        if wav.ndim > 1:
            wav = wav.mean(axis=1)
        n = max(1, int(round(self.durations[i] * sr)))
        return {"wav": torch.from_numpy(np.ascontiguousarray(wav[:n])), "label": 0, "index": i}


def extract(split, csv_path, manifest_path, ecfg, head, encoder, pooler, out_dir, batch_size, device, meta):
    tr = add_context(load_transcripts(csv_path))
    manifest = load_manifest(manifest_path)
    audio = manifest.set_index("id")[["path", "duration"]].to_dict("index")
    tr["has_audio"] = tr["id"].isin(audio.keys())
    rows = [{**r, **audio[r["id"]]} for r in tr[tr["has_audio"]].to_dict("records")]
    ds = ClipAudio(rows, ecfg["data"]["sample_rate"],
                   ecfg["data"].get("eval_max_seconds") or ecfg["data"]["max_seconds"])
    loader = DataLoader(ds, batch_sampler=SortedBatchSampler(ds.durations, batch_size),
                        num_workers=8, collate_fn=collate_waveforms)

    n = len(ds)
    head_fn = emotion_head(head, Path(meta["emotion_checkpoint"]), device)
    projector = head.projector if isinstance(head, EmotionClassifier) else head
    h = torch.zeros(n, projector.net[-1].out_features)
    z = torch.zeros(n, ecfg["encoder"]["embed_dim"])
    probs = torch.zeros(n, len(EMOTIONS))
    for batch in tqdm(loader, desc=f"{split} audio"):
        x, pad = encoder.encode_frames(batch["wav"].to(device), batch["padding_mask"].to(device))
        zb = pooler(x, pad)
        logits, hb = head_fn(zb)
        idx = batch["index"]
        z[idx], h[idx], probs[idx] = zb.float().cpu(), hb.float().cpu(), logits.softmax(-1).float().cpu()

    pred = probs.argmax(1)
    label = torch.tensor([EMOTIONS.index(r["emotion"]) for r in rows])
    paths = cache_paths(out_dir, split)
    paths["features"].parent.mkdir(parents=True, exist_ok=True)
    torch.save({"ids": [r["id"] for r in rows], "h": h, "z": z, "probs": probs, "pred": pred,
                "pred_emotion": [EMOTIONS[i] for i in pred.tolist()], "label": label,
                "emotions": EMOTIONS, "clip_seconds": ds.durations, **meta}, paths["features"])
    with open(paths["utterances"], "w") as f:
        for r in tr.to_dict("records"):
            f.write(json.dumps({k: (bool(v) if k == "has_audio" else v) for k, v in r.items()}) + "\n")

    m = classification_metrics(label.numpy(), pred.numpy())
    capped = sum(r["duration"] - d > 0.05 for r, d in zip(rows, ds.durations))
    print(f"{split}: {len(tr)} utterances, {n} with audio ({capped} longer than the cap) "
          f"| emotion model acc {m['acc']:.4f} weighted F1 {m['weighted_f1']:.4f} "
          f"| saved {paths['features']} and {paths['utterances'].name}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/llm.yaml")
    parser.add_argument("--splits", type=lambda v: v.split(","),
                        help="comma-separated names, e.g. train,dev (default: data.train_split and data.val_split)")
    parser.add_argument("overrides", nargs="*")
    args = parser.parse_args()
    cfg = load_config(args.config, args.overrides)
    torch.set_num_threads(cfg["num_threads"])
    device = torch.device(cfg["device"] if torch.cuda.is_available() else "cpu")

    ckpt_path = cfg["emotion"]["checkpoint"]
    if not ckpt_path or not Path(ckpt_path).exists():
        raise FileNotFoundError(f"emotion.checkpoint={ckpt_path!r} not found; pass "
                                "emotion.checkpoint=runs/<emotion run>/best.pt")
    ecfg, head, encoder, pooler, ckpt = load_checkpoint(Path(ckpt_path), device)
    if encoder is None:  # emotion model trained on cached features: load the frozen encoder
        encoder = frozen_encoder(ecfg).to(device).eval()
    print(f"emotion model: {cfg['emotion']['checkpoint']} (epoch {ckpt['epoch']}, "
          f"pooling {(ecfg.get('pooling') or {}).get('type', 'mean')})")
    params = (ckpt.get("stats") or {}).get("params", {}).get("total") or sum(
        p.numel() for m in (encoder, pooler, head) for p in m.parameters())
    meta = {"emotion_checkpoint": str(ckpt_path), "emotion_params": int(params)}
    for split in args.splits or [cfg["data"]["train_split"], cfg["data"]["val_split"]]:
        # audio manifests come from the emotion data config (the checkpoint's own config may
        # predate a split, e.g. test)
        dcfg = load_config(cfg["emotion"]["data_config"], [f"data_root={cfg['data_root']}"])
        manifest = split_paths(dcfg["data"], split)["manifest"]
        extract(split, cfg["data"]["splits"][split], manifest, ecfg, head, encoder, pooler,
                cfg["emotion"]["cache_dir"], cfg["emotion"]["batch_size"], device, meta)


if __name__ == "__main__":
    main()
