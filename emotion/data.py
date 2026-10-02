"""MELD datasets, splits, samplers, and dataloaders.

Three data paths feed the same training loop:
  * raw waveforms -> emotion2vec+ run inside the loop (`use_cache: false`)
  * precomputed mean-pooled utterance embeddings z (`use_cache: true`, mean pooling)
  * precomputed frame features (`use_cache: true`, attention pooling)
Caches come from `scripts/data/extract_embeddings.py`. Because the encoder is frozen and the
audio is not augmented, cached features are identical and epochs take seconds.
Batches are dicts with `label` and one of `z`, `frames` + `frame_mask`, or
`wav` + `padding_mask`.
"""
import random
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
import torch
from torch.utils.data import DataLoader, Dataset, Sampler

EMOTIONS = ["neutral", "joy", "surprise", "anger", "sadness", "disgust", "fear"]


def load_manifest(path: str) -> pd.DataFrame:
    return pd.read_csv(Path(path).expanduser())


def config_with_split(cfg: dict, data_cfg: dict, split: str) -> dict:
    """A copy of a run's config whose validation split is `split`. Splits the run's config
    predates (e.g. test, added after training) are taken from `data_cfg`."""
    import copy
    cfg = copy.deepcopy(cfg)
    if split not in cfg["data"]["splits"]:
        if split not in data_cfg["data"]["splits"]:
            raise KeyError(f"split {split!r} is not defined in the data config")
        cfg["data"]["splits"][split] = data_cfg["data"]["splits"][split]
    cfg["data"]["val_split"] = split
    return cfg


def split_paths(data_cfg: dict, split: str) -> dict:
    """Paths block for a named split (e.g. `train`, `dev`) from `data.splits`."""
    try:
        return data_cfg["splits"][split]
    except KeyError:
        raise KeyError(f"split {split!r} not in data.splits "
                       f"(have {list(data_cfg.get('splits', {}))})") from None


class WaveformDataset(Dataset):
    """Loads 16 kHz mono wavs. Clips longer than `max_seconds` are randomly
    cropped when `random_crop` is set, otherwise the first `max_seconds` kept."""

    def __init__(self, df: pd.DataFrame, sample_rate: int, max_seconds: float | None,
                 random_crop: bool):
        self.paths = df["path"].tolist()
        self.labels = df["label"].tolist()
        self.durations = (df["duration"].clip(upper=max_seconds) if max_seconds
                          else df["duration"]).tolist()
        self.max_len = int(sample_rate * max_seconds) if max_seconds else None
        self.sample_rate = sample_rate
        self.random_crop = random_crop

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        wav, sr = sf.read(self.paths[i], dtype="float32")
        assert sr == self.sample_rate, f"{self.paths[i]}: expected {self.sample_rate} Hz, got {sr}"
        if wav.ndim > 1:
            wav = wav.mean(axis=1)
        if self.max_len and len(wav) > self.max_len:
            start = random.randint(0, len(wav) - self.max_len) if self.random_crop else 0
            wav = wav[start:start + self.max_len]
        return {"wav": torch.from_numpy(wav), "label": self.labels[i], "index": i}


def collate_waveforms(items):
    lengths = torch.tensor([len(x["wav"]) for x in items])
    wav = torch.zeros(len(items), int(lengths.max()))
    for j, x in enumerate(items):
        wav[j, : len(x["wav"])] = x["wav"]
    padding_mask = torch.arange(wav.shape[1])[None, :] >= lengths[:, None]  # True = pad
    return {
        "wav": wav,
        "padding_mask": padding_mask,
        "label": torch.tensor([x["label"] for x in items]),
        "index": torch.tensor([x["index"] for x in items]),
    }


class EmbeddingDataset(Dataset):
    """Precomputed utterance embeddings z, aligned to a manifest by `id`."""

    def __init__(self, df: pd.DataFrame, cache: dict):
        pos = {k: i for i, k in enumerate(cache["ids"])}
        missing = [k for k in df["id"] if k not in pos]
        if missing:
            raise KeyError(f"{len(missing)} manifest ids missing from embedding cache, "
                           f"e.g. {missing[:3]}; re-run scripts/data/extract_embeddings.py")
        self.z = cache["z"][[pos[k] for k in df["id"]]].float()
        self.labels = torch.tensor(df["label"].tolist())

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        return {"z": self.z[i], "label": self.labels[i], "index": i}


class FrameDataset(Dataset):
    """Precomputed frame-level emotion2vec+ features (for attention pooling on a frozen
    encoder), aligned to a manifest by `id`. The cache stores all frames of all clips
    in one (N_frames, C) fp16 tensor with per-clip offsets."""

    def __init__(self, df: pd.DataFrame, cache: dict):
        pos = {k: i for i, k in enumerate(cache["ids"])}
        missing = [k for k in df["id"] if k not in pos]
        if missing:
            raise KeyError(f"{len(missing)} manifest ids missing from frame cache, e.g. "
                           f"{missing[:3]}; re-run scripts/data/extract_embeddings.py --frames")
        order = [pos[k] for k in df["id"]]
        self.frames = cache["frames"]
        self.offsets = cache["offsets"][order]
        self.lengths = cache["lengths"][order]
        self.durations = self.lengths.tolist()  # for length-sorted eval batches
        self.labels = torch.tensor(df["label"].tolist())

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        o, n = int(self.offsets[i]), int(self.lengths[i])
        return {"frames": self.frames[o:o + n], "label": self.labels[i], "index": i}


def collate_frames(items):
    lengths = torch.tensor([len(x["frames"]) for x in items])
    frames = torch.zeros(len(items), int(lengths.max()), items[0]["frames"].shape[1],
                         dtype=items[0]["frames"].dtype)
    for j, x in enumerate(items):
        frames[j, : len(x["frames"])] = x["frames"]
    return {
        "frames": frames,
        "frame_mask": torch.arange(frames.shape[1])[None, :] >= lengths[:, None],  # True = pad
        "label": torch.stack([torch.as_tensor(x["label"]) for x in items]),
        "index": torch.tensor([x["index"] for x in items]),
    }


class ClassBalancedBatchSampler(Sampler):
    """Each batch holds `samples_per_class` items from each of `classes_per_batch`
    randomly chosen classes (minority classes are sampled with replacement).
    Guarantees positives for SupCon and counteracts MELD's neutral-heavy skew.
    One epoch = len(dataset) // batch_size batches."""

    def __init__(self, labels, classes_per_batch: int, samples_per_class: int, seed: int = 0):
        labels = np.asarray(labels)
        self.by_class = {c: np.flatnonzero(labels == c) for c in np.unique(labels)}
        self.classes_per_batch = min(classes_per_batch, len(self.by_class))
        self.samples_per_class = samples_per_class
        self.num_batches = len(labels) // (self.classes_per_batch * samples_per_class)
        self.rng = np.random.default_rng(seed)

    def __len__(self):
        return self.num_batches

    def __iter__(self):
        classes = list(self.by_class)
        for _ in range(self.num_batches):
            batch = []
            for c in self.rng.choice(classes, self.classes_per_batch, replace=False):
                idx = self.by_class[c]
                replace = len(idx) < self.samples_per_class
                batch.extend(self.rng.choice(idx, self.samples_per_class, replace=replace).tolist())
            yield batch


class LengthBucketedBatchSampler(Sampler):
    """Shuffled batches of similar-length clips, to cut padding when the encoder runs
    in the loop. Indices are shuffled, split into pools of `pool_batches` batches,
    sorted by duration within each pool, cut into batches, and the batches shuffled."""

    def __init__(self, durations, batch_size: int, drop_last: bool, pool_batches: int = 50,
                 seed: int = 0):
        self.durations = np.asarray(durations)
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.pool = batch_size * pool_batches
        self.rng = np.random.default_rng(seed)

    def __len__(self):
        n = len(self.durations)
        return n // self.batch_size if self.drop_last else -(-n // self.batch_size)

    def __iter__(self):
        idx = self.rng.permutation(len(self.durations))
        batches = []
        for p in range(0, len(idx), self.pool):
            pool = idx[p:p + self.pool]
            pool = pool[np.argsort(self.durations[pool], kind="stable")]
            batches += [pool[i:i + self.batch_size].tolist()
                        for i in range(0, len(pool), self.batch_size)]
        if self.drop_last:
            batches = [b for b in batches if len(b) == self.batch_size]
        for i in self.rng.permutation(len(batches)):
            yield batches[i]


class SortedBatchSampler(Sampler):
    """Deterministic batches in order of duration (for evaluation passes)."""

    def __init__(self, durations, batch_size: int):
        order = np.argsort(np.asarray(durations), kind="stable")
        self.batches = [order[i:i + batch_size].tolist() for i in range(0, len(order), batch_size)]

    def __len__(self):
        return len(self.batches)

    def __iter__(self):
        return iter(self.batches)


def _loader(ds, cfg: dict, train: bool, collate, seed: int):
    common = dict(num_workers=cfg["num_workers"], collate_fn=collate,
                  pin_memory=True, persistent_workers=cfg["num_workers"] > 0)
    if not train:
        if hasattr(ds, "durations"):  # variable-length audio or frames: batch by length
            return DataLoader(ds, batch_sampler=SortedBatchSampler(ds.durations, cfg["batch_size"]),
                              **common)
        return DataLoader(ds, batch_size=cfg["batch_size"], shuffle=False, **common)
    if cfg["sampler"] == "class_balanced":
        sampler = ClassBalancedBatchSampler(ds.labels, cfg["classes_per_batch"],
                                            cfg["samples_per_class"], seed)
        return DataLoader(ds, batch_sampler=sampler, **common)
    if cfg["sampler"] == "length_bucketed":
        if not hasattr(ds, "durations"):
            raise ValueError("length_bucketed sampler needs variable-length inputs "
                             "(raw audio or cached frames), not cached z")
        sampler = LengthBucketedBatchSampler(ds.durations, cfg["batch_size"], drop_last=True,
                                             seed=seed)
        return DataLoader(ds, batch_sampler=sampler, **common)
    if cfg["sampler"] != "random":
        raise ValueError(f"unknown sampler {cfg['sampler']!r}")
    return DataLoader(ds, batch_size=cfg["batch_size"], shuffle=True, drop_last=True, **common)


def _load_cache(path: str, frames: bool = False) -> dict:
    path = Path(path).expanduser()
    if not path.exists():
        raise FileNotFoundError(f"no {'frame' if frames else 'embedding'} cache at {path}; run "
                                f"scripts/data/extract_embeddings.py --config <config>"
                                + (" --frames" if frames else ""))
    return torch.load(path)


def uses_frames(cfg: dict) -> bool:
    """Attention pooling needs frame features; mean pooling can use cached z."""
    return (cfg.get("pooling") or {}).get("type", "mean") != "mean"


def build_dataloaders(cfg: dict):
    """Returns (loaders, train_df, val_df), where loaders has `train` (shuffled,
    length-bucketed or class-balanced), `train_eval` (train set, deterministic, no
    crop randomness) and `val`. Evaluation over raw audio is batched by length."""
    data = cfg["data"]
    train_paths = split_paths(data, data["train_split"])
    val_paths = split_paths(data, data["val_split"])
    train_df = load_manifest(train_paths["manifest"])
    val_df = load_manifest(val_paths["manifest"])
    loader_cfg = {**cfg["train"], "num_workers": data["num_workers"]}

    if data["use_cache"] and uses_frames(cfg):
        train_ds = FrameDataset(train_df, _load_cache(train_paths["frame_cache"], frames=True))
        val_ds = FrameDataset(val_df, _load_cache(val_paths["frame_cache"], frames=True))
        train_eval_ds = train_ds
        collate = collate_frames
        loader_cfg["num_workers"] = 0
    elif data["use_cache"]:
        train_ds = EmbeddingDataset(train_df, _load_cache(train_paths["embedding_cache"]))
        val_ds = EmbeddingDataset(val_df, _load_cache(val_paths["embedding_cache"]))
        train_eval_ds = train_ds
        collate = None
        loader_cfg["num_workers"] = 0  # in-memory tensors; workers only add overhead
    else:
        sr, max_s = data["sample_rate"], data["max_seconds"]
        eval_max_s = data.get("eval_max_seconds") or max_s
        train_ds = WaveformDataset(train_df, sr, max_s, random_crop=True)
        train_eval_ds = WaveformDataset(train_df, sr, eval_max_s, random_crop=False)
        val_ds = WaveformDataset(val_df, sr, eval_max_s, random_crop=False)
        collate = collate_waveforms

    eval_cfg = {**loader_cfg, "batch_size": cfg["train"].get("eval_batch_size",
                                                             loader_cfg["batch_size"])}
    loaders = {
        "train": _loader(train_ds, loader_cfg, True, collate, cfg["seed"]),
        "train_eval": _loader(train_eval_ds, eval_cfg, False, collate, cfg["seed"]),
        "val": _loader(val_ds, eval_cfg, False, collate, cfg["seed"]),
    }
    return loaders, train_df, val_df
