"""Utilities shared by the classification and SupCon training pipelines."""
import argparse
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from sklearn.metrics import accuracy_score, f1_score

from emotion.config import load_config
from emotion.data import EMOTIONS
from emotion.model import ZSource, get_z  # noqa: F401  (re-exported for scripts)


def parse_args_and_config(default_config: str) -> dict:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=default_config)
    parser.add_argument("overrides", nargs="*", help="dotted overrides, e.g. train.lr=1e-3")
    args = parser.parse_args()
    return load_config(args.config, args.overrides)


def setup(cfg: dict) -> torch.device:
    """Seeds everything, caps CPU threads, and returns the device."""
    # PyTorch defaults to one thread per core (255 here); for small tensors that
    # just busy-spins the whole machine.
    torch.set_num_threads(cfg["num_threads"])
    set_seed(cfg["seed"])
    return torch.device(cfg["device"] if torch.cuda.is_available() else "cpu")


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def make_run_dir(cfg: dict) -> Path:
    stem = Path(cfg["output_dir"]) / f"{cfg['task']}_{time.strftime('%Y%m%d_%H%M%S')}"
    stem.parent.mkdir(parents=True, exist_ok=True)
    # Runs started in the same second get _1, _2, ...; mkdir is atomic, so
    # concurrent launches never share a directory.
    for i in range(1000):
        run_dir = stem.with_name(stem.name + (f"_{i}" if i else ""))
        try:
            run_dir.mkdir()
            break
        except FileExistsError:
            continue
    with open(run_dir / "config.yaml", "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
    return run_dir


class Logger:
    """Prints to stdout, appends JSON lines to `<run_dir>/log.jsonl`, and logs to
    Weights & Biases when `wandb.enabled` is set in the config.

    Records carry a `split` key; W&B metrics are named `<split>/<key>` and plotted
    against the global train step (the record's `step`). Records without a step
    (e.g. the final linear probe) go to the W&B run summary."""

    def __init__(self, run_dir: Path, cfg: dict):
        self.file = open(run_dir / "log.jsonl", "a")
        self.run = None
        w = cfg.get("wandb") or {}
        if w.get("enabled"):
            import wandb
            self.run = wandb.init(project=w["project"], entity=w.get("entity"),
                                  name=w.get("name") or run_dir.name,
                                  group=w.get("group") or cfg["task"], tags=w.get("tags"),
                                  mode=w.get("mode", "online"), config=cfg, dir=str(run_dir))
            self.run.define_metric("step")
            self.run.define_metric("*", step_metric="step")

    def log(self, record: dict, echo: bool = True):
        """echo=False sends the record to W&B only (for per-step train losses)."""
        if echo:
            self.file.write(json.dumps(record) + "\n")
            self.file.flush()
            print(" | ".join(f"{k}={v:.4g}" if isinstance(v, float) else f"{k}={v}"
                             for k, v in record.items()), flush=True)
        if self.run is not None:
            split = record.get("split", "misc")
            data = {f"{split}/{k}": v for k, v in record.items()
                    if k not in ("split", "step") and isinstance(v, (int, float))}
            if "step" in record:
                self.run.log({**data, "step": record["step"]})
            else:
                self.run.summary.update(data)

    def summary(self, data: dict):
        """Sets W&B run-summary values (e.g. the best validation score)."""
        if self.run is not None:
            self.run.summary.update(data)

    def close(self):
        self.file.close()
        if self.run is not None:
            self.run.finish()


def build_encoder(cfg: dict, device, trainable: bool = False):
    """The encoder is only needed when training from raw audio."""
    if cfg["data"]["use_cache"]:
        return None
    from emotion.encoder import Emotion2vecEncoder
    enc = cfg["encoder"]
    return Emotion2vecEncoder(enc["model"], enc["hub"], enc["autocast_bf16"], trainable=trainable,
                              freeze_feature_extractor=enc.get("freeze_feature_extractor", True),
                              ).to(device)


def _fmt_flops(n: float) -> str:
    for unit, scale in (("PFLOPs", 1e15), ("TFLOPs", 1e12), ("GFLOPs", 1e9)):
        if n >= scale:
            return f"{n / scale:.2f} {unit}"
    return f"{n:.0f} FLOPs"


def _flatten(d: dict, prefix: str = "") -> dict:
    out = {}
    for k, v in d.items():
        key = f"{prefix}{k}"
        if isinstance(v, dict):
            out.update(_flatten(v, key + "/"))
        elif isinstance(v, (int, float, str)):
            out[key] = v
    return out


def save_stats(cfg, run_dir: Path, logger, timer, device, *, encoder, pooler, head,
               train_df, val_df, best: dict) -> dict:
    """Collects run timings, data sizes, best metrics and (if cfg.profile) params and
    forward FLOPs; writes <run_dir>/stats.json, adds `stats` to best.pt / last.pt, and
    puts the flattened values in the W&B summary under `stats/`."""
    from emotion.profiling import model_stats

    data = cfg["data"]
    clip_s = float(train_df["duration"].clip(upper=data["max_seconds"]).mean())
    stats = {
        "task": cfg["task"], "objective": cfg.get("objective", cfg["task"]),
        "pooling": (cfg.get("pooling") or {}).get("type", "mean"),
        "run": timer.summary(),
        "data": {"train_split": data["train_split"], "train_utts": len(train_df),
                 "val_split": data["val_split"], "val_utts": len(val_df),
                 "mean_train_clip_s": clip_s, "cached_features": bool(data["use_cache"])},
        "best": best,
    }
    if cfg.get("profile", True):
        stats.update(model_stats(cfg, encoder, pooler, head, clip_s, device))
        fl = stats["forward_flops"]["per_clip"]
        finetune = cfg["task"] == "finetune"
        frozen_in_loop = fl["encoder"] if (not data["use_cache"] and not finetune) else 0
        trained = fl["pooler"] + fl["head"] + (fl["encoder"] if finetune else 0)
        # forward + backward ~ 3x forward for trained modules, 1x for frozen ones run in the loop
        stats["train_flops_est"] = timer.train_clips * (frozen_in_loop + 3 * trained)

    (run_dir / "stats.json").write_text(json.dumps(stats, indent=2))
    for name in ("best.pt", "last.pt"):
        path = run_dir / name
        if path.exists():
            ckpt = torch.load(path, map_location="cpu", weights_only=False)
            ckpt["stats"] = stats
            torch.save(ckpt, path)
    logger.summary(_flatten(stats, "stats/"))

    r = stats["run"]
    msg = (f"time: {r['wall_min']:.1f} min total, {r['train_min']:.1f} train "
           f"({r['train_min_per_epoch']:.2f}/epoch, {r['train_clips_per_s']:.0f} clips/s), "
           f"{r['eval_min']:.1f} eval")
    if "params" in stats:
        p, f = stats["params"], stats["forward_flops"]
        msg += (f"\nparams: {p['total'] / 1e6:.1f}M total, {p['trainable'] / 1e6:.2f}M trainable "
                f"| forward: {f['per_clip']['total'] / 1e9:.1f} GFLOPs per {clip_s:.2f}s clip, "
                f"{f['per_second']['total'] / 1e9:.1f} GFLOPs/s audio "
                f"| est. training compute {_fmt_flops(stats['train_flops_est'])}")
    print(msg)
    return stats


def build_optimizer(params, cfg: dict, steps_per_epoch: int):
    """AdamW with linear warmup then cosine decay, stepped per batch.

    `params` is a module (one group at train.lr) or a list of param-group dicts with
    their own `lr` and optional `start_epoch`: such a group has lr 0 until then and
    runs its own warmup + cosine over the remaining epochs (e.g. an encoder
    unfrozen after the heads warm up)."""
    t = cfg["train"]
    if isinstance(params, torch.nn.Module):
        params = [{"params": params.parameters(), "lr": t["lr"]}]
    starts = [g.pop("start_epoch", 0) * steps_per_epoch for g in params]
    opt = torch.optim.AdamW(params, weight_decay=t["weight_decay"])
    total = t["epochs"] * steps_per_epoch
    warmup = t["warmup_epochs"] * steps_per_epoch

    def make_lambda(start):
        def lr_lambda(step):
            step -= start
            if step < 0:
                return 0.0
            if step < warmup:
                return (step + 1) / warmup
            progress = (step - warmup) / max(1, total - start - warmup)
            return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))
        return lr_lambda

    return opt, torch.optim.lr_scheduler.LambdaLR(opt, [make_lambda(s) for s in starts])


def classification_metrics(y_true, y_pred) -> dict:
    return {
        "acc": float(accuracy_score(y_true, y_pred)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted")),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
    }


def per_class_f1(y_true, y_pred) -> dict:
    scores = f1_score(y_true, y_pred, average=None, labels=range(len(EMOTIONS)), zero_division=0)
    return {e: round(float(s), 4) for e, s in zip(EMOTIONS, scores)}


def class_weights(labels, scheme: str | None, num_classes: int) -> torch.Tensor | None:
    """`inverse` = N / (C * n_c); `inverse_sqrt` = its square root; None = unweighted."""
    if not scheme:
        return None
    counts = np.bincount(np.asarray(labels), minlength=num_classes).astype(np.float64)
    w = counts.sum() / (num_classes * np.maximum(counts, 1))
    if scheme == "inverse_sqrt":
        w = np.sqrt(w)
    elif scheme != "inverse":
        raise ValueError(f"unknown class weight scheme {scheme!r}")
    return torch.tensor(w, dtype=torch.float32)


@torch.no_grad()
def embed(loader, source, fn, device):
    """Runs `fn(z)` over a loader; returns (stacked outputs, labels) on CPU."""
    outs, labels = [], []
    for batch in loader:
        outs.append(fn(get_z(batch, source, device)).float().cpu())
        labels.append(batch["label"])
    return torch.cat(outs), torch.cat(labels)


def knn_predict(bank, bank_labels, query, k: int, temperature: float, num_classes: int,
                chunk: int = 1024):
    """Cosine kNN with exp(sim / T) weighted voting (as in InstDisc / SupCon evals)."""
    bank = F.normalize(bank, dim=1)
    query = F.normalize(query, dim=1)
    preds = []
    for q in query.split(chunk):
        sim, idx = (q @ bank.T).topk(k, dim=1)
        votes = torch.zeros(len(q), num_classes)
        votes.scatter_add_(1, bank_labels[idx], (sim / temperature).exp())
        preds.append(votes.argmax(1))
    return torch.cat(preds)
