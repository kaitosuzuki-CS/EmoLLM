"""Evaluate one or more trained emotion-model checkpoints on a data split, and compare them.

For every checkpoint (any pipeline), on the chosen split:
  * kNN: cosine kNN on the D-dim latent, with the train split as the bank
  * probe: the linear probe saved next to the checkpoint, or one refit on train (--fit-probe)
  * head: the trained classification head (classification checkpoints)
With two or more checkpoints, each is compared to the first with a paired bootstrap over
utterances of the difference in its headline score (head, else probe, else kNN).

    python scripts/emotion/evaluate.py runs/finetune_<stamp> --split test
    python scripts/emotion/evaluate.py runs/finetune_<A> runs/supcon_<B> --split test --fit-probe
    python scripts/emotion/evaluate.py runs/finetune_<stamp>/last.pt --split dev --k 50
    python scripts/emotion/evaluate.py runs/finetune_<stamp> --manifest my_manifest.csv --name mine

A checkpoint is a run directory (its best.pt) or a .pt file. Splits are those of the run's
config plus any defined in --data-config (e.g. test, if the run predates it). Writes
<run_dir>/eval_<split>_<checkpoint>.json per checkpoint and, with several, the comparison to
--out (default runs/compare_<split>.json).
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix, f1_score
from torch.utils.data import DataLoader

from emotion.checkpoint import frozen_encoder, load_checkpoint, projector_of
from emotion.config import load_config
from emotion.data import (
    EMOTIONS,
    SortedBatchSampler,
    WaveformDataset,
    build_dataloaders,
    collate_waveforms,
    config_with_split,
    load_manifest,
)
from emotion.model import EmotionClassifier
from emotion.trainer import ZSource, classification_metrics, knn_predict, per_class_f1

METRICS = {"weighted_f1": lambda y, p: f1_score(y, p, average="weighted"),
           "macro_f1": lambda y, p: f1_score(y, p, average="macro"),
           "acc": lambda y, p: float((y == p).mean())}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("checkpoints", nargs="+", help="run directories (best.pt) or .pt checkpoints")
    p.add_argument("--split", default="dev", help="data split to evaluate (default: dev)")
    p.add_argument("--manifest", help="evaluate on this manifest instead of a split")
    p.add_argument("--name", help="label for output files (default: the split, or the manifest stem)")
    p.add_argument("--fit-probe", action="store_true",
                   help="refit the logistic-regression probe on the train split when no probe is saved")
    p.add_argument("--k", type=int, help="kNN neighbours (default: run's eval.knn_k or 20)")
    p.add_argument("--knn-temperature", type=float, help="kNN vote temperature (default: 0.07)")
    p.add_argument("--metric", default="weighted_f1", choices=list(METRICS),
                   help="score compared across checkpoints (default: weighted_f1)")
    p.add_argument("--n-boot", type=int, default=2000, help="bootstrap resamples for comparisons")
    p.add_argument("--data-config", default="configs/emotion/base.yaml",
                   help="defines splits missing from a run's config (e.g. test)")
    p.add_argument("--data-root", help="data_root for --data-config (default: its own)")
    p.add_argument("--out", help="comparison output (default: runs/compare_<name>.json)")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--device", default="cuda")
    return p.parse_args()


def report(name, y_true, y_pred):
    metrics = classification_metrics(y_true, y_pred)
    per_class = per_class_f1(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred, labels=range(len(EMOTIONS)))
    print(f"\n== {name}")
    print("  " + " | ".join(f"{k}={v:.4f}" for k, v in metrics.items()))
    print("  per-class F1: " + ", ".join(f"{e} {s:.3f}" for e, s in per_class.items()))
    width = max(len(e) for e in EMOTIONS)
    print("  confusion (rows = true, cols = predicted):")
    print("  " + " " * width + "".join(f"{e[:5]:>7}" for e in EMOTIONS))
    for e, row in zip(EMOTIONS, cm):
        print(f"  {e:>{width}}" + "".join(f"{v:>7}" for v in row))
    return {**metrics, "per_class_f1": per_class, "confusion": cm.tolist()}


@torch.no_grad()
def embed_in_order(loader, source, fn, device):
    """fn(z) for every item, in dataset order (evaluation loaders may be length-sorted)."""
    n = len(loader.dataset)
    out, y = None, np.zeros(n, np.int64)
    for batch in loader:
        o = fn(source(batch, device)).float().cpu()
        if out is None:
            out = torch.zeros(n, o.shape[1])
        i = batch["index"]
        out[i], y[i.numpy()] = o, batch["label"].numpy()
    return out, torch.from_numpy(y)


def evaluate(path: Path, args, data_cfg, device) -> dict:
    ckpt_path = path / "best.pt" if path.is_dir() else path
    run_dir = ckpt_path.parent
    cfg, head, encoder, pooler, ckpt = load_checkpoint(ckpt_path, device)
    torch.set_num_threads(cfg.get("num_threads", 8))
    cfg["train"]["eval_batch_size"] = args.batch_size
    print(f"\n######## {ckpt_path} | task {cfg['task']}"
          + (f" ({cfg['objective']})" if cfg["task"] == "finetune" else "")
          + f" | pooling {(cfg.get('pooling') or {}).get('type', 'mean')} | epoch {ckpt.get('epoch')}")

    cfg = config_with_split(cfg, data_cfg, args.split if not args.manifest else cfg["data"]["val_split"])
    loaders, train_df, query_df = build_dataloaders(cfg)
    data = cfg["data"]
    source = query_source = ZSource(encoder, pooler)
    query_loader = loaders["val"]
    if args.manifest:  # a new manifest has no cached features: frozen runs encode it from audio
        query_df = load_manifest(args.manifest)
        ds = WaveformDataset(query_df, data["sample_rate"],
                             data.get("eval_max_seconds") or data["max_seconds"], random_crop=False)
        query_loader = DataLoader(ds, batch_sampler=SortedBatchSampler(ds.durations, args.batch_size),
                                  num_workers=data["num_workers"], collate_fn=collate_waveforms)
        if encoder is None:
            query_source = ZSource(frozen_encoder(cfg).to(device).eval(), pooler)
    name = args.name or (Path(args.manifest).stem if args.manifest else args.split)
    print(f"bank: {data['train_split']} ({len(train_df)} utts) | query: {name} ({len(query_df)} utts)")

    projector = projector_of(head)
    rep = lambda z: projector(z)  # noqa: E731
    bank, bank_y = embed_in_order(loaders["train_eval"], source, rep, device)
    query, query_y = embed_in_order(query_loader, query_source, rep, device)
    y_true = query_y.numpy()
    results = {"checkpoint": str(ckpt_path), "epoch": ckpt.get("epoch"), "query": name}
    preds = {}

    ev = cfg.get("eval", {})
    k = args.k or ev.get("knn_k", 20)
    temp = args.knn_temperature or ev.get("knn_temperature", 0.07)
    preds["knn"] = knn_predict(bank, bank_y, query, k, temp, len(EMOTIONS)).numpy()
    results["knn"] = {"k": k, "temperature": temp, **report(f"kNN (k={k}, T={temp})", y_true, preds["knn"])}

    probe_path = run_dir / "linear_probe.pt"
    if probe_path.exists():
        p = torch.load(probe_path, map_location="cpu", weights_only=False)
        probe = nn.Linear(p["in_features"], p["num_classes"])
        probe.load_state_dict(p["state_dict"])
        with torch.no_grad():
            preds["linear_probe"] = probe(F.normalize(query, dim=1)).argmax(1).numpy()
        results["linear_probe"] = {"file": str(probe_path),
                                   **report("linear probe (saved)", y_true, preds["linear_probe"])}
    elif args.fit_probe:
        from sklearn.linear_model import LogisticRegression
        clf = LogisticRegression(max_iter=2000, C=ev.get("probe_C", 1.0),
                                 class_weight=ev.get("probe_class_weight"))
        clf.fit(F.normalize(bank, dim=1).numpy(), bank_y.numpy())
        preds["linear_probe"] = clf.predict(F.normalize(query, dim=1).numpy())
        results["linear_probe"] = {"file": "refit on the train split",
                                   **report("linear probe (refit on train)", y_true, preds["linear_probe"])}

    if isinstance(head, EmotionClassifier):
        with torch.no_grad():
            preds["head"] = head.head(query.to(device)).argmax(1).cpu().numpy()
        results["head"] = report("classification head", y_true, preds["head"])

    out = run_dir / f"eval_{name}_{ckpt_path.stem}.json"
    out.write_text(json.dumps(results, indent=2))
    print(f"\nwrote {out}")
    headline = next(m for m in ("head", "linear_probe", "knn") if m in preds)
    return {"name": str(path), "ids": list(query_df["id"]), "y": y_true, "preds": preds,
            "headline": headline, "results": results}


def compare(runs: list[dict], metric: str, n_boot: int) -> list[dict]:
    """Paired bootstrap over utterances of each run's headline score minus the first run's."""
    base = runs[0]
    rng = np.random.default_rng(0)
    y = base["y"]
    idx = rng.integers(0, len(y), size=(n_boot, len(y)))
    fn = METRICS[metric]
    out = []
    pb = base["preds"][base["headline"]]
    for r in runs[1:]:
        if r["ids"] != base["ids"]:
            raise ValueError(f"{r['name']} was evaluated on different utterances than {base['name']}")
        pr = r["preds"][r["headline"]]
        diffs = np.array([fn(y[i], pr[i]) - fn(y[i], pb[i]) for i in idx])
        delta = fn(y, pr) - fn(y, pb)
        out.append({"a": base["name"], "a_score": f"{base['headline']} {metric}", "b": r["name"],
                    "b_score": f"{r['headline']} {metric}", "a_value": float(fn(y, pb)),
                    "b_value": float(fn(y, pr)), "delta_b_minus_a": float(delta),
                    "ci95": [float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))],
                    "p_b_better": float((diffs > 0).mean())})
    return out


def main():
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    data_cfg = load_config(args.data_config, [f"data_root={args.data_root}"] if args.data_root else [])
    runs = [evaluate(Path(c), args, data_cfg, device) for c in args.checkpoints]
    if len(runs) > 1:
        comps = compare(runs, args.metric, args.n_boot)
        print(f"\n######## paired bootstrap ({args.n_boot} resamples) of {args.metric}, vs {runs[0]['name']}")
        for c in comps:
            print(f"  {c['b']} ({c['b_score']}) {c['b_value']:.4f} vs {c['a_value']:.4f} ({c['a_score']}): "
                  f"delta {c['delta_b_minus_a']:+.4f} [{c['ci95'][0]:+.4f}, {c['ci95'][1]:+.4f}], "
                  f"P(better) {c['p_b_better']:.2f}")
        name = args.name or (Path(args.manifest).stem if args.manifest else args.split)
        out = Path(args.out or f"runs/compare_{name}.json")
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps({"split": name, "metric": args.metric, "comparisons": comps}, indent=2))
        print(f"wrote {out}")


if __name__ == "__main__":
    main()
