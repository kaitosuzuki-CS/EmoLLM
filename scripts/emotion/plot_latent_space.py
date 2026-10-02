"""Visualize and quantify how well emotion classes separate in the emotion encoder's
latent spaces, on train (seen by the encoder) and held-out splits (default: dev and test).

Feature sources (rows of the comparison):
  raw  untrained emotion2vec+ utterance embedding (mean-pooled, 768-d; the z cache)
  z    the checkpoint's pooled embedding (768-d)
  h    the checkpoint's projector latent (D-d) - what the LLM's soft prompts are built from

Projections: PCA and LDA are fit on train and applied unchanged to every held-out split
(LDA is supervised: it shows the best linear 2-D separation); t-SNE is run per split on a
subsample. Held-out splits are evaluation only: nothing is fit on them.
Plots are small multiples: one panel per emotion (that class in blue over everything else
in gray) plus an overview with labelled class centroids, so identity never relies on
distinguishing seven colours.

Metrics (full feature space, standardized with train statistics):
  silhouette (cosine, by true label), Fisher ratio tr(S_between) / tr(S_within), and per
  held-out split: kNN (cosine, k=20, train bank), logistic-regression probe (fit on train)
  and the class-centroid cosine-similarity matrix.

    python scripts/emotion/plot_latent_space.py runs/<emotion run>                 # train + dev + test
    python scripts/emotion/plot_latent_space.py runs/<emotion run> --split test    # train + test
    python scripts/emotion/plot_latent_space.py runs/<emotion run>/last.pt --sources z,h --methods pca,lda

The checkpoint is a run directory (its best.pt) or a .pt file.
"""
import os

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")  # long eval clips

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from emotion.checkpoint import load_checkpoint
from emotion.config import load_config
from emotion.data import EMOTIONS, build_dataloaders, config_with_split, load_manifest, split_paths
from emotion.trainer import ZSource, classification_metrics, knn_predict

# Reference palette (dataviz skill): chart chrome + slot-1 blue + blue<->red diverging.
SURFACE, INK, INK2, MUTED, BASE = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#c3c2b7"
HIGHLIGHT, CONTEXT = "#2a78d6", "#d9d8d2"
SHORT = {"raw": "untrained emotion2vec+", "z": "fine-tuned pooled z", "h": "fine-tuned latent h"}
DIVERGING = ["#2a78d6", "#f0efec", "#e34948"]


# ------------------------------------------------------------------------------ features

def encode(loader, source, projector, d, device):
    n = len(loader.dataset)
    z, h, y = np.zeros((n, d), np.float32), None, np.zeros(n, np.int64)
    with torch.no_grad():  # eval batches are length-sorted: write back by dataset index
        for batch in loader:
            zb = source(batch, device)
            hb = projector(zb).float().cpu().numpy()
            if h is None:
                h = np.zeros((n, hb.shape[1]), np.float32)
            i = batch["index"].numpy()
            z[i], h[i], y[i] = zb.float().cpu().numpy(), hb, batch["label"].numpy()
    return {"z": z, "h": h, "y": y}


def checkpoint_features(ckpt: Path, device, eval_splits, data_cfg) -> dict:
    """{split: {"z": (N, 768), "h": (N, D), "y": (N,)}} for train and each held-out split."""
    cfg, head, encoder, pooler, _ = load_checkpoint(ckpt, device)
    cfg["train"]["eval_batch_size"] = 16   # clips up to 20 s; keeps attention memory modest
    projector = head.projector if hasattr(head, "projector") else head
    source = ZSource(encoder, pooler)
    d = cfg["encoder"]["embed_dim"]
    out = {"train": encode(build_dataloaders(cfg)[0]["train_eval"], source, projector, d, device)}
    for split in eval_splits:
        loader = build_dataloaders(config_with_split(cfg, data_cfg, split))[0]["val"]
        out[split] = encode(loader, source, projector, d, device)
    return out, cfg


def raw_features(cfg: dict, data_cfg: dict, splits) -> dict:
    """Untrained emotion2vec+ z from the embedding caches, aligned to the manifests."""
    out = {}
    for name in splits:
        split = cfg["data"]["train_split"] if name == "train" else name
        paths = split_paths(config_with_split(cfg, data_cfg, split)["data"], split)
        cache = torch.load(Path(paths["embedding_cache"]).expanduser())
        pos = {k: i for i, k in enumerate(cache["ids"])}
        df = load_manifest(paths["manifest"])
        idx = [pos[k] for k in df["id"]]
        out[name] = {"raw": cache["z"][idx].float().numpy(), "y": df["label"].to_numpy()}
    return out


# ------------------------------------------------------------------------------- metrics

def fisher_ratio(x, y):
    mu = x.mean(0)
    sb = sum((y == c).sum() * np.sum((x[y == c].mean(0) - mu) ** 2) for c in np.unique(y))
    sw = sum(np.sum((x[y == c] - x[y == c].mean(0)) ** 2) for c in np.unique(y))
    return float(sb / sw)


def centroid_similarity(x, y):
    cents = np.stack([x[y == c].mean(0) for c in range(len(EMOTIONS))])
    cents /= np.linalg.norm(cents, axis=1, keepdims=True)
    return cents @ cents.T


def metrics(xtr, ytr, held_out: dict) -> dict:
    """held_out: {split: (x, y)}. Standardized with train statistics; probes fit on train."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import silhouette_samples
    mu, sd = xtr.mean(0), xtr.std(0) + 1e-6
    xtr_s = (xtr - mu) / sd
    splits = {"train": (xtr_s, ytr), **{k: ((x - mu) / sd, y) for k, (x, y) in held_out.items()}}
    probe = LogisticRegression(max_iter=3000, C=1.0).fit(xtr_s, ytr)
    out = {}
    for name, (x, y) in splits.items():
        # same subsample for every source / checkpoint (seeded by split size), so
        # silhouettes are comparable across rows and runs
        sub = np.random.default_rng(len(x)).choice(len(x), min(4000, len(x)), replace=False)
        sil = silhouette_samples(x[sub], y[sub], metric="cosine")
        out[name] = {"silhouette": float(sil.mean()),
                     "silhouette_per_class": {EMOTIONS[c]: float(sil[y[sub] == c].mean())
                                              for c in range(len(EMOTIONS)) if (y[sub] == c).any()},
                     "fisher_ratio": fisher_ratio(x, y)}
        if name == "train":
            continue
        preds = knn_predict(torch.from_numpy(xtr_s).float(), torch.from_numpy(ytr).long(),
                            torch.from_numpy(x).float(), 20, 0.07, len(EMOTIONS)).numpy()
        out[f"{name}_knn"] = classification_metrics(y, preds)
        out[f"{name}_probe"] = classification_metrics(y, probe.predict(x))
        out[f"{name}_centroid_cosine"] = centroid_similarity(x, y).round(4).tolist()
    return out


# ---------------------------------------------------------------------------- projections

def project(method, xtr, ytr, held_out: dict, rng, tsne_n):
    """-> {split: (N, 2)} for train and each held-out split, index arrays of the plotted
    points, axis labels. PCA / LDA are fit on train only."""
    from sklearn.decomposition import PCA
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    from sklearn.manifold import TSNE
    mu, sd = xtr.mean(0), xtr.std(0) + 1e-6
    xs = {"train": (xtr - mu) / sd, **{k: (x - mu) / sd for k, x in held_out.items()}}
    every = {k: np.arange(len(x)) for k, x in xs.items()}
    if method == "pca":
        p = PCA(2, random_state=0).fit(xs["train"])
        ev = p.explained_variance_ratio_ * 100
        return ({k: p.transform(x) for k, x in xs.items()}, every,
                (f"PC1 ({ev[0]:.1f}% var)", f"PC2 ({ev[1]:.1f}% var)"))
    if method == "lda":
        p = LinearDiscriminantAnalysis(n_components=2).fit(xs["train"], ytr)
        ev = p.explained_variance_ratio_ * 100
        return ({k: p.transform(x) for k, x in xs.items()}, every,
                (f"LD1 ({ev[0]:.1f}% of between-class var)", f"LD2 ({ev[1]:.1f}%)"))
    # t-SNE: no out-of-sample transform, so embed each split separately (PCA-50 first)
    pca = PCA(min(50, xs["train"].shape[1]), random_state=0).fit(xs["train"])
    out, idx = {}, {}
    for name, x in xs.items():
        idx[name] = np.sort(rng.choice(len(x), min(tsne_n, len(x)), replace=False))
        out[name] = TSNE(2, perplexity=30, init="pca", random_state=0).fit_transform(
            pca.transform(x[idx[name]]))
    return out, idx, ("t-SNE 1", "t-SNE 2")


# ---------------------------------------------------------------------------------- plots

def style_axes(ax):
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(BASE)
        ax.spines[side].set_linewidth(0.8)
    ax.set_xticks([])
    ax.set_yticks([])


def label_positions(cents: dict, lim, min_gap: float = 0.085):
    """Greedy vertical de-overlap for centroid labels: start just right of each centroid,
    then push labels apart until neighbours are at least `min_gap` (axes fraction) apart."""
    (x0, x1), (y0, y1) = lim[:, 0], lim[:, 1]
    w, h = x1 - x0, y1 - y0
    pts = [(e, (c[0] - x0) / w, (c[1] - y0) / h) for e, c in cents.items()]
    order = sorted(range(len(pts)), key=lambda i: pts[i][2])
    ys = {i: pts[i][2] for i in order}
    for _ in range(50):
        moved = False
        for a, b in zip(order, order[1:]):
            close_x = abs(pts[a][1] - pts[b][1]) < 0.35
            if close_x and ys[b] - ys[a] < min_gap:
                push = (min_gap - (ys[b] - ys[a])) / 2
                ys[a], ys[b] = ys[a] - push, ys[b] + push
                moved = True
        if not moved:
            break
    # keep the whole stack inside the panel by shifting it (clamping labels one by one
    # would pile them up against the edge again)
    lo, hi = min(ys.values()), max(ys.values())
    shift = (0.04 - lo) if lo < 0.04 else (0.96 - hi) if hi > 0.96 else 0.0
    return [(x0 + (min(pts[i][1] + 0.06, 0.8)) * w, y0 + (ys[i] + shift) * h)
            for i in range(len(pts))]


def plot_projection(coords, idx, labels, axis_names, ys, title, sil, path):
    import matplotlib.pyplot as plt
    ncol, rows = len(EMOTIONS) + 1, list(coords)
    fig, axes = plt.subplots(len(rows), ncol, figsize=(2.3 * ncol, 2.6 * len(rows)), facecolor=SURFACE,
                             squeeze=False)
    for r, split in enumerate(rows):
        xy, y = coords[split], ys[split][idx[split]]
        lim = (np.percentile(xy, [0.5, 99.5], axis=0) * 1.1)
        for c in range(ncol):
            ax = axes[r, c]
            style_axes(ax)
            ax.set_xlim(lim[:, 0])
            ax.set_ylim(lim[:, 1])
            if c == 0:  # overview: every point in gray + labelled class centroids
                ax.scatter(*xy.T, s=2, c=CONTEXT, linewidths=0, rasterized=True)
                cents = {e: np.median(xy[y == k], axis=0) for k, e in enumerate(EMOTIONS) if (y == k).any()}
                for (e, (cx, cy)), (lx, ly) in zip(cents.items(), label_positions(cents, lim)):
                    ax.scatter(cx, cy, s=36, c=INK, edgecolors=SURFACE, linewidths=1.5, zorder=3)
                    ax.annotate(e, (cx, cy), xytext=(lx, ly), textcoords="data", fontsize=8,
                                color=INK, zorder=4, va="center",
                                arrowprops={"arrowstyle": "-", "color": MUTED, "lw": 0.6,
                                            "shrinkA": 0, "shrinkB": 3})
                ax.set_title(f"{split}: all (n={len(y)})", fontsize=9, color=INK, loc="left")
                ax.set_xlabel(axis_names[0], fontsize=7, color=MUTED)
                ax.set_ylabel(axis_names[1], fontsize=7, color=MUTED)
                continue
            k = c - 1
            m = y == k
            ax.scatter(*xy[~m].T, s=2, c=CONTEXT, linewidths=0, rasterized=True)
            ax.scatter(*xy[m].T, s=5, c=HIGHLIGHT, alpha=0.7, linewidths=0, rasterized=True)
            s = sil[split]["silhouette_per_class"].get(EMOTIONS[k])
            ax.set_title(f"{EMOTIONS[k]} (n={int(m.sum())})" + ("" if s is None else f"\nsilhouette {s:+.2f}"),
                         fontsize=8.5, color=INK, loc="left")
    fig.suptitle(title, fontsize=11, color=INK, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def plot_centroid_similarity(mats: dict, path, split: str):
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list("div", DIVERGING)
    fig, axes = plt.subplots(1, len(mats), figsize=(5.2 * len(mats), 4.8), facecolor=SURFACE,
                             gridspec_kw={"wspace": 0.45})
    axes = np.atleast_1d(axes)
    for ax, (name, m) in zip(axes, mats.items()):
        m = np.asarray(m)
        im = ax.imshow(m, cmap=cmap, vmin=-1, vmax=1)
        ax.set_xticks(range(len(EMOTIONS)), [e[:4] for e in EMOTIONS], fontsize=8, color=INK2)
        ax.set_yticks(range(len(EMOTIONS)), EMOTIONS, fontsize=8, color=INK2)
        for i in range(len(EMOTIONS)):
            for j in range(len(EMOTIONS)):
                ax.text(j, i, f"{m[i, j]:.2f}", ha="center", va="center", fontsize=7,
                        color="#ffffff" if abs(m[i, j]) > 0.6 else INK)
        off = m[~np.eye(len(m), dtype=bool)]
        ax.set_title(f"{name}\nmean off-diagonal {off.mean():+.2f}", fontsize=9, color=INK, loc="left")
        for s in ax.spines.values():
            s.set_visible(False)
    cb = fig.colorbar(im, ax=axes.tolist(), shrink=0.8)
    cb.outline.set_visible(False)
    cb.ax.tick_params(labelsize=7, colors=INK2)
    fig.suptitle(f"{split.capitalize()} class-centroid cosine similarity (standardized features); lower "
                 "off-diagonal = more separated", fontsize=10, color=INK, x=0.01, ha="left")
    fig.savefig(path, dpi=150, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)


# ----------------------------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("checkpoint", help="emotion model run directory (its best.pt) or a .pt file")
    ap.add_argument("--sources", default="raw,z,h", type=lambda v: v.split(","))
    ap.add_argument("--methods", default="pca,lda,tsne", type=lambda v: v.split(","))
    ap.add_argument("--tsne-n", type=int, default=3000, help="points per split for t-SNE")
    ap.add_argument("--out", help="output dir (default: runs/latent_<run name>)")
    ap.add_argument("--name", help="label used in figure titles (default: the run directory name)")
    ap.add_argument("--split", default="dev,test", type=lambda v: v.split(","),
                    help="held-out split(s) to plot next to train, comma-separated (default: dev,test)")
    ap.add_argument("--data-config", default="configs/emotion/base.yaml",
                    help="where splits missing from the checkpoint's config (e.g. test) are defined")
    ap.add_argument("--data-root", help="data_root for --data-config (default: its own)")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    torch.set_num_threads(8)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    ckpt = Path(args.checkpoint)
    ckpt = ckpt / "best.pt" if ckpt.is_dir() else ckpt
    run = ckpt.parent
    out = Path(args.out or f"runs/latent_{run.name}")
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    data_cfg = load_config(args.data_config, [f"data_root={args.data_root}"] if args.data_root else [])
    held = args.split

    tic = time.time()
    feats, cfg = checkpoint_features(ckpt, device, held, data_cfg)
    if "raw" in args.sources:
        raw = raw_features(cfg, data_cfg, ["train", *held])
        for split in feats:
            assert (raw[split]["y"] == feats[split]["y"]).all(), "raw cache / manifest mismatch"
            feats[split]["raw"] = raw[split]["raw"]
    print(f"features in {time.time() - tic:.0f}s: " + ", ".join(f"{k} {len(v['y'])}" for k, v in feats.items()))

    ys = {s: feats[s]["y"] for s in feats}
    label = args.name or run.name
    names = {"raw": "untrained emotion2vec+ z", "z": f"{label} z (pooled)",
             "h": f"{label} h (projector latent)"}
    results, centroid_mats = {}, {split: {} for split in held}
    for src in args.sources:
        xtr = feats["train"][src]
        ho = {split: feats[split][src] for split in held}
        m = metrics(xtr, ys["train"], {split: (ho[split], ys[split]) for split in held})
        results[src] = {"dim": int(xtr.shape[1]), **m}
        for split in held:
            centroid_mats[split][f"{src}: {SHORT[src]}"] = m[f"{split}_centroid_cosine"]
        print(f"{src:>4} ({xtr.shape[1]}-d) | " + " | ".join(
            f"{split}: silhouette {m[split]['silhouette']:+.3f} Fisher {m[split]['fisher_ratio']:.3f}"
            + ("" if split == "train" else f" kNN wF1 {m[split + '_knn']['weighted_f1']:.3f}"
               f" probe wF1 {m[split + '_probe']['weighted_f1']:.3f}") for split in ["train", *held]))
        sil = ", ".join(f"{split} {m[split]['silhouette']:+.2f}" for split in ["train", *held])
        for method in args.methods:
            coords, idx, axis_names = project(method, xtr, ys["train"], ho, rng, args.tsne_n)
            title = f"{names[src]}: {method.upper()} (silhouette {sil})"
            plot_projection(coords, idx, None, axis_names, ys, title, m, out / f"{src}_{method}.png")
    for split in held:   # dev keeps the original file name
        name = "centroid_similarity.png" if split == "dev" else f"centroid_similarity_{split}.png"
        plot_centroid_similarity(centroid_mats[split], out / name, split)
    (out / "metrics.json").write_text(json.dumps(results, indent=2))
    print(f"wrote {len(args.sources) * len(args.methods)} projection plots, centroid similarity plots "
          f"and metrics.json to {out} ({time.time() - tic:.0f}s)")


if __name__ == "__main__":
    main()
