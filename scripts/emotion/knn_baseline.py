"""No-training baseline: cosine kNN on the cached, untrained emotion2vec+ utterance
embeddings, with the train split as the bank. This is the "kNN on untrained emotion2vec+ z"
row of RESULTS.md. Needs the split's embedding cache (scripts/data/extract_embeddings.py).

    python scripts/emotion/knn_baseline.py --split test
    python scripts/emotion/knn_baseline.py --split dev --k 50 --temperature 0.1
"""
import argparse

import torch

from emotion.config import load_config
from emotion.data import build_dataloaders
from emotion.trainer import ZSource, classification_metrics, embed, knn_predict, per_class_f1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="dev", help="split to query (default: dev)")
    ap.add_argument("--config", default="configs/emotion/base.yaml")
    ap.add_argument("--k", type=int, default=20)
    ap.add_argument("--temperature", type=float, default=0.07)
    ap.add_argument("overrides", nargs="*")
    args = ap.parse_args()
    cfg = load_config(args.config, args.overrides + ["data.use_cache=true", "pooling.type=mean",
                                                     f"data.val_split={args.split}"])
    torch.set_num_threads(cfg["num_threads"])
    loaders, train_df, val_df = build_dataloaders(cfg)
    bank, bank_y = embed(loaders["train_eval"], ZSource(), lambda z: z, "cpu")
    query, query_y = embed(loaders["val"], ZSource(), lambda z: z, "cpu")
    pred = knn_predict(bank, bank_y, query, args.k, args.temperature, 7).numpy()
    m = classification_metrics(query_y.numpy(), pred)
    print(f"raw emotion2vec+ z kNN (k={args.k}, T={args.temperature}) | bank {cfg['data']['train_split']} "
          f"({len(train_df)}) -> {cfg['data']['val_split']} ({len(val_df)}): "
          + " | ".join(f"{k} {v:.4f}" for k, v in m.items()))
    print("per-class F1:", per_class_f1(query_y.numpy(), pred))


if __name__ == "__main__":
    main()
