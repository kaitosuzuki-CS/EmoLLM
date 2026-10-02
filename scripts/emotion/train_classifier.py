"""Pipeline 1: frozen emotion2vec+ -> pooling (mean | attention) -> z -> projector (D)
-> linear head, cross-entropy.

    python scripts/emotion/train_classifier.py --config configs/emotion/classification.yaml [key=value ...]
"""
import itertools
import time

import torch
import torch.nn as nn

from emotion.data import EMOTIONS, build_dataloaders
from emotion.evaluation import classification_eval
from emotion.model import EmotionClassifier, build_pooler
from emotion.profiling import RunTimer
from emotion.trainer import (
    Logger,
    ZSource,
    build_encoder,
    build_optimizer,
    class_weights,
    classification_metrics,
    get_z,
    make_run_dir,
    parse_args_and_config,
    per_class_f1,
    save_stats,
    setup,
)


def main():
    cfg = parse_args_and_config("configs/emotion/classification.yaml")
    device = setup(cfg)
    run_dir = make_run_dir(cfg)
    logger = Logger(run_dir, cfg)

    loaders, train_df, val_df = build_dataloaders(cfg)
    train_loader, val_loader = loaders["train"], loaders["val"]
    print(f"train {len(train_df)} utts | val {len(val_df)} utts | run dir {run_dir}")
    encoder = build_encoder(cfg, device)
    pooler = build_pooler(cfg).to(device)
    source = ZSource(encoder, pooler)

    m = cfg["model"]
    model = EmotionClassifier(cfg["encoder"]["embed_dim"], m["hidden_dim"], m["out_dim"],
                              len(EMOTIONS), m["dropout"], m["head_dropout"]).to(device)
    weights = class_weights(train_df["label"], cfg["loss"]["class_weights"], len(EMOTIONS))
    criterion = nn.CrossEntropyLoss(
        weight=weights.to(device) if weights is not None else None,
        label_smoothing=cfg["loss"]["label_smoothing"])
    params = list(itertools.chain(model.parameters(), pooler.parameters()))
    optimizer, scheduler = build_optimizer([{"params": params, "lr": cfg["train"]["lr"]}],
                                           cfg, len(train_loader))
    timer = RunTimer(device)

    t = cfg["train"]
    best_metric, best_epoch, step = -1.0, -1, 0
    for epoch in range(1, t["epochs"] + 1):
        model.train()
        pooler.train()
        tic = time.time()
        for batch in train_loader:
            y = batch["label"].to(device)
            logits, _ = model(get_z(batch, source, device))
            loss = criterion(logits, y)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(params, t["grad_clip"])
            optimizer.step()
            scheduler.step()
            step += 1
            timer.train_clips += len(y)
            logger.log({"split": "train", "epoch": epoch, "step": step,
                        "loss": loss.item(), "lr": scheduler.get_last_lr()[0]},
                       echo=step % t["log_every"] == 0)

        timer.train_s += time.time() - tic
        timer.epochs += 1

        if epoch % t["eval_every"] == 0 or epoch == t["epochs"]:
            tic = time.time()
            pooler.eval()
            val_loss, preds, labels = classification_eval(model, val_loader, source, criterion, device)
            timer.eval_s += time.time() - tic
            metrics = classification_metrics(labels, preds)
            logger.log({"split": "val", "epoch": epoch, "step": step, "loss": val_loss, **metrics})
            if metrics[t["select_metric"]] > best_metric:
                best_metric, best_epoch = metrics[t["select_metric"]], epoch
                logger.summary({f"best/{t['select_metric']}": best_metric, "best/epoch": epoch})
                torch.save({"model": model.state_dict(), "pooler": pooler.state_dict(),
                            "config": cfg, "epoch": epoch, "metrics": metrics,
                            "per_class_f1": per_class_f1(labels, preds),
                            "emotions": EMOTIONS}, run_dir / "best.pt")

    torch.save({"model": model.state_dict(), "pooler": pooler.state_dict(), "config": cfg,
                "epoch": t["epochs"], "emotions": EMOTIONS}, run_dir / "last.pt")
    best = torch.load(run_dir / "best.pt", weights_only=False)
    print(f"best val {t['select_metric']}={best_metric:.4f} at epoch {best_epoch}")
    print(f"per-class F1: {best['per_class_f1']}")
    save_stats(cfg, run_dir, logger, timer, device, encoder=encoder, pooler=pooler, head=model,
               train_df=train_df, val_df=val_df,
               best={"epoch": best_epoch, **best["metrics"], "per_class_f1": best["per_class_f1"]})
    logger.close()


if __name__ == "__main__":
    main()
