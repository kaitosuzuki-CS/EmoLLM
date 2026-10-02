"""Pipeline 3: fine-tune emotion2vec+ end to end. Raw audio is loaded per batch,
encoded by the trainable backbone into frames, pooled (mean | attention) into z, then
projector (D) and either
  objective: classification -> linear head, cross-entropy
  objective: supcon         -> supervised contrastive loss on the normalized D-dim latent

The CNN feature extractor stays frozen (encoder.freeze_feature_extractor); the
transformer trains at train.encoder_lr, heads at train.lr. For the first
train.freeze_encoder_epochs epochs only the heads train. Heads can be initialized
from a frozen-encoder run via train.init_head_from.

    python scripts/emotion/train_finetune.py --config configs/emotion/finetune.yaml [key=value ...]
    python scripts/emotion/train_finetune.py --config configs/emotion/finetune_supcon.yaml
"""
import time

import torch
import torch.nn as nn

from emotion.data import EMOTIONS, build_dataloaders
from emotion.evaluation import classification_eval, knn_eval, linear_probe
from emotion.losses import SupConVarCovLoss
from emotion.model import FinetuneModel, build_head, build_pooler
from emotion.profiling import RunTimer
from emotion.trainer import (
    Logger,
    build_encoder,
    build_optimizer,
    class_weights,
    classification_metrics,
    make_run_dir,
    parse_args_and_config,
    per_class_f1,
    save_stats,
    setup,
)


def build_criterion(cfg, train_df, device):
    loss = cfg["loss"]
    if cfg["objective"] == "supcon":
        return SupConVarCovLoss.from_config(loss)
    weights = class_weights(train_df["label"], loss["class_weights"], len(EMOTIONS))
    return nn.CrossEntropyLoss(weight=weights.to(device) if weights is not None else None,
                               label_smoothing=loss["label_smoothing"])


def validate(model, loaders, criterion, cfg, device):
    model.eval()
    if cfg["objective"] == "supcon":
        return {"split": "val_knn", **knn_eval(model.head, loaders, model.source, cfg, device)}, None
    val_loss, preds, labels = classification_eval(model.head, loaders["val"], model.source,
                                                  criterion, device)
    metrics = {"split": "val", "loss": val_loss, **classification_metrics(labels, preds)}
    return metrics, per_class_f1(labels, preds)


def main():
    cfg = parse_args_and_config("configs/emotion/finetune.yaml")
    if cfg["data"]["use_cache"]:
        raise ValueError("fine-tuning needs raw audio: set data.use_cache: false")
    device = setup(cfg)
    run_dir = make_run_dir(cfg)
    logger = Logger(run_dir, cfg)

    loaders, train_df, val_df = build_dataloaders(cfg)
    train_loader = loaders["train"]
    print(f"objective {cfg['objective']} | train {len(train_df)} utts | val {len(val_df)} utts "
          f"| {len(train_loader)} steps/epoch | run dir {run_dir}")

    t = cfg["train"]
    head = build_head(cfg)
    if t["init_head_from"]:
        ckpt = torch.load(t["init_head_from"], map_location="cpu", weights_only=False)
        head.load_state_dict(ckpt["model"])
        print(f"initialized head from {t['init_head_from']} (epoch {ckpt['epoch']})")
    encoder = build_encoder(cfg, device, trainable=True)
    model = FinetuneModel(encoder, head, build_pooler(cfg)).to(device)
    criterion = build_criterion(cfg, train_df, device)

    enc_params = [p for p in encoder.parameters() if p.requires_grad]
    print(f"trainable params: encoder {sum(p.numel() for p in enc_params) / 1e6:.1f}M, "
          f"pooler {sum(p.numel() for p in model.pooler.parameters()) / 1e6:.2f}M, "
          f"head {sum(p.numel() for p in head.parameters()) / 1e6:.2f}M")
    optimizer, scheduler = build_optimizer([
        {"params": [*head.parameters(), *model.pooler.parameters()], "lr": t["lr"]},
        {"params": enc_params, "lr": t["encoder_lr"], "start_epoch": t["freeze_encoder_epochs"]},
    ], cfg, len(train_loader))
    all_params = [p for g in optimizer.param_groups for p in g["params"]]
    timer = RunTimer(device)

    best_metric, best_epoch, step = -1.0, -1, 0
    for epoch in range(1, t["epochs"] + 1):
        encoder.set_trainable(epoch > t["freeze_encoder_epochs"])
        model.train()
        tic, n_clips = time.time(), 0
        for batch in train_loader:
            y = batch["label"].to(device)
            out = model(batch, device)
            loss = criterion(out[0] if isinstance(out, tuple) else out, y)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(all_params, t["grad_clip"])
            optimizer.step()
            scheduler.step()
            step += 1
            n_clips += len(y)
            timer.train_clips += len(y)
            head_lr, enc_lr = scheduler.get_last_lr()
            logger.log({"split": "train", "epoch": epoch, "step": step, "loss": loss.item(),
                        **{k: v.item() for k, v in getattr(criterion, "parts", {}).items()},
                        "lr": head_lr, "encoder_lr": enc_lr,
                        "clips_per_s": n_clips / (time.time() - tic)},
                       echo=step % t["log_every"] == 0)

        timer.train_s += time.time() - tic
        timer.epochs += 1

        if epoch % t["eval_every"] == 0 or epoch == t["epochs"]:
            eval_tic = time.time()
            metrics, per_class = validate(model, loaders, criterion, cfg, device)
            timer.eval_s += time.time() - eval_tic
            logger.log({**metrics, "epoch": epoch, "step": step,
                        "epoch_min": (time.time() - tic) / 60})
            if metrics[t["select_metric"]] > best_metric:
                best_metric, best_epoch = metrics[t["select_metric"]], epoch
                logger.summary({f"best/{t['select_metric']}": best_metric, "best/epoch": epoch})
                torch.save({"model": model.state_dict(), "config": cfg, "epoch": epoch,
                            "metrics": metrics, "per_class_f1": per_class,
                            "emotions": EMOTIONS}, run_dir / "best.pt")

    torch.save({"model": model.state_dict(), "config": cfg, "epoch": t["epochs"],
                "emotions": EMOTIONS}, run_dir / "last.pt")
    print(f"best val {t['select_metric']}={best_metric:.4f} at epoch {best_epoch}")

    best = torch.load(run_dir / "best.pt", map_location="cpu", weights_only=False)
    metric_keys = ("acc", "weighted_f1", "macro_f1")
    if cfg["objective"] == "classification":
        print(f"per-class F1: {best['per_class_f1']}")
        best_stats = {"epoch": best_epoch, **{k: best["metrics"][k] for k in metric_keys},
                      "per_class_f1": best["per_class_f1"]}
    else:
        best_stats = {"epoch": best_epoch,
                      **{f"knn_{k}": best["metrics"][k] for k in metric_keys}}
    if cfg["objective"] == "supcon" and cfg["eval"]["linear_probe"]:
        model.load_state_dict(best["model"])
        model.eval()
        save_path = run_dir / "linear_probe.pt" if cfg["eval"].get("save_probe") else None
        tic = time.time()
        probe, per_class = linear_probe(model.head, loaders, model.source, cfg, device,
                                        save_path,
                                        meta={"backbone_checkpoint": str(run_dir / "best.pt"),
                                              "epoch": best_epoch})
        timer.eval_s += time.time() - tic
        logger.log({"split": "val_linear_probe", "epoch": best_epoch, **probe})
        print(f"per-class F1 (linear probe): {per_class}")
        best_stats.update({f"probe_{k}": v for k, v in probe.items()}, per_class_f1=per_class)
    save_stats(cfg, run_dir, logger, timer, device, encoder=encoder, pooler=model.pooler,
               head=head, train_df=train_df, val_df=val_df, best=best_stats)
    logger.close()


if __name__ == "__main__":
    main()
