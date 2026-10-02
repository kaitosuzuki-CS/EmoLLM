"""Pipeline 2: frozen emotion2vec+ -> pooling (mean | attention) -> z -> projector (D),
supervised contrastive loss on the L2-normalized D-dim latent space.

Validation: cosine kNN of val utterances against the train set in the latent space
(model selection), plus a logistic-regression linear probe on the best checkpoint.

    python scripts/emotion/train_supcon.py --config configs/emotion/supcon.yaml [key=value ...]
"""
import itertools
import time

import torch
import torch.nn as nn

from emotion.data import EMOTIONS, build_dataloaders
from emotion.evaluation import knn_eval, linear_probe
from emotion.losses import SupConVarCovLoss
from emotion.model import EmotionProjector, build_pooler
from emotion.profiling import RunTimer
from emotion.trainer import (
    Logger,
    ZSource,
    build_encoder,
    build_optimizer,
    get_z,
    make_run_dir,
    parse_args_and_config,
    save_stats,
    setup,
)


def main():
    cfg = parse_args_and_config("configs/emotion/supcon.yaml")
    device = setup(cfg)
    run_dir = make_run_dir(cfg)
    logger = Logger(run_dir, cfg)

    loaders, train_df, val_df = build_dataloaders(cfg)
    train_loader = loaders["train"]
    print(f"train {len(train_df)} utts | val {len(val_df)} utts | run dir {run_dir}")
    encoder = build_encoder(cfg, device)
    pooler = build_pooler(cfg).to(device)
    source = ZSource(encoder, pooler)

    m = cfg["model"]
    model = EmotionProjector(cfg["encoder"]["embed_dim"], m["hidden_dim"], m["out_dim"],
                             m["dropout"]).to(device)
    criterion = SupConVarCovLoss.from_config(cfg["loss"])
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
            loss = criterion(model(get_z(batch, source, device)), y)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(params, t["grad_clip"])
            optimizer.step()
            scheduler.step()
            step += 1
            timer.train_clips += len(y)
            logger.log({"split": "train", "epoch": epoch, "step": step,
                        "loss": loss.item(), **{k: v.item() for k, v in criterion.parts.items()},
                        "lr": scheduler.get_last_lr()[0]},
                       echo=step % t["log_every"] == 0)

        timer.train_s += time.time() - tic
        timer.epochs += 1

        if epoch % t["eval_every"] == 0 or epoch == t["epochs"]:
            tic = time.time()
            pooler.eval()
            metrics = knn_eval(model, loaders, source, cfg, device)
            timer.eval_s += time.time() - tic
            logger.log({"split": "val_knn", "epoch": epoch, "step": step, **metrics})
            if metrics[t["select_metric"]] > best_metric:
                best_metric, best_epoch = metrics[t["select_metric"]], epoch
                logger.summary({f"best/{t['select_metric']}": best_metric, "best/epoch": epoch})
                torch.save({"model": model.state_dict(), "pooler": pooler.state_dict(),
                            "config": cfg, "epoch": epoch, "knn_metrics": metrics,
                            "emotions": EMOTIONS}, run_dir / "best.pt")

    torch.save({"model": model.state_dict(), "pooler": pooler.state_dict(), "config": cfg,
                "epoch": t["epochs"], "emotions": EMOTIONS}, run_dir / "last.pt")
    print(f"best val kNN {t['select_metric']}={best_metric:.4f} at epoch {best_epoch}")

    best = torch.load(run_dir / "best.pt", weights_only=False)
    best_stats = {"epoch": best_epoch, **{f"knn_{k}": v for k, v in best["knn_metrics"].items()}}
    if cfg["eval"]["linear_probe"]:
        model.load_state_dict(best["model"])
        pooler.load_state_dict(best["pooler"])
        pooler.eval()
        save_path = run_dir / "linear_probe.pt" if cfg["eval"].get("save_probe") else None
        tic = time.time()
        probe, per_class = linear_probe(model, loaders, source, cfg, device, save_path,
                                        meta={"backbone_checkpoint": str(run_dir / "best.pt"),
                                              "epoch": best_epoch})
        timer.eval_s += time.time() - tic
        logger.log({"split": "val_linear_probe", "epoch": best_epoch, **probe})
        print(f"per-class F1 (linear probe): {per_class}")
        best_stats.update({f"probe_{k}": v for k, v in probe.items()}, per_class_f1=per_class)
    save_stats(cfg, run_dir, logger, timer, device, encoder=encoder, pooler=pooler, head=model,
               train_df=train_df, val_df=val_df, best=best_stats)
    logger.close()


if __name__ == "__main__":
    main()
