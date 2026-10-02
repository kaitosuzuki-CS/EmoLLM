"""Stage 1: supervised fine-tuning of the emotion-conditioned LLM on cached judge pseudo labels.

    frozen emotion model latent (cached) -> EmotionPromptProjector -> K soft tokens
    [system + context + transcript + soft tokens] -> Qwen2.5-3B-Instruct + LoRA -> reply

Trains the LoRA adapters and the projector with cross-entropy on the pseudo-label reply
only. adapter/, projector.pt and policy.json hold the checkpoint with the lowest dev loss
(re-saved whenever it improves; GRPO starts from it); last_adapter/, last_projector.pt and
last_policy.json hold the final step. Also writes stats.json.

    python scripts/llm/train_sft.py --config configs/llm.yaml [key=value ...]
"""
import json
import math
import time

import torch
from torch.utils.data import DataLoader

from emollm.data import UtteranceDataset
from emollm.model import (
    EmotionLLM,
    add_lora,
    check_budget,
    load_llm,
    load_tokenizer,
    new_projector,
    pipeline_params,
    save_policy,
)
from emollm.prompt import Collator
from emollm.utils import garbled_rate, generate_texts, greedy_kwargs, warmup_cosine
from emotion.trainer import Logger, make_run_dir, parse_args_and_config, setup


@torch.no_grad()
def dev_loss(policy, loader, max_batches):
    policy.eval()
    total, n = 0.0, 0
    for i, batch in enumerate(loader):
        if max_batches and i >= max_batches:
            break
        out = policy(batch["input_ids"], batch["attention_mask"], batch["emotion_mask"],
                     batch["emotion"], labels=batch["labels"])
        tokens = int((batch["labels"] != -100).sum())
        total, n = total + out.loss.item() * tokens, n + tokens
    policy.train()
    return total / max(1, n)


def main():
    cfg = parse_args_and_config("configs/llm.yaml")
    cfg["task"] = "llm_sft"
    device = setup(cfg)
    run_dir = make_run_dir(cfg)
    logger = Logger(run_dir, cfg)
    s, p = cfg["sft"], cfg["policy"]

    train_ds = UtteranceDataset(cfg, cfg["data"]["train_split"], require_response=True,
                                limit=s["train_limit"])
    val_ds = UtteranceDataset(cfg, cfg["data"]["val_split"], require_response=True)
    if not len(train_ds):
        raise ValueError("no pseudo-labelled train utterances; run scripts/llm/make_pseudo_labels.py")
    print(f"train {len(train_ds)} labelled utts | dev {len(val_ds)} labelled utts | run dir {run_dir}")

    tok = load_tokenizer(p["model"])
    llm = load_llm(p["model"], device=device)
    if p["gradient_checkpointing"]:
        llm.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    llm = add_lora(llm, p["lora"])
    policy = EmotionLLM(llm, new_projector(cfg, train_ds.feature_dim, llm).to(device), tok)
    params = pipeline_params(train_ds.emotion_params, policy)
    check_budget(params, cfg["param_budget"])

    collate = Collator(tok, cfg, p["num_emotion_tokens"], policy.placeholder_id, "sft")
    prompt_collate = Collator(tok, cfg, p["num_emotion_tokens"], policy.placeholder_id, "prompt")
    loader = DataLoader(train_ds, batch_size=s["batch_size"], shuffle=True, collate_fn=collate)
    val_loader = DataLoader(val_ds, batch_size=s["batch_size"], shuffle=False, collate_fn=collate)
    steps_per_epoch = math.ceil(len(loader) / s["grad_accum"])
    total_steps = s["max_steps"] or s["epochs"] * steps_per_epoch

    lora_params = [q for q in policy.llm.parameters() if q.requires_grad]
    opt = torch.optim.AdamW([{"params": lora_params, "lr": s["lr"]},
                             {"params": policy.projector.parameters(), "lr": s["projector_lr"]}],
                            weight_decay=s["weight_decay"])
    sched = warmup_cosine(opt, int(s["warmup_ratio"] * total_steps), total_steps)
    all_params = lora_params + list(policy.projector.parameters())
    sample_items = val_ds.items[:s["num_samples"]] or train_ds.items[:s["num_samples"]]
    samples_file = open(run_dir / "samples.jsonl", "a")

    meta_extra = {"stage": "sft", "emotion_checkpoint": train_ds.emotion_checkpoint,
                  "emotion_feature": cfg["emotion"]["feature"]}
    best = {"loss": float("inf"), "step": None}

    def evaluate(step):
        if not len(val_ds):
            return
        loss = dev_loss(policy, val_loader, s["eval_batches"])
        logger.log({"split": "val", "step": step, "loss": loss})
        if loss < best["loss"]:   # keep the best-dev-loss checkpoint as the run's policy
            best.update(loss=loss, step=step)
            save_policy(run_dir, policy, p["model"], [], extra={**meta_extra, "step": step,
                                                                "dev_loss": loss})
            # sampled (not greedy) replies derail into garbage if the soft tokens drift
            rate, bad = garbled_rate(policy, sample_items, prompt_collate)
            policy.train()
            logger.log({"split": "val", "step": step, "garbled_rate": rate})
            logger.summary({"best/dev_loss": loss, "best/step": step, "best/garbled_rate": rate})
            print(f"  new best dev loss {loss:.4f} at step {step}: saved {run_dir / 'adapter'} "
                  f"(garbled samples {rate:.0%}{', e.g. ' + repr(bad[0][:60]) if bad else ''})")

    def sample(step):
        texts = generate_texts(policy, sample_items, prompt_collate,
                               **greedy_kwargs(cfg["generation"]["max_new_tokens"]))
        for it, t in zip(sample_items, texts):
            samples_file.write(json.dumps({"step": step, "id": it["id"], "emotion": it["emotion"],
                                           "text": it["text"], "reply": t,
                                           "pseudo_label": it["response"]}) + "\n")
            print(f"  [{it['emotion']}] {it['speaker']}: {it['text'][:80]!r}\n      -> {t!r}")
        samples_file.flush()

    policy.train()
    step, micro, tokens, tic = 0, 0, 0, time.time()
    done = False
    for epoch in range(1, 10**9):
        for batch in loader:
            out = policy(batch["input_ids"], batch["attention_mask"], batch["emotion_mask"],
                         batch["emotion"], labels=batch["labels"])
            (out.loss / s["grad_accum"]).backward()
            micro += 1
            tokens += int((batch["labels"] != -100).sum())
            if micro % s["grad_accum"]:
                continue
            grad_norm = torch.nn.utils.clip_grad_norm_(all_params, 1.0)
            opt.step()
            sched.step()
            opt.zero_grad(set_to_none=True)
            step += 1
            logger.log({"split": "train", "epoch": epoch, "step": step, "loss": out.loss.item(),
                        "grad_norm": float(grad_norm), "lr": sched.get_last_lr()[0],
                        "projector_lr": sched.get_last_lr()[1],
                        "target_tokens_per_s": tokens / (time.time() - tic)},
                       echo=step % s["log_every"] == 0)
            if step % s["eval_every"] == 0:
                evaluate(step)
            if step % s["sample_every"] == 0:
                sample(step)
            if step >= total_steps:
                done = True
                break
        if done:
            break

    if step % s["eval_every"]:
        evaluate(step)
    if step % s["sample_every"]:
        sample(step)
    save_policy(run_dir, policy, p["model"], [], tag="last_", extra={**meta_extra, "step": step})
    if best["step"] is None:  # no dev labels: the final step is the policy
        save_policy(run_dir, policy, p["model"], [], extra={**meta_extra, "step": step})
    meta = json.loads((run_dir / "policy.json").read_text())
    stats = {"params": params, "steps": step, "epochs_seen": step / steps_per_epoch,
             "best_step": best["step"], "best_dev_loss": best["loss"] if best["step"] else None,
             "train_min": (time.time() - tic) / 60, "target_tokens": tokens,
             "peak_gpu_mem_gb": torch.cuda.max_memory_allocated() / 1e9 if device.type == "cuda" else None}
    (run_dir / "stats.json").write_text(json.dumps(stats, indent=2))
    logger.summary({f"stats/{k}": v for k, v in stats.items() if isinstance(v, (int, float))})
    print(f"policy (best dev loss {best['loss']:.4f} at step {best['step']}): {meta['adapter']} | "
          f"last step: {run_dir / 'last_adapter'} | {stats['train_min']:.1f} min, "
          f"peak {stats['peak_gpu_mem_gb']:.1f} GB")
    logger.close()


if __name__ == "__main__":
    main()
