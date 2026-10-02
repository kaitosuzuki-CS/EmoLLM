"""Stage 2: GRPO of the emotion-conditioned LLM with judge rewards.

Starts from an SFT run: its LoRA is merged into the base weights and a fresh LoRA is
added, so the adapter-disabled model (with a frozen copy of the SFT projector) is exactly
the SFT policy and serves as the KL reference. Each step:
  1. sample `group_size` replies for each of `prompts_per_step` train utterances
  2. reward = judge_weight    * (s_true - 1) / 9
             + contrast_weight * (s_true - s_cf) / 9
             + pmi_weight      * clip(PMI, +-pmi_clip)
     s_true: judge score (1-10) given the ground-truth emotion, context and transcript.
     s_cf:   judge score of the same reply when the judge is shown a counterfactual emotion
             (valence flip or a random other emotion) as the annotation. A reply that fits
             any emotion (e.g. generic sympathy) gets ~0 contrast, so this rewards replies
             specific to the speaker's actual emotion.
     PMI:    mean per-token log p(reply | prompt, true emotion features)
                              - log p(reply | prompt, features of a train utterance with a
             different emotion), from the current policy: positive only if the reply depends
             on the emotion tokens (a policy that ignores them gets 0).
     contrast_weight = pmi_weight = 0 recovers plain judge-reward GRPO.
  3. group-relative advantage A = (r - mean_group) / std_group
  4. clipped policy-gradient loss + beta * KL(policy || SFT reference), per-token k3
     estimator, averaged per sequence then over the batch
Log-probs use logits / grpo.temperature, i.e. the distribution the replies were sampled from.
Only the new LoRA and the projector train.

    python scripts/llm/train_grpo.py --config configs/llm.yaml grpo.init_from=runs/llm_sft_<stamp>
"""
import copy
import json
import random
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from emollm.data import UtteranceDataset
from emollm.grpo import compose_rewards, group_advantages, logprobs, rows
from emollm.judge import Judge, counterfactual
from emollm.model import (
    EmotionLLM,
    add_lora,
    check_budget,
    load_llm,
    load_projector,
    load_tokenizer,
    pipeline_params,
    save_policy,
)
from emollm.prompt import Collator
from emollm.utils import generate_texts, greedy_kwargs, sample_completions, warmup_cosine
from emotion.trainer import Logger, make_run_dir, parse_args_and_config, setup


def main():
    cfg = parse_args_and_config("configs/llm.yaml")
    cfg["task"] = "llm_grpo"
    g, p = cfg["grpo"], cfg["policy"]
    if not g["init_from"]:
        raise ValueError("set grpo.init_from=<SFT run directory>")
    device = setup(cfg)
    run_dir = make_run_dir(cfg)
    logger = Logger(run_dir, cfg)

    sft = json.loads((Path(g["init_from"]) / "policy.json").read_text())
    merged = sft["merged_adapters"] + [sft["adapter"]]
    tok = load_tokenizer(sft["base_model"])
    llm = load_llm(sft["base_model"], merged, device)          # base + SFT = reference weights
    if p["gradient_checkpointing"]:
        llm.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    llm = add_lora(llm, p["lora"], dropout=g["lora_dropout"])
    ref_projector = load_projector(sft["projector"]).to(device).requires_grad_(False)
    policy = EmotionLLM(llm, copy.deepcopy(ref_projector).requires_grad_(True), tok)
    train_ds = UtteranceDataset(cfg, cfg["data"]["train_split"])
    params = pipeline_params(train_ds.emotion_params, policy)
    check_budget(params, cfg["param_budget"])

    eval_items = UtteranceDataset(cfg, cfg["data"]["val_split"], limit=g["eval_prompts"]).items
    k = policy.k
    collate = Collator(tok, cfg, k, policy.placeholder_id, "prompt")
    loader = DataLoader(train_ds, batch_size=g["prompts_per_step"], shuffle=True, drop_last=True,
                        collate_fn=lambda items: items)
    judge = Judge(cfg)
    print(f"GRPO from {g['init_from']} | train prompts {len(train_ds)} | eval prompts "
          f"{len(eval_items)} | {g['prompts_per_step']} prompts x {g['group_size']} samples/step "
          f"| run dir {run_dir}")

    lora_params = [q for q in policy.llm.parameters() if q.requires_grad]
    opt = torch.optim.AdamW([{"params": lora_params, "lr": g["lr"]},
                             {"params": policy.projector.parameters(), "lr": g["projector_lr"]}],
                            weight_decay=g["weight_decay"])
    sched = warmup_cosine(opt, g["warmup_steps"], g["steps"])
    all_params = lora_params + list(policy.projector.parameters())
    micro = g.get("micro_batch") or g["prompts_per_step"] * g["group_size"]
    rcfg = {"judge_weight": 1.0, "contrast_weight": 0.0, "contrast_label": "valence",
            "pmi_weight": 0.0, "pmi_clip": 1.0, **(g.get("reward") or {})}
    use_contrast, use_pmi = rcfg["contrast_weight"] > 0, rcfg["pmi_weight"] > 0
    feats_by_label = {}
    for it in train_ds.items:
        feats_by_label.setdefault(it["emotion"], []).append(it["emotion_feat"])
    print(f"reward: judge x{rcfg['judge_weight']} + contrast ({rcfg['contrast_label']}) "
          f"x{rcfg['contrast_weight']} + PMI (clip {rcfg['pmi_clip']}) x{rcfg['pmi_weight']}")

    def shuffled_features(group, rng):
        """Emotion features of random train utterances with a different emotion label."""
        out = []
        for it in group:
            label = rng.choice([e for e in feats_by_label if e != it["emotion"]])
            out.append(rng.choice(feats_by_label[label]))
        return torch.stack(out)
    samples_file = open(run_dir / "samples.jsonl", "a")
    gen_kwargs = {"do_sample": True, "temperature": g["temperature"], "top_p": g["top_p"],
                  "top_k": None, "max_new_tokens": g["max_new_tokens"]}

    def evaluate(step):
        tic = time.time()
        texts = generate_texts(policy, eval_items, collate, **greedy_kwargs(g["max_new_tokens"]))
        scores = [s["score"] for s in judge.score_many(eval_items, texts)]
        valid = [s for s in scores if s is not None]
        logger.log({"split": "val", "step": step, "judge_score": sum(valid) / max(1, len(valid)),
                    "judge_failures": len(scores) - len(valid),
                    "reply_words": sum(len(t.split()) for t in texts) / max(1, len(texts)),
                    "eval_min": (time.time() - tic) / 60})
        for it, t, s in list(zip(eval_items, texts, scores))[:2]:
            print(f"  [{it['emotion']}] {it['speaker']}: {it['text'][:70]!r}\n      -> {t!r} ({s})")

    def batches():
        while True:
            yield from loader

    if g["eval_every"]:
        evaluate(0)
    policy.train()
    stream, tic_run = batches(), time.time()
    for step in range(1, g["steps"] + 1):
        items = next(stream)
        group = [it for it in items for _ in range(g["group_size"])]
        batch = collate(group)

        t0 = time.time()
        ids, cmask, texts = sample_completions(policy, batch, **gen_kwargs)
        t1 = time.time()
        cf_group = [{**it, "emotion": counterfactual(f"{it['id']}:{step}", it["emotion"],
                                                     rcfg["contrast_label"], cfg["seed"])}
                    for it in group] if use_contrast else []
        # true- and counterfactual-label scores in one concurrent batch
        judged_all = judge.score_many(group + cf_group, texts + (texts if use_contrast else []))
        judged, judged_cf = judged_all[:len(group)], judged_all[len(group):]
        t2 = time.time()

        with policy.llm.disable_adapter():
            ref_lp = logprobs(policy, batch, ids, cmask, micro, g["temperature"], ref_projector)
        old_lp = logprobs(policy, batch, ids, cmask, micro, g["temperature"])
        pmi = None
        if use_pmi:  # log p(y | true emotion) - log p(y | other-emotion features), per token
            shuf = {**batch, "emotion": shuffled_features(group, random.Random(cfg["seed"] * 100003 + step))}
            shuf_lp = logprobs(policy, shuf, ids, cmask, micro, g["temperature"])
            m = cmask.to(old_lp.device).float()
            pmi = ((old_lp - shuf_lp) * m).sum(1) / m.sum(1).clamp(min=1)
        total, parts = compose_rewards([j["score"] for j in judged],
                                       [j["score"] for j in judged_cf] if use_contrast else None,
                                       pmi, rcfg)
        rewards, adv, zero_std = group_advantages(total, g["group_size"])
        adv_d, mask_f = adv.to(device), cmask.to(device).float()
        n = ids.shape[0]
        for _ in range(g["inner_updates"]):
            opt.zero_grad(set_to_none=True)
            stats = {"kl": 0.0, "clip_frac": 0.0}
            for i in range(0, n, micro):
                sl = slice(i, i + micro)
                lp = policy.completion_logprobs(rows(batch, sl), ids[sl], cmask[sl],
                                                temperature=g["temperature"])
                m = mask_f[sl]
                ratio = torch.exp(lp - old_lp[sl])
                a = adv_d[sl].unsqueeze(1)
                pg = -torch.min(ratio * a, ratio.clamp(1 - g["clip_eps"], 1 + g["clip_eps"]) * a)
                log_r = ref_lp[sl] - lp
                kl = torch.exp(log_r) - log_r - 1
                per_seq = ((pg + g["beta"] * kl) * m).sum(1) / m.sum(1).clamp(min=1)
                (per_seq.sum() / n).backward()
                with torch.no_grad():  # token-averaged per micro-batch, weighted by its share
                    w, tok_n = min(micro, n - i) / n, m.sum().clamp(min=1)
                    stats["kl"] += ((kl * m).sum() / tok_n).item() * w
                    clipped = ((ratio - 1).abs() > g["clip_eps"]).float()
                    stats["clip_frac"] += ((clipped * m).sum() / tok_n).item() * w
            grad_norm = torch.nn.utils.clip_grad_norm_(all_params, 1.0)
            opt.step()
        sched.step()
        t3 = time.time()

        scores = [j["score"] for j in judged]
        valid = [s for s in scores if s is not None]
        comp = {f"reward_{k}": torch.nanmean(v).item() for k, v in parts.items()}
        logger.log({"split": "train", "step": step, "reward": rewards.mean().item(), **comp,
                    "reward_std": rewards.view(-1, g["group_size"]).std(1).mean().item(),
                    "judge_score": sum(valid) / max(1, len(valid)),
                    "judge_failures": len(scores) - len(valid), "zero_std_groups": zero_std,
                    "kl": stats["kl"], "clip_frac": stats["clip_frac"], "grad_norm": float(grad_norm),
                    "completion_tokens": cmask.sum(1).float().mean().item(),
                    "lr": sched.get_last_lr()[0], "gen_s": t1 - t0, "judge_s": t2 - t1,
                    "update_s": t3 - t2}, echo=step % g["log_every"] == 0)
        for i, (it, t, j, r, a) in enumerate(zip(group, texts, judged, rewards.tolist(), adv.tolist())):
            rec = {"step": step, "id": it["id"], "emotion": it["emotion"], "reply": t,
                   "score": j["score"], "reason": j["reason"], "reward": r, "advantage": a}
            if use_contrast:
                rec.update(cf_emotion=cf_group[i]["emotion"], cf_score=judged_cf[i]["score"])
            if use_pmi:
                rec["pmi"] = float(pmi[i])
            samples_file.write(json.dumps(rec) + "\n")
        samples_file.flush()
        if g["eval_every"] and step % g["eval_every"] == 0:
            evaluate(step)
        if step % g["save_every"] == 0 or step == g["steps"]:
            save_policy(run_dir, policy, sft["base_model"], merged,
                        extra={"stage": "grpo", "init_from": str(g["init_from"]), "step": step,
                               "reward": rcfg,
                               "emotion_checkpoint": train_ds.emotion_checkpoint,
                               "emotion_feature": cfg["emotion"]["feature"]})

    stats = {"params": params, "steps": g["steps"], "train_min": (time.time() - tic_run) / 60,
             "peak_gpu_mem_gb": torch.cuda.max_memory_allocated() / 1e9 if device.type == "cuda" else None}
    (run_dir / "stats.json").write_text(json.dumps(stats, indent=2))
    print(f"saved {run_dir / 'adapter'} | {stats['train_min']:.1f} min, peak {stats['peak_gpu_mem_gb']:.1f} GB")
    logger.close()


if __name__ == "__main__":
    main()
