"""Training helpers shared by SFT, GRPO and evaluation."""
import math
import re

import torch

from emollm.model import completion_mask_from_ids
from emollm.prompt import _pad

# CJK / Hangul / replacement char: a sampled English reply containing these has derailed
GARBLED = re.compile(r"[\u3040-\u30ff\u4e00-\u9fff\uac00-\ud7af\ufffd]")


def warmup_cosine(optimizer, warmup_steps: int, total_steps: int):
    def f(step):
        if step < warmup_steps:
            return (step + 1) / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, f)


def eos_token_ids(policy) -> set[int]:
    ids = {policy.tokenizer.eos_token_id}
    gen_eos = policy.llm.generation_config.eos_token_id
    ids |= set(gen_eos if isinstance(gen_eos, (list, tuple)) else [gen_eos])
    return {i for i in ids if i is not None}


def sample_completions(policy, batch: dict, **gen_kwargs):
    """Generate for a left-padded prompt batch. Returns (ids (B, L), mask (B, L), texts),
    where the mask covers tokens up to and including the first EOS."""
    was_training = policy.training
    policy.eval()
    ids = policy.generate(batch, **gen_kwargs)
    policy.train(was_training)
    mask = completion_mask_from_ids(ids, eos_token_ids(policy))
    texts = [policy.tokenizer.decode(row[m.bool()], skip_special_tokens=True).strip()
             for row, m in zip(ids, mask)]
    return ids, mask, texts


def generate_texts(policy, items, collate, batch_size: int = 16, **gen_kwargs) -> list[str]:
    out = []
    for i in range(0, len(items), batch_size):
        out += sample_completions(policy, collate(items[i:i + batch_size]), **gen_kwargs)[2]
    return out


def greedy_kwargs(max_new_tokens: int) -> dict:
    return {"do_sample": False, "max_new_tokens": max_new_tokens, "temperature": None,
            "top_p": None, "top_k": None}


@torch.no_grad()
def baseline_texts(llm, tok, messages, batch_size, max_new_tokens):
    """Plain chat generation for text-only baselines (left-padded, greedy)."""
    out = []
    for i in range(0, len(messages), batch_size):
        ids = [tok(tok.apply_chat_template(m, tokenize=False, add_generation_prompt=True),
                   add_special_tokens=False)["input_ids"] for m in messages[i:i + batch_size]]
        input_ids = _pad(ids, tok.pad_token_id, left=True).to(llm.device)
        mask = _pad([[1] * len(x) for x in ids], 0, left=True).to(llm.device)
        gen = llm.generate(input_ids=input_ids, attention_mask=mask, pad_token_id=tok.pad_token_id,
                           **greedy_kwargs(max_new_tokens))[:, input_ids.shape[1]:]
        eos = {tok.eos_token_id} | set(llm.generation_config.eos_token_id or [])
        for row in gen.tolist():
            cut = next((j for j, t in enumerate(row) if t in eos), len(row))
            out.append(tok.decode(row[:cut], skip_special_tokens=True).strip())
    return out


def garbled_rate(policy, items, collate, samples: int = 4, temperature: float = 0.9,
                 top_p: float = 0.95, max_new_tokens: int = 60) -> tuple[float, list[str]]:
    """Share of sampled replies that derail into non-English garbage. A projector trained
    with too high a learning rate shows up here (greedy replies can still look fine)."""
    batch = collate([it for it in items for _ in range(samples)])
    texts = sample_completions(policy, batch, do_sample=True, temperature=temperature,
                               top_p=top_p, top_k=None, max_new_tokens=max_new_tokens)[2]
    bad = [t for t in texts if GARBLED.search(t)]
    return len(bad) / max(1, len(texts)), bad
