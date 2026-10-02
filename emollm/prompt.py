"""Chat prompts with a slot for the K emotion soft-prompt tokens."""
import torch

EMOTION_MARKER = "<<EMOTION>>"


def user_content(item: dict, context_turns: int, emotion_slot: str) -> str:
    lines = []
    context = item["context"][-context_turns:] if context_turns else []
    if context:
        lines.append("Conversation so far:")
        lines += [f"{t['speaker']}: {t['text']}" for t in context]
        lines.append("")
    speaker = item["speaker"]
    lines.append(f'{speaker} now says: "{item["text"]}"')
    lines.append(f"{speaker}'s vocal emotion: {emotion_slot}")
    lines.append(f"Reply to {speaker}.")
    return "\n".join(lines)


def policy_messages(item: dict, cfg: dict) -> list[dict]:
    """The policy's prompt: the emotion is only available through the soft tokens."""
    return [{"role": "system", "content": cfg["prompt"]["system"]},
            {"role": "user", "content": user_content(item, cfg["data"]["context_turns"],
                                                     EMOTION_MARKER)}]


def candidate_messages(item: dict, cfg: dict, emotion: str) -> list[dict]:
    """Prompt for pseudo-label candidates from the base model (emotion as a word)."""
    return [{"role": "system", "content": cfg["prompt"]["candidate_system"]},
            {"role": "user", "content": user_content(item, cfg["data"]["context_turns"], emotion)}]


def no_emotion_messages(item: dict, cfg: dict) -> list[dict]:
    """Text-only baseline without any emotion information."""
    lines = user_content(item, cfg["data"]["context_turns"], "").split("\n")
    user = "\n".join(line for line in lines if not line.endswith("vocal emotion: "))
    system = cfg["prompt"]["candidate_system"].split(" Besides the words")[0] + (
        " Reply directly to the speaker in 1-3 short sentences, stay grounded in what they said "
        "and the conversation, and be natural rather than formulaic.")
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def encode_prompt(tokenizer, messages: list[dict], k: int, placeholder_id: int):
    """Chat-template the messages and replace EMOTION_MARKER with K placeholder tokens.
    Returns (input_ids, emotion_mask) as lists; the placeholders' embeddings are later
    overwritten with the projected emotion tokens."""
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    pre, post = text.split(EMOTION_MARKER)
    pre_ids = tokenizer(pre, add_special_tokens=False)["input_ids"]
    post_ids = tokenizer(post, add_special_tokens=False)["input_ids"]
    ids = pre_ids + [placeholder_id] * k + post_ids
    mask = [0] * len(pre_ids) + [1] * k + [0] * len(post_ids)
    return ids, mask


def _pad(seqs, value, left: bool):
    n = max(len(s) for s in seqs)
    return torch.tensor([[value] * (n - len(s)) + s if left else s + [value] * (n - len(s))
                         for s in seqs])


class Collator:
    """Batches utterance items for the policy.
    mode="prompt": left-padded prompts (generation, GRPO).
    mode="sft": right-padded prompt + pseudo-label response, labels only on the response."""

    def __init__(self, tokenizer, cfg: dict, k: int, placeholder_id: int, mode: str):
        self.tok, self.cfg, self.k, self.ph, self.mode = tokenizer, cfg, k, placeholder_id, mode
        self.pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None \
            else tokenizer.eos_token_id

    def __call__(self, items):
        ids, masks, labels = [], [], []
        for it in items:
            p_ids, p_mask = encode_prompt(self.tok, policy_messages(it, self.cfg), self.k, self.ph)
            if self.mode == "sft":
                r_ids = self.tok(it["response"] + self.tok.eos_token,
                                 add_special_tokens=False)["input_ids"]
                ids.append(p_ids + r_ids)
                masks.append(p_mask + [0] * len(r_ids))
                labels.append([-100] * len(p_ids) + r_ids)
            else:
                ids.append(p_ids)
                masks.append(p_mask)
        left = self.mode != "sft"
        input_ids = _pad(ids, self.pad_id, left)
        batch = {
            "input_ids": input_ids,
            "attention_mask": _pad([[1] * len(s) for s in ids], 0, left),
            "emotion_mask": _pad(masks, 0, left).bool(),
            "emotion": torch.stack([it["emotion_feat"] for it in items]),
            "items": items,
        }
        if self.mode == "sft":
            batch["labels"] = _pad(labels, -100, left)
        return batch
