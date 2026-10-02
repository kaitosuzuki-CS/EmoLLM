"""Judge LLM (OpenAI-compatible server, e.g. vLLM) for pseudo labels and GRPO rewards.
The judge sees the ground-truth (annotated) emotion, the dialogue context and the
transcript; the policy only gets the emotion through its soft-prompt tokens."""
import json
import random
import re
from concurrent.futures import ThreadPoolExecutor

EMOTIONS = ["neutral", "joy", "surprise", "anger", "sadness", "disgust", "fear"]

# Valence flip: positive <-> negative; neutral and surprise go to a clearly opposite state.
VALENCE_FLIP = {"joy": "sadness", "surprise": "sadness", "neutral": "anger", "sadness": "joy",
                "anger": "joy", "disgust": "joy", "fear": "joy"}


def counterfactual(key: str, emotion: str, scheme: str, seed: int = 0) -> str:
    """A different emotion label: the valence flip, or a random other emotion that is stable
    for a given (seed, key)."""
    if scheme == "valence":
        return VALENCE_FLIP[emotion]
    if scheme != "random":
        raise ValueError(f"unknown counterfactual scheme {scheme!r}")
    return random.Random(f"{seed}:{key}").choice([e for e in EMOTIONS if e != emotion])

JUDGE_SYSTEM = ("You are an expert in empathetic, emotionally intelligent conversation. "
                "You evaluate and write replies an assistant gives to a person in a conversation.")

GUIDELINES = ("A good reply: 1-3 short sentences; responds directly to the speaker and is "
              "specific to what they said and the conversation; attuned to how they feel "
              "without naming the emotion mechanically or saying it was detected; natural, not "
              "formulaic or preachy; no emojis, stage directions or role labels.")

SCORE_RUBRIC = ("Score the reply from 1 (bad) to 10 (excellent) on emotional attunement to "
                "the speaker's annotated emotion, relevance and grounding in what was said, and "
                "naturalness and concision. Deduct heavily if it ignores or contradicts the "
                "speaker's emotion, is generic or off-topic, is longer than about 3 sentences, "
                "says an emotion was detected or mentions labels/tokens, or is empty, garbled "
                "or repetitive.")

PSEUDO_SCHEMA = {"type": "object", "properties": {
    "best_candidate": {"type": "integer"}, "reply": {"type": "string"}},
    "required": ["best_candidate", "reply"]}
SCORE_SCHEMA = {"type": "object", "properties": {
    "reason": {"type": "string"}, "score": {"type": "integer", "minimum": 1, "maximum": 10}},
    "required": ["reason", "score"]}


def situation(item: dict, context_turns: int) -> str:
    lines = []
    context = item["context"][-context_turns:] if context_turns else []
    if context:
        lines.append("Conversation so far:")
        lines += [f"{t['speaker']}: {t['text']}" for t in context]
        lines.append("")
    lines.append(f'{item["speaker"]} now says: "{item["text"]}"')
    lines.append(f"{item['speaker']}'s emotion (human-annotated): {item['emotion']}")
    return "\n".join(lines)


def pseudo_label_prompt(item: dict, candidates: list[str], context_turns: int) -> str:
    cands = "\n".join(f"[{i + 1}] {c.strip()}" for i, c in enumerate(candidates))
    return (f"{situation(item, context_turns)}\n\nCandidate replies from the assistant:\n{cands}"
            f"\n\n{GUIDELINES}\n\nWrite the best reply the assistant could give to "
            f"{item['speaker']}. Start from the strongest candidate and improve it only as much "
            f"as needed; write a new reply if every candidate is poor. Return JSON: "
            f'{{"best_candidate": <1-{len(candidates)}>, "reply": "<final reply>"}}')


def score_prompt(item: dict, response: str, context_turns: int) -> str:
    return (f"{situation(item, context_turns)}\n\nAssistant's reply to {item['speaker']}:\n"
            f'"{response.strip()}"\n\n{GUIDELINES}\n\n{SCORE_RUBRIC}\n\nReturn JSON: '
            f'{{"reason": "<one short sentence>", "score": <integer 1-10>}}')


def _parse_json(text: str) -> dict | None:
    try:
        return json.loads(text)
    except (json.JSONDecodeError, TypeError):
        m = re.search(r"\{.*\}", text or "", re.S)
        if m:
            try:
                return json.loads(m.group(0))
            except json.JSONDecodeError:
                return None
    return None


class OpenAIChat:
    """Thin threaded wrapper over an OpenAI-compatible chat endpoint."""

    def __init__(self, base_url: str, model: str, max_concurrency: int = 32, timeout_s: float = 300):
        from openai import OpenAI
        self.client = OpenAI(base_url=base_url, api_key="EMPTY", timeout=timeout_s, max_retries=3)
        self.model = model
        self.max_concurrency = max_concurrency

    def chat(self, messages, schema=None, **kw) -> list[str]:
        if schema is not None:
            kw["response_format"] = {"type": "json_schema",
                                     "json_schema": {"name": "out", "schema": schema}}
        out = self.client.chat.completions.create(model=self.model, messages=messages, **kw)
        return [c.message.content or "" for c in out.choices]

    def map(self, fn, xs):
        with ThreadPoolExecutor(self.max_concurrency) as pool:
            return list(pool.map(fn, xs))


class Judge(OpenAIChat):
    def __init__(self, cfg: dict):
        j = cfg["judge"]
        super().__init__(j["base_url"], j["model"], j["max_concurrency"], j["timeout_s"])
        self.context_turns = cfg["data"]["context_turns"]
        self.gen = {"temperature": j["temperature"], "max_tokens": j["max_tokens"]}

    def _ask(self, prompt, schema):
        msgs = [{"role": "system", "content": JUDGE_SYSTEM}, {"role": "user", "content": prompt}]
        for _ in range(2):  # one retry on unparseable output
            parsed = _parse_json(self.chat(msgs, schema=schema, **self.gen)[0])
            if parsed is not None:
                return parsed
        return None

    def pseudo_label(self, item: dict, candidates: list[str]) -> dict | None:
        out = self._ask(pseudo_label_prompt(item, candidates, self.context_turns), PSEUDO_SCHEMA)
        if not out or not str(out.get("reply", "")).strip():
            return None
        best = out.get("best_candidate")
        return {"reply": out["reply"].strip(),
                "best_candidate": best if isinstance(best, int) and 1 <= best <= len(candidates)
                else None}

    def score(self, item: dict, response: str) -> dict:
        """{"score": 1-10 or None if unparseable, "reason": str}. Empty replies score 1."""
        if not response.strip():
            return {"score": 1, "reason": "empty reply"}
        out = self._ask(score_prompt(item, response, self.context_turns), SCORE_SCHEMA)
        try:
            score = min(10, max(1, int(out["score"])))
        except (TypeError, KeyError, ValueError):
            return {"score": None, "reason": "unparseable judge output"}
        return {"score": score, "reason": str(out.get("reason", ""))}

    def score_many(self, items: list[dict], responses: list[str]) -> list[dict]:
        return self.map(lambda x: self.score(*x), list(zip(items, responses)))
