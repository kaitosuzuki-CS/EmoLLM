"""Held-out pairwise judging: both orders per pair, a win only when the orders agree, and
win rates with bootstrap intervals."""
import numpy as np

from emollm.judge import GUIDELINES, JUDGE_SYSTEM, OpenAIChat, _parse_json, situation

PAIR_SCHEMA = {"type": "object", "properties": {
    "reason": {"type": "string"}, "winner": {"type": "string", "enum": ["1", "2", "tie"]}},
    "required": ["reason", "winner"]}


def pair_prompt(item: dict, first: str, second: str, context_turns: int) -> str:
    return (f"{situation(item, context_turns)}\n\nReply 1:\n\"{first.strip()}\"\n\nReply 2:\n"
            f"\"{second.strip()}\"\n\n{GUIDELINES}\n\nWhich reply is better for {item['speaker']}? "
            f"Weigh emotional attunement to {item['speaker']}'s emotion ({item['emotion']}) "
            f"most, then relevance to what was said and the conversation, then naturalness. "
            f"Ignore the order the replies are shown in and do not prefer a reply just for being "
            f'longer. Return JSON: {{"reason": "<one short sentence>", "winner": "1" | "2" | "tie"}}')


class PairwiseJudge(OpenAIChat):
    def __init__(self, cfg: dict):
        j = cfg["pairwise_judge"]
        super().__init__(j["base_url"], j["model"], j["max_concurrency"], j["timeout_s"])
        self.context_turns = cfg["data"]["context_turns"]
        self.gen = {"temperature": j["temperature"], "max_tokens": j["max_tokens"]}

    def _once(self, item, first, second) -> str | None:
        msgs = [{"role": "system", "content": JUDGE_SYSTEM},
                {"role": "user", "content": pair_prompt(item, first, second, self.context_turns)}]
        for _ in range(2):
            out = _parse_json(self.chat(msgs, schema=PAIR_SCHEMA, **self.gen)[0])
            if out and str(out.get("winner")) in ("1", "2", "tie"):
                return str(out["winner"])
        return None

    def compare(self, item, a: str, b: str) -> dict:
        """Judge (A, B) and (B, A). Outcome 'a' / 'b' if both orders agree, else 'tie'."""
        ab = self._once(item, a, b)        # A shown first
        ba = self._once(item, b, a)        # B shown first
        pick_ab = {"1": "a", "2": "b", "tie": "tie"}.get(ab)
        pick_ba = {"1": "b", "2": "a", "tie": "tie"}.get(ba)
        if pick_ab is None or pick_ba is None:
            outcome = None
        else:
            outcome = pick_ab if pick_ab == pick_ba else "tie"
        return {"outcome": outcome, "a_first": pick_ab, "b_first": pick_ba,
                "consistent": pick_ab is not None and pick_ab == pick_ba}


def win_rate(outcomes: list[str], n_boot: int = 2000, seed: int = 0) -> dict:
    s = np.array([{"a": 1.0, "b": 0.0, "tie": 0.5}[o] for o in outcomes])
    if not len(s):
        return {"n": 0}
    boot = np.random.default_rng(seed).choice(s, (n_boot, len(s))).mean(1)
    return {"n": len(s), "win_rate": float(s.mean()), "ci95": [float(np.percentile(boot, 2.5)),
                                                               float(np.percentile(boot, 97.5))],
            "wins": int((s == 1).sum()), "losses": int((s == 0).sum()), "ties": int((s == 0.5).sum())}
