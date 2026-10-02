"""Reply-generating systems for evaluation, named by a spec string:

    runs/llm_grpo_<stamp>             a trained SFT or GRPO policy, fed its own emotion features
    runs/llm_grpo_<stamp>:shuffled    the same policy, fed other utterances' emotion features
    baseline:none                     the base LLM, text only, no emotion information
    baseline:predicted                the base LLM, told the emotion model's prediction as a word
    baseline:oracle                   the base LLM, told the ground-truth emotion (upper bound)
    path/to/replies.jsonl             precomputed replies (one {"id", "reply"} per line)

Replies are generated greedily once per system and split and cached as JSONL
(<run>/replies_<split>[_shuffled].jsonl, or runs/llm_baselines/replies_<split>_<baseline>.jsonl),
so evaluate.py and evaluate_pairwise.py share them.
"""
import json
from dataclasses import dataclass
from pathlib import Path

import torch

from emollm.data import UtteranceDataset, read_jsonl
from emotion.config import load_config

BASELINES = ("none", "predicted", "oracle")


@dataclass
class System:
    spec: str
    kind: str            # policy | baseline | file
    path: Path | None    # run directory or reply file
    variant: str = ""    # shuffled (policy) | none / predicted / oracle (baseline)

    @property
    def name(self) -> str:
        if self.kind == "baseline":
            return f"baseline_{self.variant}"
        if self.kind == "file":
            return self.path.stem.removeprefix("replies_").removeprefix("eval_")
        return self.path.name + ("_shuffled" if self.variant else "")

    def reply_cache(self, split: str, baseline_dir="runs/llm_baselines") -> Path:
        if self.kind == "file":
            return self.path
        if self.kind == "baseline":
            return Path(baseline_dir) / f"replies_{split}_{self.variant}.jsonl"
        return self.path / f"replies_{split}{'_shuffled' if self.variant else ''}.jsonl"


def parse_system(spec: str) -> System:
    if spec.startswith("baseline:"):
        variant = spec.split(":", 1)[1]
        if variant not in BASELINES:
            raise ValueError(f"unknown baseline {variant!r}; choose from {', '.join(BASELINES)}")
        return System(spec, "baseline", None, variant)
    path, _, variant = spec.partition(":")
    if variant not in ("", "shuffled"):
        raise ValueError(f"unknown policy variant {variant!r} in {spec!r} (only :shuffled)")
    p = Path(path)
    if p.suffix == ".jsonl":
        if variant:
            raise ValueError("a reply file cannot be shuffled")
        if not p.exists():
            raise FileNotFoundError(p)
        return System(spec, "file", p)
    if not (p / "policy.json").exists():
        raise FileNotFoundError(f"{p} is not a policy run (no policy.json), a reply .jsonl "
                                f"or baseline:{{{','.join(BASELINES)}}}")
    return System(spec, "policy", p, variant)


def system_config(system: System, config: str, overrides: list[str]) -> dict:
    """The config used to generate a system's replies. A policy always gets the emotion features
    it was trained on (feature cache and feature type from the run's own config), whatever
    `emotion.*` overrides say: those apply to baselines (e.g. which model's predictions
    baseline:predicted uses), so one override list can serve both systems of a comparison."""
    cfg = load_config(config, overrides)
    if system.kind == "policy":
        run_emotion = load_config(str(system.path / "config.yaml"))["emotion"]
        cfg["emotion"]["cache_dir"] = str(Path(run_emotion["cache_dir"]).expanduser())
        cfg["emotion"]["feature"] = run_emotion["feature"]
        if not Path(cfg["emotion"]["cache_dir"]).exists():
            raise FileNotFoundError(f"{system.path} was trained on features in {cfg['emotion']['cache_dir']}, "
                                    "which no longer exists; re-create it with scripts/llm/extract_features.py "
                                    "(same emotion checkpoint and emotion.cache_dir)")
    return cfg


def check_features(cfg: dict, what: str):
    """Baselines still read transcripts, dialogue context and (baseline:predicted) the emotion
    model's predictions from a feature cache made by scripts/llm/extract_features.py."""
    if not Path(cfg["emotion"]["cache_dir"]).expanduser().exists():
        raise FileNotFoundError(f"{what}: no feature cache at {cfg['emotion']['cache_dir']}; pass "
                                "emotion.cache_dir=<a directory made by scripts/llm/extract_features.py>")


def _generate(system: System, cfg: dict, items: list[dict], batch_size: int) -> list[str]:
    from emollm.model import load_llm, load_policy, load_tokenizer
    from emollm.prompt import Collator, candidate_messages, no_emotion_messages
    from emollm.utils import baseline_texts, generate_texts, greedy_kwargs

    max_new = cfg["generation"]["max_new_tokens"]
    if system.kind == "policy":
        policy, _ = load_policy(system.path)
        if system.variant == "shuffled":  # every utterance gets another utterance's features
            perm = torch.randperm(len(items), generator=torch.Generator().manual_seed(0))
            feats = [items[j]["emotion_feat"] for j in perm.tolist()]
            items = [{**it, "emotion_feat": f} for it, f in zip(items, feats)]
        collate = Collator(policy.tokenizer, cfg, policy.k, policy.placeholder_id, "prompt")
        texts = generate_texts(policy, items, collate, batch_size, **greedy_kwargs(max_new))
        del policy
    else:
        tok = load_tokenizer(cfg["policy"]["model"])
        llm = load_llm(cfg["policy"]["model"]).eval()
        if system.variant == "none":
            messages = [no_emotion_messages(it, cfg) for it in items]
        else:
            key = "pred_emotion" if system.variant == "predicted" else "emotion"
            messages = [candidate_messages(it, cfg, it[key]) for it in items]
        texts = baseline_texts(llm, tok, messages, batch_size, max_new)
        del llm
    torch.cuda.empty_cache()
    return texts


def replies(system: System, cfg: dict, split: str, limit: int | None = None,
            batch_size: int = 16, regenerate: bool = False) -> tuple[list[dict], list[str]]:
    """(items, replies) on the first `limit` utterances of `split` (all when None/0). Uses the
    system's reply cache when it covers those utterances, otherwise generates and caches."""
    items = UtteranceDataset(cfg, split, limit=limit or None).items
    cache = system.reply_cache(split)
    cached = {r["id"]: r["reply"] for r in read_jsonl(cache)}
    if system.kind == "file":
        items = [it for it in items if it["id"] in cached]
        if not items:
            raise ValueError(f"{cache} has no replies for {split} utterances")
        return items, [cached[it["id"]] for it in items]
    if not regenerate and all(it["id"] in cached for it in items):
        print(f"{system.name}: {len(items)} cached replies from {cache}")
        return items, [cached[it["id"]] for it in items]
    print(f"{system.name}: generating {len(items)} replies on {split}")
    texts = _generate(system, cfg, items, batch_size)
    if len(cached) <= len(items):   # never overwrite a larger cache with a smaller one
        cache.parent.mkdir(parents=True, exist_ok=True)
        with open(cache, "w") as f:
            for it, t in zip(items, texts):
                f.write(json.dumps({"id": it["id"], "emotion": it["emotion"], "reply": t}) + "\n")
    return items, texts
