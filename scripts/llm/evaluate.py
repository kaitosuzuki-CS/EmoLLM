"""Score one system's replies on a data split with the training judge (absolute 1-10 scores).

A system is a policy run, a text-only baseline or a reply file (see emollm/systems.py):
    python scripts/llm/evaluate.py runs/llm_grpo_<stamp> --split test
    python scripts/llm/evaluate.py runs/llm_grpo_<stamp>:shuffled --split test   # control: other
                                                                    #   utterances' emotion features
    python scripts/llm/evaluate.py baseline:none --split test        # base LLM, no emotion information
    python scripts/llm/evaluate.py baseline:predicted --split test emotion.cache_dir=<features>
    python scripts/llm/evaluate.py baseline:oracle --split dev --limit 200

Replies are generated greedily (or read from the system's reply cache) and scored by the judge
(config `judge`, Qwen2.5-32B by default; needs its server). Writes eval_<split>_<name>.jsonl
(reply, score, reason per utterance) and eval_<split>_<name>.json (mean score overall and per
ground-truth emotion) to the run directory, or runs/llm_baselines for baselines. With
--no-judge, only generates and caches the replies.
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path

from emollm.judge import Judge
from emollm.systems import check_features, parse_system, replies, system_config
from emotion.data import EMOTIONS


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("system", help="policy run dir [:shuffled], baseline:{none,predicted,oracle} "
                                   "or a reply .jsonl")
    ap.add_argument("--split", default="dev", help="data split (default: dev)")
    ap.add_argument("--limit", type=int, default=0, help="first N utterances only (default: all)")
    ap.add_argument("--no-judge", action="store_true", help="only generate and cache replies")
    ap.add_argument("--regenerate", action="store_true", help="ignore cached replies")
    ap.add_argument("--config", default="configs/llm.yaml")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--out", help="output directory (default: the run dir, or runs/llm_baselines)")
    ap.add_argument("overrides", nargs="*", help="config overrides, e.g. data_root=/data/meld")
    args = ap.parse_intermixed_args()

    system = parse_system(args.system)
    cfg = system_config(system, args.config, args.overrides)
    check_features(cfg, system.name)
    print(f"{system.name} on {args.split} | emotion features {cfg['emotion']['cache_dir']} "
          f"({cfg['emotion']['feature']})")
    items, texts = replies(system, cfg, args.split, args.limit, args.batch_size, args.regenerate)
    if args.no_judge:
        return

    judged = Judge(cfg).score_many(items, texts)
    out_dir = Path(args.out or (system.path if system.kind == "policy" else "runs/llm_baselines"))
    out_dir.mkdir(parents=True, exist_ok=True)
    name = f"{args.split}_{system.name}"
    per_emotion = defaultdict(list)
    with open(out_dir / f"eval_{name}.jsonl", "w") as f:
        for it, t, j in zip(items, texts, judged):
            if j["score"] is not None:
                per_emotion[it["emotion"]].append(j["score"])
            f.write(json.dumps({"id": it["id"], "emotion": it["emotion"],
                                "pred_emotion": it["pred_emotion"], "text": it["text"],
                                "reply": t, "score": j["score"], "reason": j["reason"]}) + "\n")
    scores = [s for v in per_emotion.values() for s in v]
    summary = {"system": args.system, "name": name, "split": args.split, "n": len(items),
               "judge": cfg["judge"]["model"], "judge_failures": len(items) - len(scores),
               "judge_score": sum(scores) / max(1, len(scores)),
               "reply_words": sum(len(t.split()) for t in texts) / max(1, len(texts)),
               "per_emotion": {e: round(sum(v) / len(v), 3) for e in EMOTIONS
                               if (v := per_emotion.get(e))}}
    (out_dir / f"eval_{name}.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
