"""Cache SFT pseudo labels: the base policy model proposes candidate replies, then the
judge (seeing the ground-truth emotion, context and transcript) picks the best one and
refines it. Results are appended to <pseudo.out_dir>/<split>.jsonl; ids already there are
skipped, so the script resumes after interruption and labels are never regenerated.

Needs two OpenAI-compatible servers (e.g. `vllm serve`, see docs/reproduce.md): the judge
(judge.base_url) and the candidate model (pseudo.candidate_base_url).

    python scripts/llm/make_pseudo_labels.py --config configs/llm.yaml
    python scripts/llm/make_pseudo_labels.py --config configs/llm.yaml --splits dev pseudo.limit=32
"""
import argparse
import json
import time

from emollm.data import UtteranceDataset, pseudo_label_path, read_jsonl
from emollm.judge import Judge, OpenAIChat
from emollm.prompt import candidate_messages
from emotion.config import load_config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/llm.yaml")
    parser.add_argument("--splits", type=lambda v: v.split(","),
                        help="comma-separated names, e.g. train,dev (default: data.train_split and data.val_split)")
    parser.add_argument("--chunk", type=int, default=256, help="items per append to the cache")
    parser.add_argument("overrides", nargs="*")
    args = parser.parse_args()
    cfg = load_config(args.config, args.overrides)
    p = cfg["pseudo"]

    judge = Judge(cfg)
    cand = OpenAIChat(p["candidate_base_url"], p["candidate_model"], cfg["judge"]["max_concurrency"],
                      cfg["judge"]["timeout_s"])
    cand_kw = {"n": p["num_candidates"], "temperature": p["candidate_temperature"],
               "top_p": 0.95, "max_tokens": cfg["generation"]["max_new_tokens"]}

    def label(item):
        emotion = item["emotion"] if p["candidate_emotion_hint"] else "unknown"
        cands = [c.strip() for c in cand.chat(candidate_messages(item, cfg, emotion), **cand_kw)]
        out = judge.pseudo_label(item, cands)
        return {"id": item["id"], "emotion": item["emotion"], "pred_emotion": item["pred_emotion"],
                "candidates": cands, "best_candidate": out and out["best_candidate"],
                "response": out and out["reply"], "judge_model": judge.model,
                "candidate_model": cand.model}

    for split in args.splits or [cfg["data"]["train_split"], cfg["data"]["val_split"]]:
        out_path = pseudo_label_path(cfg, split)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        done = {r["id"] for r in read_jsonl(out_path) if r.get("response")}
        items = [it for it in UtteranceDataset(cfg, split, limit=p["limit"]).items
                 if it["id"] not in done]
        print(f"{split}: {len(done)} cached, {len(items)} to label -> {out_path}")
        failed, tic = 0, time.time()
        for start in range(0, len(items), args.chunk):
            chunk = items[start:start + args.chunk]
            results = judge.map(label, chunk)
            with open(out_path, "a") as f:
                for r in results:
                    failed += r["response"] is None
                    f.write(json.dumps(r) + "\n")
            n = start + len(chunk)
            print(f"  {n}/{len(items)} labelled ({failed} judge failures) "
                  f"{n / (time.time() - tic):.1f} items/s", flush=True)
        if items:
            ex = results[0]
            print(f"  example [{ex['emotion']}] candidates={ex['candidates'][:2]}... "
                  f"-> best {ex['best_candidate']}: {ex['response']!r}")


if __name__ == "__main__":
    main()
