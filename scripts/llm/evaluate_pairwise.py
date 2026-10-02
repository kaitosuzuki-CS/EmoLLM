"""Pairwise evaluation of two sets of replies with a held-out judge, plus an emotion-
alignment test.

The judge (config `pairwise_judge`, by default Mistral-Small-3.2, a different family from
the Qwen judge used for pseudo labels and GRPO rewards) sees the dialogue context, the
transcript, an emotion label presented as the speaker's annotated emotion, and both
replies. Each pair is judged twice with the order swapped: a win needs both orders to
agree, otherwise the pair is a tie, which cancels position bias.
Win rate of A = (wins + 0.5 * ties) / n, with a 95% bootstrap interval.

Emotion alignment: the same comparison is repeated with the true emotion label replaced
by a counterfactual one (a random different emotion, or a valence flip). A reply set
that really adapts to the speaker's emotion should win clearly under the true label and
lose that edge under the counterfactual label:
    alignment gap = win_rate(true label) - win_rate(counterfactual label)
A gap near 0 means A's advantage (if any) does not come from matching the emotion.

Systems A and B are policy runs, text-only baselines or reply files (see emollm/systems.py);
replies are generated greedily, or read from each system's reply cache:
    python scripts/llm/evaluate_pairwise.py runs/llm_grpo_<stamp> baseline:none --split test
    python scripts/llm/evaluate_pairwise.py runs/llm_grpo_<A> runs/llm_grpo_<B> --split test
    python scripts/llm/evaluate_pairwise.py runs/llm_grpo_<stamp> runs/llm_grpo_<stamp>:shuffled \
        --split dev --counterfactual valence --limit 200
Needs the pairwise judge server:
    vllm serve mistralai/Mistral-Small-3.2-24B-Instruct-2506 --port 8002 --max-model-len 4096 \\
        --tokenizer-mode mistral --config-format mistral --load-format mistral
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from emollm.judge import counterfactual
from emollm.pairwise import PairwiseJudge, win_rate
from emollm.systems import check_features, parse_system, replies, system_config


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("a", help="system A: policy run dir [:shuffled], baseline:{none,predicted,oracle} "
                              "or a reply .jsonl")
    ap.add_argument("b", help="system B (the opponent), same forms")
    ap.add_argument("--split", default="dev", help="data split (default: dev)")
    ap.add_argument("--limit", type=int, default=0, help="first N utterances only (default: all)")
    ap.add_argument("--labels", default="true,counterfactual", type=lambda v: v.split(","),
                    help="emotion labels shown to the judge: true, counterfactual, predicted")
    ap.add_argument("--counterfactual", default="random", choices=["random", "valence"])
    ap.add_argument("--config", default="configs/llm.yaml")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--batch-size", type=int, default=16, help="generation batch size")
    ap.add_argument("--out", help="output JSON (default: runs/llm_pairwise/<split>/<A>_vs_<B>.json)")
    ap.add_argument("overrides", nargs="*", help="config overrides, e.g. data_root=/data/meld")
    args = ap.parse_intermixed_args()

    split = args.split
    sys_a, sys_b = parse_system(args.a), parse_system(args.b)
    name_a, name_b = sys_a.name, sys_b.name
    if name_a == name_b:
        name_b += "_b"
    cfg_a = system_config(sys_a, args.config, args.overrides)
    cfg_b = system_config(sys_b, args.config, args.overrides)
    # a baseline without its own emotion.cache_dir reads transcripts (and, for baseline:predicted,
    # emotion predictions) from the opposing policy's feature cache
    if not any(o.startswith("emotion.cache_dir=") for o in args.overrides):
        for s, c, other, oc in ((sys_a, cfg_a, sys_b, cfg_b), (sys_b, cfg_b, sys_a, cfg_a)):
            if s.kind != "policy" and other.kind == "policy":
                c["emotion"]["cache_dir"] = oc["emotion"]["cache_dir"]
    check_features(cfg_a, name_a)
    check_features(cfg_b, name_b)
    items_a, texts_a = replies(sys_a, cfg_a, split, args.limit, args.batch_size)
    items_b, texts_b = replies(sys_b, cfg_b, split, args.limit, args.batch_size)
    a = {it["id"]: t for it, t in zip(items_a, texts_a)}
    b = {it["id"]: t for it, t in zip(items_b, texts_b)}
    items = [it for it in items_a if it["id"] in b]   # transcript, context and labels are shared
    if not items:
        raise ValueError(f"{name_a} and {name_b} share no utterances on {split}")
    cfg = cfg_a
    judge = PairwiseJudge(cfg)
    print(f"{name_a} vs {name_b}: {len(items)} shared utterances | judge {judge.model} | "
          f"labels {args.labels} (counterfactual: {args.counterfactual})")

    records, summary = [], {"a": name_a, "b": name_b, "a_system": args.a, "b_system": args.b,
                            "judge": judge.model, "split": split,
                            "counterfactual_scheme": args.counterfactual}
    for label in args.labels:
        shown = []
        for it in items:
            if label == "true":
                e = it["emotion"]
            elif label == "predicted":
                e = it["pred_emotion"]
            else:
                e = counterfactual(it["id"], it["emotion"], args.counterfactual, args.seed)
            shown.append({**it, "emotion": e})
        results = judge.map(lambda x: judge.compare(x, a[x["id"]], b[x["id"]]), shown)
        per_emotion = defaultdict(list)
        for it, s, r in zip(items, shown, results):
            records.append({"id": it["id"], "label_condition": label, "true_emotion": it["emotion"],
                            "shown_emotion": s["emotion"], **r})
            if r["outcome"] is not None:
                per_emotion[it["emotion"]].append(r["outcome"])
        outcomes = [r["outcome"] for r in results if r["outcome"] is not None]
        summary[label] = {**win_rate(outcomes, seed=args.seed),
                          "judge_failures": sum(r["outcome"] is None for r in results),
                          "order_consistency": float(np.mean([r["consistent"] for r in results])),
                          "per_true_emotion": {e: win_rate(v, seed=args.seed).get("win_rate")
                                               for e, v in per_emotion.items()}}
        w = summary[label]
        print(f"  label={label:<14} win rate A {w['win_rate']:.3f} [{w['ci95'][0]:.3f}, {w['ci95'][1]:.3f}] "
              f"(W/L/T {w['wins']}/{w['losses']}/{w['ties']}, order-consistent {w['order_consistency']:.2f}, "
              f"failures {w['judge_failures']})")

    if "true" in summary and "counterfactual" in summary:
        # paired bootstrap over utterances for the gap
        by = defaultdict(dict)
        for r in records:
            if r["outcome"] is not None:
                by[r["id"]][r["label_condition"]] = {"a": 1.0, "b": 0.0, "tie": 0.5}[r["outcome"]]
        pairs = np.array([(v["true"], v["counterfactual"]) for v in by.values()
                          if "true" in v and "counterfactual" in v])
        diff = pairs[:, 0] - pairs[:, 1]
        boot = np.random.default_rng(args.seed).choice(diff, (2000, len(diff))).mean(1)
        summary["alignment_gap"] = {"gap": float(diff.mean()), "ci95": [float(np.percentile(boot, 2.5)),
                                                                        float(np.percentile(boot, 97.5))],
                                    "n": int(len(diff))}
        g = summary["alignment_gap"]
        print(f"  alignment gap (true - counterfactual) {g['gap']:+.3f} [{g['ci95'][0]:+.3f}, {g['ci95'][1]:+.3f}]")

    out = Path(args.out or f"runs/llm_pairwise/{split}/{name_a}_vs_{name_b}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2))
    with open(out.with_suffix(".jsonl"), "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    print(f"wrote {out} and {out.with_suffix('.jsonl').name}")


if __name__ == "__main__":
    main()
