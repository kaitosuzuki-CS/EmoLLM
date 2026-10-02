"""Print a markdown table of every run that has a stats.json (written by the training
scripts), for pasting into RESULTS.md.

    python scripts/emotion/summarize_runs.py                 # all runs under runs/
    python scripts/emotion/summarize_runs.py runs/finetune_* # selected runs
"""
import json
import sys
from pathlib import Path


def fmt(v, digits=3):
    return "–" if v is None else f"{v:.{digits}f}"


def row(run_dir: Path, s: dict) -> str:
    b, r = s["best"], s["run"]
    objective = s["objective"]
    if objective == "classification":
        scores = f"head {fmt(b.get('weighted_f1'))} / {fmt(b.get('macro_f1'))}"
    else:
        scores = f"kNN {fmt(b.get('knn_weighted_f1'))} / {fmt(b.get('knn_macro_f1'))}"
        if "probe_weighted_f1" in b:
            scores += f"; probe {fmt(b['probe_weighted_f1'])} / {fmt(b['probe_macro_f1'])}"
    pipeline = ("fine-tuned" if s["task"] == "finetune" else "frozen") + f" {objective}"
    p, f = s.get("params", {}), s.get("forward_flops", {})
    params = (f"{p['total'] / 1e6:.1f}M / {p['trainable'] / 1e6:.2f}M" if p else "–")
    gflops = (f"{f['per_second']['total'] / 1e9:.1f}" if f else "–")
    train_flops = s.get("train_flops_est")
    return (f"| `{run_dir.name}` | {pipeline} | {s['pooling']} | {s['data']['val_split']} | "
            f"{b.get('epoch')} / {r['epochs']} | {scores} | {r['train_min']:.1f} | {r['eval_min']:.1f} | "
            f"{r['train_clips_per_s']:.0f} | {r.get('peak_gpu_mem_gb', 0):.1f} | {params} | {gflops} | "
            f"{'–' if train_flops is None else f'{train_flops / 1e12:.1f}'} |")


def main():
    dirs = [Path(a) for a in sys.argv[1:]] or sorted(Path("runs").iterdir())
    print("| run | pipeline | pooling | val | best / total epochs | weighted F1 / macro F1 "
          "| train min | eval min | train clips/s | peak GPU GB | params total / trainable "
          "| fwd GFLOPs per s audio | train TFLOPs (est.) |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for d in dirs:
        stats = d / "stats.json"
        if stats.exists():
            print(row(d, json.loads(stats.read_text())))


if __name__ == "__main__":
    main()
