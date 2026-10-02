# Reproducing the results

How to regenerate every number and figure in [RESULTS.md](../RESULTS.md). For what each script
does and every option you can change, see the [usage guide](usage.md). Installation and the MELD
download are in the [README](../README.md#installation). All commands run from the repository
root with the `emotion` environment active.

## Overview

```
MELD mp4 clips ─► scripts/data/prepare_audio.py ─► 16 kHz wavs + manifests (train, dev, test)
               ─► scripts/data/extract_embeddings.py --frames ─► cached emotion2vec+ z / frame features
               ─► scripts/emotion/train_classifier.py | train_supcon.py   (frozen encoder, cached features)
                  scripts/emotion/train_finetune.py                        (encoder fine-tuned on raw audio)
               ─► scripts/emotion/evaluate.py, knn_baseline.py, plot_latent_space.py, summarize_runs.py

emotion checkpoint ─► scripts/llm/extract_features.py ─► per-utterance emotion latent + transcripts
  ─► scripts/llm/make_pseudo_labels.py ─► scripts/llm/train_sft.py ─► scripts/llm/train_grpo.py
  ─► scripts/llm/evaluate.py, scripts/llm/evaluate_pairwise.py
```

Two drivers run everything end to end:

| Driver | Covers | Time |
|---|---|---|
| `scripts/reproduce_emotion.sh` | data preparation (train, dev, test), the 17 emotion-model experiments, test evaluation of every checkpoint, the kNN baseline on dev and test, latent plots | ~2 h on one A100 |
| `scripts/reproduce_llm.sh` | features, pseudo labels, SFT, GRPO, training-judge evaluation, held-out pairwise evaluation | ~6 h on two A100s (+ ~3 h per extra split) |

Dev was used for every choice: best epochs, configurations, LLM hyperparameters. Test was only
evaluated at the end.

## Emotion models

```bash
scripts/reproduce_emotion.sh            # optional experiment-name prefix filter, e.g. sweep_
```

| Variable | Default | Meaning |
|---|---|---|
| `DATA_ROOT` | data/meld | Data location |
| `WANDB_MODE` | offline | Weights & Biases mode |

The driver runs these steps; the RESULTS.md section each one feeds is in brackets.
1. **Data.** Prepares wavs and caches for train, dev and test (skipped when the caches exist).
2. **Training** [Emotion models, Ablations]. Trains each experiment with the command recorded in
   `runs/rerun_index.tsv` (name → run directory → command).
3. **Test evaluation** [test columns]. Runs
   `scripts/emotion/evaluate.py <run> --split test --fit-probe` on each checkpoint.
   `--fit-probe` refits the linear probe on train for runs that did not save one.
4. **Baseline** [untrained kNN row]. Runs `scripts/emotion/knn_baseline.py --split dev` and
   `--split test`.
5. **Latent plots** [Latent space]. Runs `scripts/emotion/plot_latent_space.py <run>` (train,
   dev and test rows) into `runs/latent/<name>`.
   - The figures in `docs/figures/latent/` are these outputs.
   - The metric tables are read from each figure directory's `metrics.json`.

Individual experiments, e.g. the best dev model and its test score:

```bash
python scripts/emotion/train_finetune.py --config configs/emotion/finetune.yaml pooling.type=attention
python scripts/emotion/evaluate.py runs/finetune_<stamp> --split test
```

Approximate training time on one A100-80GB:

| Pipeline | Mean pooling | Attention pooling |
|---|---|---|
| Frozen classification | 0.2 min | 3.5 min |
| Frozen SupCon | 0.5 min | 7.5 min |
| Fine-tuned classification (peak 20 GB) | 5 min | 5 min |
| Fine-tuned SupCon (peak 46 GB) | 11 min | 11 min |

## Emotion-conditioned LLM

Two GPUs: one for the policies, one for the judge LLMs (served by vLLM from the `vllm`
environment).

```bash
POLICY_GPU=0 JUDGE_GPU=1 scripts/reproduce_llm.sh \
    cls_attention=runs/finetune_<stamp>/best.pt \
    supcon_attention=runs/finetune_<stamp>/best.pt
```

Each `name=checkpoint` is a frozen emotion model. The RESULTS.md runs use the fine-tuned
classification and SupCon models with attention pooling. One policy is trained per emotion
model, side by side. The driver:
1. extracts features for train, dev and `SPLIT`;
2. serves Qwen2.5-32B (training judge) and Qwen2.5-3B (candidates) and generates pseudo labels;
3. runs SFT, then GRPO;
4. scores the SFT and GRPO policies and the no-emotion baseline with the training judge;
5. switches to Mistral-Small-3.2 for the held-out comparisons, each with true and counterfactual
   labels:
   - GRPO vs the no-emotion baseline (random and valence counterfactuals),
   - GRPO vs itself with shuffled emotion features,
   - GRPO vs SFT,
   - the first emotion model's policy vs the second's.

| Variable | Default | Meaning |
|---|---|---|
| `POLICY_GPU`, `JUDGE_GPU` | required | GPU for the policies / for the judge servers |
| `STAGES` | `features labels sft grpo eval pairwise` | Run a subset; later stages read earlier runs from `runs/llm_index.tsv` |
| `SPLIT` | dev | Split for the `eval` and `pairwise` stages |
| `EVAL_LIMIT` | 200 | Utterances scored by the training judge in the `eval` stage (0 = all) |
| `GRPO_TAG` | grpo | Stage name of the GRPO runs; use a new tag to compare reward settings |
| `COMPARE_TAG` | – | Earlier GRPO tag to compare against in the pairwise stage |
| `EXTRA` | – | Config overrides for SFT and GRPO, e.g. `EXTRA="grpo.steps=200"` |
| `DATA_ROOT`, `VLLM`, `WANDB_MODE` | data/meld, vllm, offline | Data location, vLLM executable, W&B mode |

Results go to `runs/logs/`, `runs/llm_index.tsv` and `runs/llm_pairwise/<split>/`.

**Test-split results** [Emotion-conditioned LLM, test], from the trained policies:

```bash
STAGES="features eval pairwise" SPLIT=test EVAL_LIMIT=0 POLICY_GPU=0 JUDGE_GPU=1 \
    scripts/reproduce_llm.sh cls_attention=... supcon_attention=...
```

**The emotion-specific GRPO variant** [contrastive + PMI reward]. It trains from the existing SFT
runs and is compared with plain GRPO:

```bash
STAGES="grpo eval pairwise" GRPO_TAG=grpo_cpmi COMPARE_TAG=grpo \
EXTRA="grpo.reward.contrast_weight=0.5 grpo.reward.pmi_weight=0.1" \
POLICY_GPU=0 JUDGE_GPU=1 scripts/reproduce_llm.sh cls_attention=... supcon_attention=...
```

Step by step, with the judge servers from [usage §5](usage.md#judge-servers) running and
`C="emotion.cache_dir=data/meld/llm/cls_attention"`:

```bash
python scripts/llm/extract_features.py --splits train,dev,test emotion.checkpoint=runs/finetune_<stamp>/best.pt $C
python scripts/llm/make_pseudo_labels.py $C
python scripts/llm/train_sft.py $C
python scripts/llm/train_grpo.py $C grpo.init_from=runs/llm_sft_<stamp>

G=runs/llm_grpo_<stamp>
python scripts/llm/evaluate.py $G --split test                                  # training-judge scores
python scripts/llm/evaluate.py baseline:none --split test $C
python scripts/llm/evaluate_pairwise.py $G baseline:none --split test           # held-out judge (port 8002)
python scripts/llm/evaluate_pairwise.py $G baseline:none --split test --counterfactual valence
python scripts/llm/evaluate_pairwise.py $G $G:shuffled --split test
python scripts/llm/evaluate_pairwise.py $G runs/llm_sft_<stamp> --split test
```
