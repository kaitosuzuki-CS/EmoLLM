# EmoLLM: A Conversational Assistant That Hears How You Feel

The same words can mean very different things depending on how they are said. "Great, just
great" can be delight or exasperation. A text-only assistant hears the words but misses the
voice, so it often answers the literal sentence instead of the person.

**The goal of this project** is a small assistant that runs locally and replies in a way that
fits both what the speaker said and how they sounded. It listens to a spoken turn, recognizes
the speaker's vocal emotion, and writes an empathetic reply conditioned on that emotion. The
complete inference path stays under 6B parameters (4.06B including speech recognition).

It is built and evaluated on **MELD** (emotion-labelled dialogue from the TV series *Friends*)
in three parts:

1. **Emotion encoder.** emotion2vec+ base turns each utterance's audio into an embedding.
   Mean or attention pooling and a projector (Linear → GELU → Linear) map it to a D-dim
   emotion latent. The latent is trained with a classification head, supervised contrastive
   learning (+ VICReg variance/covariance terms), or end-to-end fine-tuning of the encoder.
2. **Emotion-conditioned LLM.** The frozen emotion latent is projected to *K* soft-prompt
   tokens for **Qwen2.5-3B-Instruct** (LoRA), next to the dialogue context and transcript.
   The LLM writes an empathetic reply. It is trained with SFT on pseudo labels from a larger
   judge LLM (Qwen2.5-32B), then **GRPO** with judge rewards (optionally with a contrastive judge
   reward and an emotion-dependence bonus). The trained model is 3.2B parameters; with Whisper
   speech recognition for live use, the inference path is 4.06B.
3. **Evaluation.** Emotion recognition on MELD dev and test. Held-out pairwise judging by a
   different-family LLM (Mistral-Small-3.2), with an **emotion-alignment** test: does a reply
   still win when the judge is told the wrong emotion?

| Document | Contents |
|---|---|
| **[docs/walkthrough.md](docs/walkthrough.md)** | One spoken utterance traced through every stage to both outputs (emotion + reply) |
| **[RESULTS.md](RESULTS.md)** | All numbers, latent-space figures and example replies |
| **[docs/design.md](docs/design.md)** | Design decisions with evidence and trade-offs; external and generated components |
| **[docs/usage.md](docs/usage.md)** | Everything you can run and configure |
| **[docs/chat.md](docs/chat.md)** | Talking to the assistant with your voice |
| **[docs/reproduce.md](docs/reproduce.md)** | Regenerating every result |
| **[docs/llm.md](docs/llm.md)** | The LLM's architecture and training |

## Headline results

Model choices were made on MELD dev; MELD test was used only for final evaluation.

| Emotion model | Dev weighted F1 | **Test weighted F1** | Test accuracy |
|---|---|---|---|
| kNN on untrained emotion2vec+ embeddings | 0.427 | **0.456** | **0.476** |
| Fine-tuned emotion2vec+, attention pooling, classification (best on dev) | **0.444** | 0.449 | 0.469 |
| All 17 trained configurations (RESULTS.md) | 0.427–0.444 | 0.442–0.458 | – |

On test, no training recipe improves on the frozen emotion2vec+ features.

Fine-tuning separates the emotion classes well on train but barely on dev or test. emotion2vec+
appears to have been pretrained on MELD train. See the latent-space analysis in RESULTS.md.

| Emotion-conditioned LLM (Qwen2.5-3B + 4 soft emotion tokens, SFT → GRPO) | vs no-emotion baseline, test (dev) | Alignment gap (valence), test (dev) |
|---|---|---|
| conditioned on the classification encoder | 0.727 (0.749) win rate | +0.047 (+0.053) |
| conditioned on the SupCon encoder | 0.734 (0.756) win rate | +0.043 (+0.080) |

Win rates come from held-out pairwise judging (Mistral-Small-3.2, both orders) on all 2,610 test
and 1,106 dev utterances. The alignment gap is how much the win rate drops when the judge is
told the opposite emotion. Details and example replies are in
[RESULTS.md](RESULTS.md#emotion-conditioned-llm).

## Inference path and resources

The system is **conversational and turn-based**: it answers each spoken turn after the user
stops speaking. Everything on this path runs locally; the judge LLMs are used only for training
and evaluation.

| Component on the inference path | Learned parameters |
|---|---|
| emotion2vec+ encoder (fine-tuned) + attention pooling + projector + probe | 94.4M |
| Whisper large-v3-turbo (speech recognition) | 808.9M |
| Soft-prompt projector | 8.7M |
| Qwen2.5-3B-Instruct | 3,085.9M |
| LoRA adapters, SFT + GRPO (merged at load; counted unmerged) | 59.8M |
| **Total** | **4.058B** (limit: 6B total parameters) |

The counts come from the loaded models (`scripts/llm/chat.py --trace`).
`chat.py --baseline` loads a second copy of Qwen2.5-3B, for side-by-side comparison only; that
mode is not part of the system and totals 7.14B.

Observed requirements, measured on an NVIDIA A100-SXM4-80GB with an AMD EPYC 7742 host:

| | GPU (A100) | CPU only (EPYC 7742, shared node) |
|---|---|---|
| Time from end of speech to full reply | median **1.26 s**, p90 1.58 s (15 turns; first turn 2.4 s) | 47–70 s |
| … speech recognition / emotion model / reply | 114 / 16 / 1,133 ms (median) | 14–17 s / 0.3–0.6 s / 32–53 s |
| Reply generation speed | 24–31 tokens/s (greedy, batch 1) | ~1 token/s |
| Peak GPU memory | 8.6 GB | – |
| Peak host RAM | 4.9 GB | 12.1 GB |
| Startup (models already downloaded) | ~20–25 s | – |

So conversational use needs a GPU with about 9 GB free.

**Training and evaluation** (all on A100-80GB GPUs):

| Stage | Time | GPU memory |
|---|---|---|
| Emotion models | 0.2–11 min per run; ~2 h for all 17 experiments on one GPU | < 0.1–46 GB (fine-tuned SupCon is the largest) |
| Pseudo labels | ~20 min | Qwen2.5-32B judge on the judge GPU; Qwen2.5-3B candidate generator on the policy GPU (both served with vLLM) |
| SFT / GRPO, per policy | ~50 min / ~2 h | 13.4 GB / 9.7 GB, plus the judge GPU |
| Evaluation on test (2 policies) | ~3.5 h | Policy GPU + judge GPU (Mistral-Small-3.2 or Qwen2.5-32B) |

The LLM pipeline needs two 80 GB GPUs: one for the policy, one for the judge.

**Disk:**
- MELD raw: 21 GB.
- Generated audio and feature caches: 4.7 GB.
- Downloaded models: ~120 GB, mostly the judges; the inference path alone needs ~8 GB.
- One trained system: 361 MB emotion checkpoint, 2 × 115 MB LoRA adapters and a 34 MB
  projector.

## Installation

Two Python environments: one for training and evaluation, one for serving the judge LLMs with
vLLM (which pins its own PyTorch). They communicate over HTTP.

```bash
# training / evaluation (Python 3.10, CUDA GPU)
conda create -n emotion -c conda-forge python=3.10 ffmpeg
conda activate emotion
pip install -r requirements.txt
pip install -e .                # installs the `emotion` and `emollm` packages

# judge serving (only needed for the LLM pipeline)
conda create -n vllm python=3.12
conda activate vllm
pip install -r requirements-vllm.txt
```

Model weights download automatically on first use: emotion2vec+ from ModelScope, the LLMs from
Hugging Face. The LLM pipeline needs about 120 GB of disk for Qwen2.5-3B/32B and
Mistral-Small-3.2.

## Data

Download MELD (the [raw release](https://affective-meld.github.io/), `MELD.Raw.tar.gz`) and
extract it, including the inner `train.tar.gz`, `dev.tar.gz` and `test.tar.gz`, so that:

```
data/meld/MELD.Raw/train_splits/dia<D>_utt<U>.mp4                  data/meld/MELD.Raw/train_sent_emo.csv
data/meld/MELD.Raw/dev_splits_complete/dia<D>_utt<U>.mp4           data/meld/MELD.Raw/dev_sent_emo.csv
data/meld/MELD.Raw/output_repeated_splits_test/dia<D>_utt<U>.mp4   data/meld/MELD.Raw/test_sent_emo.csv
```

Every path in the configs is relative to `data_root` (default `data/meld`). Pass
`data_root=/path/to/meld` to any script to use another location.

## Usage

Everything below is documented, with all options, in the **[usage guide](docs/usage.md)**.

```bash
# 1. data: 16 kHz audio + cached emotion2vec+ features for each split (~15 min)
python scripts/data/prepare_audio.py      --config configs/emotion/base.yaml --splits train,dev,test
python scripts/data/extract_embeddings.py --config configs/emotion/base.yaml --splits train,dev,test --frames

# 2. train an emotion model (four pipelines; any config key can be overridden as key=value)
python scripts/emotion/train_classifier.py --config configs/emotion/classification.yaml   # frozen encoder + classifier
python scripts/emotion/train_supcon.py     --config configs/emotion/supcon.yaml           # frozen encoder + SupCon
python scripts/emotion/train_finetune.py   --config configs/emotion/finetune.yaml pooling.type=attention   # fine-tuned
python scripts/emotion/train_finetune.py   --config configs/emotion/finetune_supcon.yaml  # fine-tuned + SupCon

# 3. evaluate emotion models on a split; several checkpoints are also compared (paired bootstrap)
python scripts/emotion/evaluate.py runs/finetune_<stamp> --split test
python scripts/emotion/evaluate.py runs/finetune_<A> runs/supcon_<B> --split test --fit-probe
python scripts/emotion/knn_baseline.py --split test                      # untrained-embedding baseline
python scripts/emotion/plot_latent_space.py runs/finetune_<stamp>        # PCA / LDA / t-SNE + separation metrics

# 4. emotion-conditioned LLM (judge LLMs served with vLLM; see the guide)
C="emotion.cache_dir=data/meld/llm/cls_attention"
python scripts/llm/extract_features.py --splits train,dev,test emotion.checkpoint=runs/finetune_<stamp>/best.pt $C
python scripts/llm/make_pseudo_labels.py $C
python scripts/llm/train_sft.py $C
python scripts/llm/train_grpo.py $C grpo.init_from=runs/llm_sft_<stamp>

# 5. evaluate LLMs on a split: judge scores for one system, or pairwise + emotion alignment for two
python scripts/llm/evaluate.py runs/llm_grpo_<stamp> --split test
python scripts/llm/evaluate_pairwise.py runs/llm_grpo_<stamp> baseline:none --split test
python scripts/llm/evaluate_pairwise.py runs/llm_grpo_<stamp> runs/llm_grpo_<stamp>:shuffled --split test
```

## Talk to the assistant

Record your voice in the terminal and get a reply from a trained policy. Whisper transcribes what
you said, and the policy's emotion model hears how you said it. Full guide:
**[docs/chat.md](docs/chat.md)**.

```bash
pip install -e ".[demo]" && sudo apt install libportaudio2   # microphone support (optional)
python scripts/llm/chat.py runs/llm_grpo_<stamp>                # Enter to record, Enter to stop
python scripts/llm/chat.py runs/llm_grpo_<stamp> --baseline     # also show the no-emotion LLM's reply
python scripts/llm/chat.py runs/llm_grpo_<stamp> --wav my_clip.wav   # audio files instead of a microphone
```

## Reproducing the results

To regenerate everything in RESULTS.md, use the two drivers in **[docs/reproduce.md](docs/reproduce.md)**:

```bash
scripts/reproduce_emotion.sh                                                   # 1 GPU, ~2 h
POLICY_GPU=0 JUDGE_GPU=1 scripts/reproduce_llm.sh cls_attention=<ckpt> supcon_attention=<ckpt>   # 2 GPUs, ~6 h
```

Training logs go to Weights & Biases; add `wandb.mode=disabled` to turn that off.

## Scope

**Completed**
- **Data pipeline** for MELD train, dev and test: audio extraction and cached emotion2vec+
  features.
- **Four emotion-model training pipelines** (frozen or fine-tuned encoder; classification or
  supervised contrastive with VICReg terms), mean and attention pooling. 17 experiments,
  evaluated on dev (selection) and test (evaluation only), with kNN / probe / head metrics and a
  latent-space analysis of every checkpoint.
- **An emotion-conditioned LLM** under a 6B parameter budget: soft emotion tokens for
  Qwen2.5-3B-Instruct, trained with SFT on judge-refined pseudo labels and then GRPO. Two reward
  designs were compared.
- **Evaluation:** training-judge scores, held-out pairwise judging, an emotion-alignment test with
  counterfactual labels, and shuffled-feature controls, on dev and test.
- **A local, turn-based voice demo** with per-stage tracing; reproduction drivers; unit tests.

**Intentionally left out**
- **Spoken output.** Replies are text; there is no text-to-speech.
- **Streaming.** Each turn is processed after the speaker stops; there are no partial
  transcripts, early replies or interruptions.
- **The video modality of MELD** (faces, gestures). Only audio and text are used.
- **Human evaluation.** All reply quality is judged by LLMs.
- **Multiple seeds** and a hyperparameter search for the LLM stages.
- **Test evaluation of the contrastive + PMI reward variant.** It was rejected on dev.
- **Training on ASR transcripts, multi-speaker live input** (diarization), other languages, and
  deployment work (quantization, a serving API).

## Limitations

- **Emotion recognition is weak.** 0.45–0.46 weighted F1 on MELD test, no better than kNN on the
  untrained emotion2vec+ embeddings. Disgust and fear are near 0. emotion2vec+ likely saw MELD
  train during pretraining.
- **The LLM's emotion use is modest.** Most of its 73% win rate over the no-emotion baseline is
  general reply quality: the alignment gap is +0.04–0.05.
- **Misheard emotions are passed on confidently.** The policy also tends to name the feeling
  outright and to reuse stock empathetic phrasing, a side effect of optimizing an LLM judge
  ([examples](RESULTS.md#where-it-goes-wrong)).
- **Only LLM judges.** They are biased toward longer replies (12–14 points), and the policies
  write longer replies than the baseline.
- **Narrow domain:** acted English sitcom speech, single seed. The policy was trained on gold
  transcripts but hears ASR output live.
- **Turn-based latency** (~1.3 s on an A100) is conversational but not streaming. CPU-only
  inference is far too slow for conversation.

## Next steps

- **Streaming with action chunking.** Process audio incrementally and generate the reply in
  short chunks that are emitted (and can be revised or interrupted) while the user is still
  speaking or as soon as they stop. This would make the assistant feel responsive, not just
  turn-based.
- **End-to-end audio LLMs.** Replace the emotion encoder + transcript pipeline with a large
  multimodal model that reads the speech audio directly and generates the reply. Prosody,
  emotion and words would be modeled jointly instead of being compressed into 4 soft tokens and
  a transcript.
- **Better emotion signal and evaluation.** Use a stronger or multimodal (audio + video) emotion
  model, train the policy on ASR transcripts, add human preference evaluation, and penalize
  explicit emotion labeling and length in the GRPO reward.

## Repository layout

```
configs/
  emotion/          base.yaml + classification / supcon / finetune / finetune_supcon pipelines
  llm.yaml          emotion-conditioned LLM: policy, prompts, judges, pseudo labels, SFT, GRPO
emotion/            emotion-model package
  config.py           YAML configs: inheritance, CLI overrides, ${key} interpolation
  data.py             MELD manifests, cached features, samplers, dataloaders
  encoder.py          emotion2vec+ wrapper (exact padded batching; frozen or trainable)
  model.py            pooling (mean / attention), projector, heads, fine-tuning model
  losses.py           SupCon + VICReg variance / covariance terms
  evaluation.py       classification metrics, kNN, linear probe
  checkpoint.py       load any trained checkpoint
  trainer.py          optimizer / schedule, logging (stdout, JSONL, W&B), run stats
  profiling.py        parameter counts, FLOPs, timings
emollm/             emotion-conditioned LLM package
  data.py             transcripts, dialogue context, cached emotion features, pseudo labels
  prompt.py           chat prompts with the soft-prompt slot; batching
  model.py            soft-prompt projector, LoRA policy, saving / loading
  judge.py            judge client: pseudo labels and 1-10 reward scores; counterfactual labels
  grpo.py             reward composition (judge, contrastive, PMI) and group advantages
  pairwise.py         held-out pairwise judging and win rates
  emotion_features.py audio -> emotion features with the frozen emotion model
  systems.py          evaluated systems (policy runs, baselines, reply files) and cached replies
  utils.py            sampling / generation helpers, garbled-output check
scripts/
  data/             prepare_audio.py, extract_embeddings.py
  emotion/          train_classifier.py, train_supcon.py, train_finetune.py, evaluate.py,
                    knn_baseline.py, plot_latent_space.py, summarize_runs.py
  llm/              extract_features.py, make_pseudo_labels.py, train_sft.py, train_grpo.py,
                    evaluate.py, evaluate_pairwise.py, chat.py
  reproduce_emotion.sh, reproduce_llm.sh     end-to-end reproduction
tests/              CPU unit tests (pytest)
docs/               walkthrough.md, design.md, usage.md, chat.md, reproduce.md, llm.md, figures/
```

## Development

```bash
pip install -e ".[dev]"
pytest          # CPU-only unit tests, a few seconds
ruff check .
```

## Acknowledgements

[MELD](https://affective-meld.github.io/) ·
[emotion2vec](https://github.com/ddlBoJack/emotion2vec) ·
[FunASR](https://github.com/modelscope/FunASR) ·
[Qwen2.5](https://huggingface.co/Qwen) ·
[Mistral-Small-3.2](https://huggingface.co/mistralai/Mistral-Small-3.2-24B-Instruct-2506) ·
[vLLM](https://github.com/vllm-project/vllm)
