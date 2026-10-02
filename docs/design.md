# Design decisions, components and provenance

Why the system is built the way it is, what each choice cost, and where every external or
generated piece comes from. Numbers link to [RESULTS.md](../RESULTS.md). The LLM's training
details are in [llm.md](llm.md).

## Decisions and trade-offs

| Decision | Why | Evidence | Trade-off |
|---|---|---|---|
| **emotion2vec+ base as the speech encoder** | A self-supervised speech-emotion model, small (94M) and strong across emotion benchmarks | kNN on its untrained embeddings already reaches 0.456 test weighted F1 | It appears to have been pretrained on MELD train, so train-set features are unrealistically separable, and fine-tuning gains on dev don't transfer (latent-space section) |
| **Train four encoder pipelines** (frozen / fine-tuned × classification / SupCon), mean or attention pooling | Find out what actually helps on MELD instead of assuming | 17 runs; all 0.442–0.458 on test | No recipe beats the untrained baseline on test. The trained encoders are kept because their dev selection was the honest procedure at the time |
| **Model selection on dev, test only for evaluation** | Avoid optimistic test numbers | Test was used only for final scoring | Dev margins (e.g. fine-tuning +0.017) turned out to be noise; the summary cites test |
| **A 256-d emotion latent, not an emotion word, feeds the LLM** | Keeps the model's uncertainty and intensity, and lets SFT/GRPO learn how to use them | Each policy beats its own shuffled-feature version (0.53–0.54 on test) | Harder to inspect than a label; a misheard emotion is still passed on confidently |
| **4 soft-prompt tokens + LoRA (r 16)** on a frozen base | Cheap conditioning: 38.6M trainable parameters, 13.4 GB peak for SFT, 9.7 GB for GRPO | Policies use the tokens (alignment gap +0.04–0.05 on test) | Less capacity than full fine-tuning; the projector learning rate must stay low (1e-4) or sampling degenerates (llm.md) |
| **Qwen2.5-3B-Instruct as the policy** | Leaves room in the 6B budget for the emotion model and a strong ASR model (4.06B in total). A 7B policy plus Whisper would be ~8.5B | Policy and baseline share the same base, so comparisons isolate the training | A 3B model writes plainer, more formulaic replies than larger models |
| **Whisper large-v3-turbo for live transcripts** | Accurate, fast (median 114 ms per turn on an A100), 809M parameters | – | The policy was trained on MELD's gold transcripts, so ASR errors are unseen at training time |
| **SFT on judge-refined pseudo labels** | MELD has no reference replies; a larger judge (Qwen2.5-32B) picks and refines the best of 4 candidates | SFT policy: 6.61–6.66 training-judge score on test vs 6.21 for the base model | The labels inherit the judge's taste; candidates saw the true emotion as a word, so labels are cleaner than what the policy hears at test time |
| **GRPO with an LLM-judge reward** | Scalable preference signal without human raters | GRPO beats SFT 0.59–0.61 with a held-out judge (test) | Reward hacking risk: replies get longer and label feelings explicitly (13% of replies) |
| **Plain judge reward, not the contrastive + PMI variant** | The emotion-specific terms made replies more emotion-dependent, but worse overall | Variant loses to plain GRPO 0.39–0.40 (dev) | Plain GRPO's emotion use is modest |
| **Held-out judge from a different family** (Mistral-Small-3.2) and both-order judging | The policy is never graded by the model it was optimized against; position bias cancels | Judge reproduces its own verdicts (79 / 80 on a re-run) | Still an LLM judge: it prefers longer replies by 12–14 points |
| **Alignment gap and shuffled-feature controls** | Separate "better replies" from "replies that fit the emotion" | Valence gap +0.04–0.05; random-label gap only +0.01–0.02 (test) | Shows most of the 73% win rate is general quality, not emotion use |
| **Turn-based interaction ("real-time" = conversational)** | The application answers each user turn after they stop speaking | Median 1.26 s from end of speech to full reply (GPU) | No streaming: nothing is shown while generating, and the user can't interrupt |
| **Separate vLLM environment for judges** | vLLM pins its own PyTorch; judges are only needed for training and evaluation | – | Two environments to install; judges need a second GPU |

## Components and provenance

### External, pretrained, and used as-is or fine-tuned here

| Component | Role | Used at inference? | Source / license |
|---|---|---|---|
| emotion2vec+ base (`iic/emotion2vec_plus_base`) | Speech-emotion encoder; fine-tuned in the `finetune` pipelines | Yes | [emotion2vec](https://github.com/ddlBoJack/emotion2vec), via FunASR / ModelScope |
| Qwen2.5-3B-Instruct | Policy LLM (frozen base + our LoRA) and the no-emotion baseline | Yes | [Qwen](https://huggingface.co/Qwen/Qwen2.5-3B-Instruct), Qwen Research License |
| Whisper large-v3-turbo | Speech recognition in the live demo | Yes (demo) | [OpenAI](https://huggingface.co/openai/whisper-large-v3-turbo), MIT |
| Qwen2.5-32B-Instruct | Training judge: pseudo labels, GRPO rewards, absolute scores | No | Apache 2.0 |
| Mistral-Small-3.2-24B-Instruct-2506 | Held-out judge for pairwise and alignment evaluation | No | Apache 2.0 |
| MELD | Audio, transcripts and emotion labels (train / dev / test) | No | [MELD](https://affective-meld.github.io/), GPL-3.0; not redistributed here |
| FunASR, Transformers, PEFT, vLLM, scikit-learn, librosa, ffmpeg | Model loading, LoRA, judge serving, probes / metrics, audio decoding | Partly | Open source |

The judges run only during training and evaluation, never on the inference path. All
inference runs locally; no remote API is called.

### Trained here

| Component | Parameters | Produced by |
|---|---|---|
| Attention pooler, projector, classification head / linear probe | 0.6M | `scripts/emotion/train_*.py` |
| Fine-tuned emotion2vec+ weights (`finetune` pipelines) | 89.0M trainable of 93.8M | `scripts/emotion/train_finetune.py` |
| Soft-prompt projector | 8.7M | SFT, then GRPO |
| LoRA adapters (SFT, GRPO) | 29.9M each | `scripts/llm/train_sft.py`, `train_grpo.py` |

### Generated data

- **Pseudo-label replies** for 9,976 train and 1,106 dev utterances. Qwen2.5-3B wrote the
  candidates and Qwen2.5-32B selected and refined them (0 judge failures).
  - They are the SFT targets, so the policy's style comes from these two models.
  - Spot checks found good labels overall, with occasional confusion about who is speaking in
    multi-party scenes.
- **Judge scores and verdicts** behind every LLM number: GRPO rewards, training-judge scores,
  pairwise outcomes. Raw per-utterance verdicts are saved next to each summary (`.jsonl`), so
  every reported number can be recomputed.
- **Emotion features and predictions** cached per emotion model (`scripts/llm/extract_features.py`).

### How the results were checked

- **Emotion pipeline:** reruns reproduced every dev number to three decimals. Refit test probes
  reproduce the saved dev probes to ±0.001.
- **Encoder batching:** padded batched encoding matches single-clip FunASR output exactly
  (pad-frame hooks).
- **Evaluation integrity:** a policy is always evaluated with its own emotion features. An early
  evaluation that violated this was caught, rerun and corrected.
- **Judge stability:** the held-out judge reproduces 79 of 80 pairwise verdicts on a re-run.
- **Unit tests:** cover config handling, GRPO reward and advantage math, emotion-model pieces and
  system parsing (`pytest`, CPU-only).
