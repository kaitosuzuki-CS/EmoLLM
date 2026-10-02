# Usage guide

Everything you can run, what it does, and what you can change. Installation and the MELD
download are in the [README](../README.md#installation); the exact commands behind each number
in RESULTS.md are in [reproduce.md](reproduce.md); the LLM design is in [llm.md](llm.md).

- [1. Conventions](#1-conventions)
- [2. Data preparation](#2-data-preparation)
- [3. Training emotion models](#3-training-emotion-models)
- [4. Evaluating emotion models](#4-evaluating-emotion-models)
- [5. The emotion-conditioned LLM](#5-the-emotion-conditioned-llm)
- [6. Evaluating the LLM](#6-evaluating-the-llm)
- [7. Configuration reference](#7-configuration-reference)
- [8. Outputs](#8-outputs)

To talk to a trained policy with your voice, see [chat.md](chat.md).

## 1. Conventions

All commands run from the repository root with the `emotion` environment active.

- **Configs and overrides.** Training and data scripts take `--config <yaml>` plus any number of
  `dotted.key=value` overrides, parsed as YAML (`true`, `null`, `[a, b]`, numbers). For example
  `train.lr=3e-4 pooling.type=attention`.
- **Data location.** Every data path is built from `data_root` (default `data/meld`); pass
  `data_root=/path/to/meld` (or `--data-root` where a script has that flag) to use another location.
- **Splits.** `train`, `dev` and `test` (MELD's three splits). Training uses `train`; model
  selection (best epoch, early comparisons) uses `dev`; `test` is for final evaluation only.
  Every evaluation script takes `--split` (default `dev`).
- **Checkpoints.** An emotion-model checkpoint is a run directory (its `best.pt` is used) or a
  `.pt` file such as `runs/finetune_<stamp>/last.pt`.
- **LLM systems.** LLM evaluations take *systems*:

  | System | Meaning |
  |---|---|
  | `runs/llm_grpo_<stamp>` (or `llm_sft_…`) | A trained policy, fed the emotion features it was trained on |
  | `runs/llm_grpo_<stamp>:shuffled` | The same policy, fed *other* utterances' emotion features (does it use them?) |
  | `baseline:none` | The base LLM (Qwen2.5-3B-Instruct), text only, no emotion information |
  | `baseline:predicted` | The base LLM, told the emotion model's predicted emotion as a word |
  | `baseline:oracle` | The base LLM, told the ground-truth emotion as a word (upper bound) |
  | `path/to/replies.jsonl` | Your own replies, one `{"id": "dia<D>_utt<U>", "reply": "..."}` per line |

- **Logging.** Training logs to stdout, `runs/<run>/log.jsonl` and Weights & Biases; add
  `wandb.mode=offline` or `wandb.mode=disabled` to keep it local.

## 2. Data preparation

Expected layout after downloading MELD:

```
data/meld/MELD.Raw/train_splits/dia<D>_utt<U>.mp4                  train_sent_emo.csv
data/meld/MELD.Raw/dev_splits_complete/dia<D>_utt<U>.mp4           dev_sent_emo.csv
data/meld/MELD.Raw/output_repeated_splits_test/dia<D>_utt<U>.mp4   test_sent_emo.csv
```

Each utterance is its own clip, matched to its transcript and label by dialogue and utterance id.
Other files in the folders (macOS `._*` metadata, the test folder's `final_videos_test*`
repeats) are never read.

```bash
python scripts/data/prepare_audio.py --config configs/emotion/base.yaml --splits train,dev,test
python scripts/data/extract_embeddings.py --config configs/emotion/base.yaml --splits train,dev,test --frames
```

| Script | What it does | Options |
|---|---|---|
| `prepare_audio.py` | Decodes every clip to 16 kHz mono wav (ffmpeg) and writes one manifest per split; drops unreadable clips and clips shorter than `data.min_seconds` | `--splits` (default `train,dev`), `--workers` (32), overrides |
| `extract_embeddings.py` | Caches the frozen emotion2vec+ utterance embedding `z` (768-d, ~35 MB per 10k clips); `--frames` also caches frame features (~2.6 GB fp16 for train), needed for attention pooling on a frozen encoder | `--splits` (default `train,dev`), `--frames`, overrides (e.g. `encoder.autocast_bf16=true` for ~2× speed) |

Fine-tuning reads raw audio and needs only `prepare_audio.py`. The frozen pipelines need the
caches of `train` and the split they validate on.

## 3. Training emotion models

The model: emotion2vec+ base → pooling (mean or attention) → `z` (768-d) → projector
(Linear → GELU → Linear) → `h` (D = 256), then a head or loss on `h`.

| Pipeline | Command | Encoder | Trains | Objective |
|---|---|---|---|---|
| Frozen classification | `python scripts/emotion/train_classifier.py --config configs/emotion/classification.yaml` | frozen, cached features | pooler (if attention), projector, linear head | weighted cross-entropy |
| Frozen SupCon | `python scripts/emotion/train_supcon.py --config configs/emotion/supcon.yaml` | frozen, cached features | pooler (if attention), projector | SupCon + VICReg variance / covariance |
| Fine-tuned classification | `python scripts/emotion/train_finetune.py --config configs/emotion/finetune.yaml` | trained on raw audio | transformer, pooler, projector, head | weighted cross-entropy |
| Fine-tuned SupCon | `python scripts/emotion/train_finetune.py --config configs/emotion/finetune_supcon.yaml` | trained on raw audio | transformer, pooler, projector | SupCon + variance / covariance |

Every run validates on `data.val_split` (dev) each epoch and keeps the best epoch as `best.pt`.
Classification selects on the head's weighted F1. SupCon has no classifier during training, so
it selects on a kNN classifier over the latent (train split as the bank). At the end, a
logistic-regression probe is fit on the best checkpoint's latent (`eval.linear_probe`) and,
with `eval.save_probe=true`, saved as `linear_probe.pt`.

**What you'll most often change** (full list in [§7.1](#71-emotion-models-configsemotionyaml)):

| To change | Override | Notes |
|---|---|---|
| Pooling | `pooling.type=attention` | Frozen pipelines then need the `--frames` cache |
| Latent size | `model.out_dim=128 model.hidden_dim=512` | `D` is what the LLM's soft prompts are built from |
| Length / learning rate | `train.epochs=…`, `train.lr=…`, `train.encoder_lr=…` (fine-tuning) | Fine-tuning trains only the heads for `train.freeze_encoder_epochs` (1) first |
| SupCon terms | `loss.var_weight=0 loss.cov_weight=0` (plain SupCon), `loss.temperature=…` | See the ablations in RESULTS.md |
| Class imbalance | `loss.class_weights=inverse` / `null`, `train.sampler=class_balanced` | SupCon uses class-balanced batches by default |
| Start fine-tuning from a frozen run | `train.init_head_from=runs/classification_<stamp>/best.pt` | Copies its projector and head (same objective and sizes) |
| Save the probe | `eval.save_probe=true` | On by default for fine-tuned SupCon |
| Validation split | `data.val_split=dev` | Keep dev: test is for final evaluation only |

Each run writes `runs/<task>_<stamp>/` with the resolved config, logs, `best.pt`, `last.pt` and
`stats.json` (parameter counts, FLOPs per clip, training time, throughput, peak memory).

## 4. Evaluating emotion models

### Score one or more checkpoints on a split

```bash
python scripts/emotion/evaluate.py runs/finetune_<stamp> --split test
python scripts/emotion/evaluate.py runs/classification_<A> runs/supcon_<B> --split test --fit-probe
```

For each checkpoint, on the chosen split:
- **kNN:** cosine kNN on `h`, with the train split as the bank (`--k`, `--knn-temperature`);
- **linear probe:** the saved `linear_probe.pt`, or with `--fit-probe` one refit on train
  exactly as training does;
- **classification head:** for classification checkpoints.

It prints accuracy, weighted and macro F1, per-class F1 and the confusion matrix, and writes
`<run>/eval_<split>_<checkpoint>.json`.

With **two or more checkpoints**, each is also compared with the first. The comparison is a
paired bootstrap over utterances of the difference in their headline score: the head if
present, else the probe, else kNN. `--metric` picks `weighted_f1` (default), `macro_f1` or
`acc`. It prints the difference, its 95% interval and P(better), and writes
`runs/compare_<split>.json` (`--out`).

| Option | Default | Meaning |
|---|---|---|
| `checkpoints` | required | One or more run directories or `.pt` files |
| `--split` | dev | Any split in the run's config or in `--data-config` (so `test` works for runs trained before it was added) |
| `--fit-probe` | off | Refit the probe on train when none was saved |
| `--manifest`, `--name` | – | Evaluate on your own manifest (CSV with `id`, `path` to a 16 kHz wav, integer `label` in the order neutral, joy, surprise, anger, sadness, disgust, fear, and `duration` in s) instead of a split |
| `--metric`, `--n-boot` | weighted_f1, 2000 | Comparison metric and bootstrap resamples |
| `--k`, `--knn-temperature` | 20, 0.07 | kNN settings |
| `--data-config`, `--data-root` | configs/emotion/base.yaml, its own | Where splits are defined |
| `--batch-size`, `--device` | 64, cuda | |

### Untrained baseline

```bash
python scripts/emotion/knn_baseline.py --split test
```

Cosine kNN on the untrained emotion2vec+ embeddings (train split as the bank). Options: `--k`,
`--temperature`, `--config`, overrides. It needs the split's embedding cache.

### Latent-space analysis

```bash
python scripts/emotion/plot_latent_space.py runs/finetune_<stamp>                  # train + dev + test
python scripts/emotion/plot_latent_space.py runs/finetune_<stamp> --split test     # train + test
```

How separated are the emotion classes, on train (seen in training) and on held-out splits?
- **Features compared:** the untrained embedding (`raw`), the checkpoint's pooled embedding `z`,
  and its latent `h`.
- **Projections:** PCA and LDA are fit on train and applied to the held-out splits; t-SNE is
  run per split.
- **Plots:** small multiples, one panel per emotion and one row per split.
- **Metrics:** silhouette, Fisher ratio, kNN / probe F1 and class-centroid cosine similarity,
  written to `metrics.json`.

Options: `--split` (default `dev,test`), `--sources` (`raw,z,h`), `--methods` (`pca,lda,tsne`),
`--tsne-n` (3000 points per split), `--out` (`runs/latent_<run>`), `--name`, `--data-root`.

### Summary table of runs

```bash
python scripts/emotion/summarize_runs.py                # every run under runs/
python scripts/emotion/summarize_runs.py runs/finetune_*
```

Prints a markdown table from each run's `stats.json`: best dev scores, epochs, time, parameters,
FLOPs and memory.

## 5. The emotion-conditioned LLM

A frozen emotion model turns each utterance into `h`. A projector maps `h` to K = 4 soft-prompt
tokens for Qwen2.5-3B-Instruct (LoRA). The policy sees those tokens with the dialogue context
and transcript, and writes an empathetic reply. Training is SFT on pseudo labels picked by a
judge LLM, then GRPO with the judge's scores as reward. Design and rationale: [llm.md](llm.md).

### Judge servers

The judges run as OpenAI-compatible vLLM servers (in the separate `vllm` environment), on a
different GPU from the policy:

```bash
vllm serve Qwen/Qwen2.5-32B-Instruct --port 8000 --max-model-len 4096 --gpu-memory-utilization 0.92   # training judge
vllm serve Qwen/Qwen2.5-3B-Instruct  --port 8001 --max-model-len 4096 --gpu-memory-utilization 0.25   # pseudo-label candidates
vllm serve mistralai/Mistral-Small-3.2-24B-Instruct-2506 --port 8002 --max-model-len 4096 \
    --tokenizer-mode mistral --config-format mistral --load-format mistral                            # held-out judge
```

| Server | Needed by | Config |
|---|---|---|
| Qwen2.5-32B, port 8000 | pseudo labels, GRPO rewards, `scripts/llm/evaluate.py` | `judge.*` |
| Qwen2.5-3B, port 8001 | pseudo labels | `pseudo.candidate_*` |
| Mistral-Small-3.2, port 8002 | `scripts/llm/evaluate_pairwise.py` | `pairwise_judge.*` |

Any OpenAI-compatible endpoint works: set `judge.base_url` / `judge.model` (and the
`pairwise_judge.*` equivalents).

### Training steps

Use one feature cache per emotion model, e.g. `C="emotion.cache_dir=data/meld/llm/cls_attention"`:

```bash
python scripts/llm/extract_features.py emotion.checkpoint=runs/finetune_<stamp>/best.pt $C   # 1. emotion features
python scripts/llm/make_pseudo_labels.py $C                                                 # 2. pseudo labels
python scripts/llm/train_sft.py $C                                                          # 3. SFT
python scripts/llm/train_grpo.py $C grpo.init_from=runs/llm_sft_<stamp>                     # 4. GRPO
```

| Step | What it does | Time (A100) | Key options |
|---|---|---|---|
| 1. `extract_features.py` | Runs the frozen emotion model over each split (`--splits`, default `train,dev`; add `test` to evaluate on it). It caches `h` and `z`, the predicted emotion, transcripts and dialogue context. SupCon checkpoints need a saved `linear_probe.pt` for the predictions. | ~5 min | `emotion.checkpoint` (required), `emotion.cache_dir`, `emotion.feature` (`h` or `z`) |
| 2. `make_pseudo_labels.py` | For train and dev: generates `pseudo.num_candidates` replies with the 3B model (told the true emotion as a word), and the 32B judge picks the best. Cached and resumable. | ~20 min | `pseudo.num_candidates`, `pseudo.limit`, `--splits` |
| 3. `train_sft.py` | Trains LoRA + projector on the pseudo labels. Keeps the lowest-dev-loss checkpoint as the policy. | ~50 min | `sft.epochs`, `sft.lr`, `sft.projector_lr`, `policy.num_emotion_tokens`, `policy.lora.*` |
| 4. `train_grpo.py` | Starts from the SFT run: merges it, adds a fresh LoRA, and uses the SFT policy as the KL reference. Samples `group_size` replies per prompt, scores them with the judge, and applies a clipped policy gradient with group-relative advantages. | ~2–2.5 h | `grpo.steps`, `grpo.beta`, `grpo.group_size`, `grpo.temperature`, `grpo.reward.*` |

GRPO reward variants (`grpo.reward.*`):

| Override | Effect |
|---|---|
| `judge_weight=1.0` (default) | Judge score (1–10) given the true emotion, rescaled to 0–1 |
| `contrast_weight=0.5` | Adds the score under the true emotion minus the score under a counterfactual one (`contrast_label`: `valence` flip or `random`). Rewards replies specific to the speaker's emotion. |
| `pmi_weight=0.1 pmi_clip=0.5` | Adds a bonus when the reply is more likely under the true emotion features than under another emotion's. Values above ~0.1 get gamed (short, emotion-word-heavy replies). |

Each run writes `runs/llm_{sft,grpo}_<stamp>/`: `adapter/` (LoRA), `projector.pt`,
`policy.json` (everything needed to rebuild the policy), logs, samples, `stats.json`.

## 6. Evaluating the LLM

### Judge scores for one system

```bash
python scripts/llm/evaluate.py runs/llm_grpo_<stamp> --split test
python scripts/llm/evaluate.py runs/llm_grpo_<stamp>:shuffled --split test
python scripts/llm/evaluate.py baseline:none --split test emotion.cache_dir=data/meld/llm/cls_attention
```

Generates greedy replies for the split, or reuses the system's cached replies. The training
judge (Qwen2.5-32B) then scores each reply from 1 to 10, given the context, transcript and
true emotion. Prints the mean score, per-emotion means and the reply length. Needs the
training-judge server.

| Option | Default | Meaning |
|---|---|---|
| `system` | required | See [§1](#1-conventions) |
| `--split` | dev | Data split |
| `--limit` | 0 (all) | First N utterances only |
| `--no-judge` | off | Only generate and cache the replies |
| `--regenerate` | off | Ignore cached replies |
| `--out` | the run dir / `runs/llm_baselines` | Output directory |
| `--batch-size` | 16 | Generation batch size |
| overrides | – | e.g. `data_root=…`; `emotion.cache_dir=…` for baselines (policies always use their own features) |

### Pairwise comparison and emotion alignment of two systems

```bash
python scripts/llm/evaluate_pairwise.py runs/llm_grpo_<stamp> baseline:none --split test
python scripts/llm/evaluate_pairwise.py runs/llm_grpo_<stamp> runs/llm_grpo_<stamp>:shuffled --split test
python scripts/llm/evaluate_pairwise.py runs/llm_grpo_<A> runs/llm_grpo_<B> --split test --counterfactual valence
```

The held-out judge (Mistral-Small-3.2, a different model family from the training judge) sees
the context, the transcript, an emotion label and both replies, and picks the better one.

**Win rate of A.**
- Each pair is judged in both orders. A win needs both orders to agree; otherwise it's a tie.
- Win rate = (wins + ½ ties) / n, with a 95% bootstrap interval.

**Emotion alignment.**
- The comparison is repeated with the true label replaced by a counterfactual one:
  - `--counterfactual random`: another emotion;
  - `--counterfactual valence`: the opposite valence, e.g. joy ↔ sadness.
- Alignment gap = win rate (true label) − win rate (counterfactual), with a paired bootstrap
  interval.
- If A's edge comes from fitting the speaker's actual emotion, it should shrink under the wrong
  label, giving a gap above 0.

Replies are generated (greedy) or reused from each system's reply cache, so the policy GPU is
needed only the first time. Needs the held-out judge server.

| Option | Default | Meaning |
|---|---|---|
| `a`, `b` | required | The two systems; the win rate is A's |
| `--split`, `--limit` | dev, 0 (all) | Data split; first N utterances only |
| `--labels` | true,counterfactual | Label conditions shown to the judge: `true`, `counterfactual`, `predicted` |
| `--counterfactual` | random | `random` (another emotion) or `valence` (valence flip) |
| `--seed` | 0 | Counterfactual labels and bootstrap |
| `--out` | `runs/llm_pairwise/<split>/<A>_vs_<B>.json` | Summary JSON; per-pair outcomes go to the matching `.jsonl` |
| overrides | – | As for `evaluate.py`; a baseline without `emotion.cache_dir` uses the opposing policy's cache |

Typical comparisons:

| A vs B | Question it answers |
|---|---|
| policy vs `baseline:none` | Is the trained, emotion-conditioned policy better than the untrained LLM without emotion? |
| policy vs `policy:shuffled` | Does the policy use the emotion tokens, or would any emotion features do? |
| GRPO run vs its SFT run | What did GRPO add? |
| policy A vs policy B | Which emotion encoder makes the better assistant? |

## 7. Configuration reference

- **Files.** `configs/emotion/base.yaml` holds shared emotion-model settings; the four pipeline
  files inherit it via `base:` (`finetune_supcon` → `finetune` → `base`). `configs/llm.yaml`
  configures the whole LLM pipeline.
- **Interpolation.** Strings may reference other keys as `${key}`, resolved after overrides.
- **YAML numbers.** Scientific notation needs a signed exponent (`1.0e-4`, `6.0e+9`); `6.0e9` is
  read as a string.
- **Saved config.** The resolved config is saved with every run (`runs/<run>/config.yaml`) and
  inside its checkpoints.

### 7.1 Emotion models (`configs/emotion/*.yaml`)

| Key | Default | Meaning |
|---|---|---|
| `data_root` | data/meld | Root of MELD.Raw and every generated file |
| `seed`, `device`, `num_threads` | 42, cuda, 8 | RNG seed; device; CPU threads for torch |
| `output_dir` | runs | Run directories are `<output_dir>/<task>_<timestamp>[_N]` |
| `profile` | true | Params / FLOPs / timings into `stats.json` and checkpoints |
| `data.train_split`, `data.val_split` | train, dev | Splits for training (and kNN bank) and validation |
| `data.splits.<split>.video_dir / label_csv` | MELD.Raw paths | Inputs of `prepare_audio.py` |
| `data.splits.<split>.wav_dir / manifest` | `${data_root}/wav16k/...` | Outputs of `prepare_audio.py` |
| `data.splits.<split>.embedding_cache / frame_cache` | `${data_root}/cache/...` | Outputs of `extract_embeddings.py` (`frame_cache` with `--frames`) |
| `data.use_cache` | true (false in finetune) | Train on cached features instead of running the encoder each batch |
| `data.sample_rate`, `data.min_seconds` | 16000, 0.1 | Audio rate; shorter clips are dropped from the manifest |
| `data.max_seconds`, `data.eval_max_seconds` | 20 (8 in finetune), null (20 in finetune) | Train crop (random) / eval crop (first N s) |
| `data.num_workers` | 8 | DataLoader workers for raw audio |
| `encoder.model`, `encoder.hub` | iic/emotion2vec_plus_base, ms | Backbone (Hugging Face: `hub=hf`, emotion2vec/emotion2vec_plus_base) |
| `encoder.embed_dim` | 768 | Size of z |
| `encoder.autocast_bf16` | false (true in finetune) | bf16 autocast (fp32 matches FunASR exactly) |
| `encoder.extract_batch_size` | 32 | Batch size of `extract_embeddings.py` |
| `encoder.freeze_feature_extractor` | true (finetune) | Keep the CNN waveform front end frozen while fine-tuning |
| `pooling.type`, `pooling.hidden_dim` | mean, 128 | Frames → z: `mean` or `attention` (scorer width) |
| `model.hidden_dim`, `model.out_dim`, `model.dropout` | 512, 256, 0.1 | Projector z → Linear → GELU → Linear → D |
| `model.head_dropout` | 0.1 | Dropout before the classification head |
| `train.epochs` | 50 / 100 (SupCon) / 10 (finetune) | |
| `train.batch_size`, `train.eval_batch_size` | 128 / 256 (SupCon) / 32 (finetune); eval 64 in finetune | |
| `train.sampler` | random / class_balanced / length_bucketed | `class_balanced`: `classes_per_batch` × `samples_per_class` per batch (7 × 32 SupCon, 7 × 16 finetune SupCon); `length_bucketed`: similar-length clips (raw audio) |
| `train.lr`, `train.weight_decay`, `train.grad_clip` | 1e-3, 1e-2, 1.0 | AdamW for heads / projector / pooler |
| `train.warmup_epochs` | 2 (1 in finetune) | Linear warmup, then cosine; per parameter group |
| `train.encoder_lr`, `train.freeze_encoder_epochs` | 2e-5, 1 | Fine-tuning: transformer lr; heads-only epochs first |
| `train.init_head_from` | null | Fine-tuning: start the head from a frozen-pipeline `best.pt` |
| `train.select_metric` | weighted_f1 | Model selection: acc / weighted_f1 / macro_f1 (kNN-based for SupCon) |
| `train.log_every`, `train.eval_every` | 20 steps, 1 epoch | |
| `loss.class_weights`, `loss.label_smoothing` | inverse_sqrt, 0.1 | Cross-entropy class weights: null / inverse / inverse_sqrt |
| `loss.temperature`, `loss.base_temperature` | 0.1, 0.1 | SupCon temperature |
| `loss.supcon_weight`, `loss.var_weight`, `loss.cov_weight` | 1.0, 1.0, 0.04 | total = w_s·SupCon + w_v·L_var + w_c·L_cov (0 / 0 = plain SupCon) |
| `loss.var_gamma`, `loss.var_eps` | 1.0, 1e-4 | Target per-dimension std; std = sqrt(Var + eps) |
| `eval.knn_k`, `eval.knn_temperature` | 20, 0.07 | SupCon validation kNN |
| `eval.linear_probe`, `eval.save_probe` | true; false (supcon) / true (finetune) | Fit a logistic-regression probe on the best checkpoint; save it as `linear_probe.pt` |
| `eval.probe_C`, `eval.probe_class_weight` | 1.0, null | Probe regularization; null / balanced |
| `wandb.enabled / project / entity / name / group / tags / mode` | true, emotion2vec-meld, null, run dir, task, [], online | `mode`: online / offline / disabled |

### 7.2 LLM (`configs/llm.yaml`)

| Key | Default | Meaning |
|---|---|---|
| `data_root`, `output_dir`, `seed`, `device` | data/meld, runs, 42, cuda | As above |
| `param_budget` | 6.0e+9 | Emotion model + projector + LLM + LoRA must stay below (checked at startup) |
| `emotion.checkpoint` | null (required by `extract_features.py`) | Frozen emotion model checkpoint |
| `emotion.feature` | h | LLM input: `h` (D-dim projector latent) or `z` (768-d pooled embedding) |
| `emotion.data_config` | configs/emotion/base.yaml | Where each split's audio manifest is |
| `emotion.cache_dir`, `emotion.batch_size` | `${data_root}/llm/features`, 32 | Feature cache + utterance table (one directory per emotion model); extraction batch size |
| `data.splits.<split>` | MELD.Raw CSVs | Transcript CSV per split |
| `data.train_split`, `data.val_split` | train, dev | Training split; SFT / GRPO validation split |
| `data.context_turns` | 5 | Previous dialogue turns shown in prompts (policy and judges) |
| `policy.model` | Qwen/Qwen2.5-3B-Instruct | Policy LLM |
| `policy.num_emotion_tokens`, `policy.projector_hidden` | 4, 1024 | K soft tokens; latent → Linear(hidden) → GELU → Linear(K × LLM hidden) |
| `policy.gradient_checkpointing` | true | |
| `policy.lora.r / alpha / dropout / target_modules` | 16 / 32 / 0.05 / all attention + MLP projections | LoRA |
| `prompt.system`, `prompt.candidate_system` | see file | Policy system prompt; system prompt for pseudo-label candidates (emotion as a word) |
| `generation.max_new_tokens / temperature / top_p` | 96 / 0.8 / 0.95 | Reply length (all stages); sampling defaults (evaluation is greedy) |
| `judge.base_url / model` | localhost:8000, Qwen/Qwen2.5-32B-Instruct | Training judge |
| `judge.max_concurrency / timeout_s / temperature / max_tokens` | 32 / 300 / 0.0 / 400 | |
| `pairwise_judge.base_url / model` | localhost:8002, mistralai/Mistral-Small-3.2-24B-Instruct-2506 | Held-out judge |
| `pairwise_judge.max_concurrency / timeout_s / temperature / max_tokens` | 32 / 300 / 0.0 / 300 | |
| `pseudo.candidate_base_url / candidate_model` | localhost:8001, Qwen/Qwen2.5-3B-Instruct | Candidate generator |
| `pseudo.num_candidates / candidate_temperature / candidate_emotion_hint` | 4 / 0.9 / true | Candidates see the ground-truth emotion as a word |
| `pseudo.out_dir`, `pseudo.limit` | `${data_root}/llm/pseudo`, null | Label cache; label only the first N utterances per split |
| `sft.epochs / batch_size / grad_accum` | 2 / 8 / 4 | Effective batch 32 |
| `sft.lr`, `sft.projector_lr`, `sft.weight_decay`, `sft.warmup_ratio`, `sft.max_steps` | 2e-4, 1e-4, 0.0, 0.03, null | `max_steps` overrides epochs |
| `sft.eval_every / eval_batches` | 100 / 20 | Dev loss every N steps; the best checkpoint becomes the policy |
| `sft.sample_every / num_samples / log_every / train_limit` | 200 / 4 / 10 / null | Greedy dev samples; logging; train on the first N labels only |
| `grpo.init_from` | null (required) | SFT run directory (merged into the base; the KL reference) |
| `grpo.steps / prompts_per_step / group_size` | 500 / 8 / 4 | Optimizer steps; utterances per step; replies per utterance |
| `grpo.lr`, `grpo.projector_lr`, `grpo.weight_decay`, `grpo.warmup_steps` | 1e-5, 1e-5, 0.0, 10 | |
| `grpo.beta`, `grpo.clip_eps`, `grpo.inner_updates`, `grpo.micro_batch` | 0.04, 0.2, 1, 16 | KL weight (k3); PPO clip; updates per batch; sequences per chunk |
| `grpo.temperature`, `grpo.top_p`, `grpo.max_new_tokens`, `grpo.lora_dropout` | 0.9, 0.95, 96, 0.0 | Sampling (log-probs use the same temperature) |
| `grpo.reward.judge_weight` | 1.0 | Weight of (s_true − 1) / 9 |
| `grpo.reward.contrast_weight`, `grpo.reward.contrast_label` | 0.0, valence | Weight of (s_true − s_cf) / 9; counterfactual label: valence / random |
| `grpo.reward.pmi_weight`, `grpo.reward.pmi_clip` | 0.0, 0.5 | Weight and clip (nats/token) of the emotion-dependence (PMI) bonus |
| `grpo.save_every / eval_every / eval_prompts / log_every` | 50 / 50 / 16 / 1 | Eval = greedy replies on fixed dev prompts, scored by the judge |
| `wandb.*` | project emotion2vec-meld-llm | As above |

## 8. Outputs

| Script | Writes |
|---|---|
| `scripts/data/prepare_audio.py` | `${data_root}/wav16k/<split>/*.wav`, `${data_root}/wav16k/<split>_manifest.csv` |
| `scripts/data/extract_embeddings.py` | `${data_root}/cache/emotion2vec_plus_base_<split>[_frames].pt` |
| Emotion training | `runs/<task>_<stamp>/`: `config.yaml`, `log.jsonl`, `best.pt`, `last.pt`, `stats.json`, `linear_probe.pt` (with `eval.save_probe`) |
| `scripts/emotion/evaluate.py` | `<run>/eval_<split>_<checkpoint>.json` (metrics, per-class F1, confusion matrix); `runs/compare_<split>.json` with several checkpoints |
| `scripts/emotion/plot_latent_space.py` | `<out>/<source>_<method>.png`, `centroid_similarity[_<split>].png`, `metrics.json` |
| `scripts/llm/extract_features.py` | `<emotion.cache_dir>/features_<split>.pt`, `utterances_<split>.jsonl` |
| `scripts/llm/make_pseudo_labels.py` | `<pseudo.out_dir>/<split>.jsonl` |
| `scripts/llm/train_{sft,grpo}.py` | `runs/llm_{sft,grpo}_<stamp>/`: `adapter/`, `projector.pt`, `policy.json`, `samples.jsonl`, `log.jsonl`, `stats.json` |
| `scripts/llm/evaluate.py` | Reply cache `replies_<split>[_shuffled].jsonl` (run dir, or `runs/llm_baselines/` for baselines); `eval_<split>_<system>.jsonl` (reply, score, reason) and `.json` (summary) |
| `scripts/llm/evaluate_pairwise.py` | `<out>.json` (win rates, alignment gap) and `<out>.jsonl` (per-pair outcomes, both orders) |
