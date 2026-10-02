# Emotion-conditioned LLM

A small instruct LLM that replies to a speaker in a conversation, conditioned on how the speaker
*sounds*. The speaker's vocal emotion reaches the LLM only through soft-prompt tokens computed from
a frozen speech-emotion model. Results are in [RESULTS.md](../RESULTS.md#emotion-conditioned-llm);
commands are in [reproduce.md](reproduce.md#emotion-conditioned-llm).

## Architecture

```
utterance audio (MELD clip dia<D>_utt<U>.mp4, ≤ 20 s)
  → frozen emotion model (emotion2vec+ → pooling → projector)      scripts/llm/extract_features.py
  → emotion latent h (256-d), cached per utterance
  → EmotionPromptProjector: LayerNorm → Linear 1024 → GELU → Linear K·2048, plus a learned
    per-slot offset; each token normalized and scaled to the LLM's token-embedding RMS
  → K = 4 soft tokens fill a placeholder in the user turn
  → Qwen2.5-3B-Instruct + LoRA (r 16, α 32, all attention and MLP projections) → reply
```

Prompt (user turn; the system prompt asks for a warm, specific 1–3 sentence reply that does not
name the emotion):

```
Conversation so far:
<speaker>: <text>            (up to 5 previous turns)

<speaker> now says: "<transcript>"
<speaker>'s vocal emotion: <e1><e2><e3><e4>
Reply to <speaker>.
```

| Component | Parameters | Trainable |
|---|---|---|
| Emotion model (frozen) | 94.4M | 0 |
| Soft-prompt projector | 8.7M | 8.7M |
| Qwen2.5-3B-Instruct | 3.086B | 0 |
| LoRA | 29.9M | 29.9M |
| **Total** | **3.22B** (must stay < `param_budget` = 6B; checked at startup) | **38.6M** |

This table is the trained model: emotion model, projector and LLM, with one LoRA. The
complete local inference path in the voice demo also includes Whisper large-v3-turbo (809M) and
counts both LoRA adapters (SFT and GRPO, merged at load) separately: **4.058B** in total (see
the [README](../README.md#inference-path-and-resources)). The judge LLMs are not part of the
model; they are served separately with vLLM for training and evaluation only.

| Judge | Used for | Sees |
|---|---|---|
| Qwen2.5-32B-Instruct ("training judge") | pseudo labels, GRPO rewards, `scripts/llm/evaluate.py` scores | ground-truth emotion, context, transcript |
| Mistral-Small-3.2-24B-Instruct ("held-out judge") | pairwise evaluation only | same, or a counterfactual emotion (alignment test) |

Using a different model family for evaluation avoids grading the policy with the model it was
optimized against.

## Training

### 1. Pseudo labels (`scripts/llm/make_pseudo_labels.py`)

MELD has no reference replies, so labels are generated:
- Qwen2.5-3B-Instruct writes 4 candidate replies, given the ground-truth emotion as a word (the
  soft-prompt projector is not trained yet).
- The training judge picks the best candidate and refines it.

Labels are text-only, so they are shared by every emotion model. They are appended to
`<data_root>/llm/pseudo/<split>.jsonl`, and existing ids are skipped, so interrupted runs resume
and labels are never regenerated.

### 2. SFT (`scripts/llm/train_sft.py`)

- **Objective:** cross-entropy on the pseudo-label reply tokens only.
- **Optimization:** LoRA at lr 2e-4, projector at 1e-4, effective batch 32, 2 epochs.
- **Checkpoint:** dev loss is evaluated every 100 steps; the lowest-dev-loss checkpoint is saved
  as the policy (`adapter/`, `projector.pt`, `policy.json`), the final step as `last_*`.
- **Garbled-output check:** each time a new best is saved, sampled replies are checked for
  garbage (`val/garbled_rate`).

### 3. GRPO (`scripts/llm/train_grpo.py`)

**Reference model.** The SFT LoRA is merged into the base weights and a fresh LoRA is added. With
the adapter disabled (plus a frozen copy of the SFT projector), the model is exactly the SFT
policy, which serves as the KL reference. Only the new LoRA and the projector train.

**Each step:**
- sample 4 replies (T 0.9, top-p 0.95) for each of 8 train utterances;
- score them, then normalize rewards within each group of 4 (group-relative advantage);
- loss = PPO-clipped policy gradient + 0.04 · KL to the reference (k3 estimator), with
  per-token log-probs from the temperature-scaled distribution the replies were sampled from.

**Reward** (`emollm/grpo.py`, config `grpo.reward`):

```
reward = judge_weight    · (s_true − 1) / 9
       + contrast_weight · (s_true − s_cf) / 9
       + pmi_weight      · clip(PMI, ±pmi_clip)
```

- **s_true:** training-judge score (1–10) of the reply given the ground-truth emotion.
- **Contrastive term:**
  - s_cf is the score of the *same reply* when the judge is shown a counterfactual emotion as
    the annotation, by default a valence flip (`contrast_label: valence`): anger / sadness /
    disgust / fear → joy, joy / surprise → sadness, neutral → anger.
  - A generic reply that fits any emotion (e.g. blanket sympathy) gets no contrast, so this
    rewards replies specific to the speaker's actual emotion.
- **PMI bonus:** the mean per-token log p(reply | prompt, true emotion features) −
  log p(reply | prompt, features of a random train utterance with a different emotion), under
  the current policy.
  - It is positive only for replies that depend on the emotion tokens; a policy that ignores
    them gets 0.
  - It costs one extra forward pass and no extra judge calls.
- **Plain GRPO:** with `contrast_weight = pmi_weight = 0` the reward is the judge score alone.

The default is the plain judge reward (contrast and PMI weights 0), which gave the best policies.
The emotion-specific variant evaluated in RESULTS.md uses contrast 0.5 and PMI 0.1 (clip 0.5
nat/token, so the bonus is a tiebreaker worth at most half a judge point): it made replies somewhat
more emotion-dependent but lost to plain GRPO overall. The contrastive term doubles
the judge calls per step. Both extra terms also break the judge's frequent ties (integer
scores at temperature 0), which otherwise leave about a quarter of groups with no learning
signal.

## Evaluation

Both evaluation scripts take a data split (`--split`) and *systems*
(`emollm/systems.py`; commands in the [usage guide](usage.md#6-evaluating-the-llm)):
- **A policy run.** It always gets the emotion feature cache recorded in that run's config.
- **`<run>:shuffled`.** Every utterance gets another utterance's features (do the tokens
  matter?).
- **`baseline:none|predicted|oracle`.** Text-only base-model baselines, with no emotion, the
  emotion model's prediction, or the ground-truth emotion given as a word.
- **A reply file.**

Replies are greedy and cached per system and split.

**`scripts/llm/evaluate.py`** scores one system's replies 1–10 with the training judge.

**`scripts/llm/evaluate_pairwise.py`** compares two systems with the held-out judge.
- **Both orders:** each pair is judged with A first and with B first; a win counts only if both
  orders agree, otherwise the pair is a tie.
- **Win rate:** (wins + 0.5 · ties) / n, with a 95% bootstrap interval.
- **Emotion alignment:** the comparison is repeated with the judge shown a counterfactual emotion
  (`--counterfactual random|valence`).
  - **Alignment gap** = win rate(true label) − win rate(counterfactual), with a paired bootstrap
    interval.
  - It is positive only when A's advantage comes from fitting the speaker's real emotion.

`scripts/reproduce_llm.sh` runs these comparisons for every policy:
- vs the no-emotion baseline (random and valence counterfactuals),
- vs itself with shuffled emotion features,
- vs its SFT checkpoint,
- the policies against each other.

## Design notes

- **Projector learning rate.** At 1e-3, the soft tokens drift off the embedding manifold within a
  few SFT steps. Greedy replies still look fine, but sampled replies degenerate into
  multilingual garbage, which breaks GRPO sampling. 1e-4 is safe; SFT logs the garbled rate.
- **Sampling temperature.** Full-vocabulary T = 1.0 sampling also derails Qwen into other
  scripts. GRPO samples at T 0.9 / top-p 0.95, and its log-probs use the same temperature.
- **Feature caches must match.** A policy must be evaluated with the emotion features of the
  model it was trained on. Each emotion model gets its own cache directory, recorded in the run
  config and picked up automatically by `scripts/llm/evaluate.py`.
- **Emotion signal gap.** The emotion models are ~86–91% accurate on train clips (which they were
  trained on) but ~45–47% on dev. The LLM therefore sees a far more reliable emotion signal during
  training than on new audio.
- **Pseudo-label quality.** Most judge-refined labels are good. Some confuse who is speaking in
  multi-party scenes. A second judge pass that drops low-scoring labels would be a cheap
  improvement.
- **Keep the PMI bonus small.** At weight 0.5 / clip 1.0 it outweighed the judge: within 150
  GRPO steps the policy learned short replies whose few tokens depend strongly on the emotion
  features. PMI rose (0.17 → 0.58 nats/token) while the judge score fell and KL grew 100×.
  Watch `reward_judge` alongside `reward_pmi` in the GRPO logs.
- **Judge length bias.** Policies' replies get longer than the baseline's (≈ 21–24 vs 13 words).
  Although the judges are told not to prefer length, the held-out judge picks the longer reply
  12–14 points more often in direct comparisons. The alignment gap and the shuffled-features
  comparison are robust to this, because they hold the reply (or its length) fixed; raw win
  rates against the baseline are not.
