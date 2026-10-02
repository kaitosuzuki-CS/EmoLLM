# One utterance, end to end

This page follows a single spoken utterance through the whole system to its two outputs:
1. **the speaker's emotion**, predicted by the emotion model;
2. **an emotion-aware reply**, written by the LLM policy.

Every number below was printed by the terminal demo's trace mode on one NVIDIA A100-SXM4-80GB.
You can rerun it yourself.

```bash
python scripts/llm/chat.py runs/llm_grpo_<stamp> --trace --speaker Monica \
    --wav data/meld/wav16k/test/dia17_utt8.wav data/meld/wav16k/test/dia17_utt9.wav
```

The input is two consecutive MELD test clips. In them, Monica sarcastically "admires" her
ex-husband's mattress commercial, in an angry voice. The policy is the SupCon-encoder GRPO policy
from [RESULTS.md](../RESULTS.md#emotion-conditioned-llm).

## The system

```
                   ┌──────────────────────────────────────────────┐
 speech ──┬──────► │ Whisper large-v3-turbo (ASR)                 │ ──► transcript ─────────────┐
          │        └──────────────────────────────────────────────┘                             │
          │        ┌──────────────────────────────────────────────┐                             ▼
          └──────► │ emotion model                                │      ┌───────────────────────────────────┐
                   │ emotion2vec+ → attention pool → z (768)      │      │ chat prompt: system prompt,        │
                   │ → projector → h (256) ──┬─► probe → softmax  │ ───► │ last 5 turns, transcript, and      │
                   └─────────────────────────┼────────────────────┘      │ 4 soft emotion tokens              │
                                             │   OUTPUT 1: emotion       │        ▼                           │
                                             └─► soft-prompt projector ─►│ Qwen2.5-3B-Instruct + LoRA         │
                                                 (h → 4 × 2048)          └───────────────┬───────────────────┘
                                                                                         ▼
                                                                          OUTPUT 2: reply (text)
```

The emotion reaches the LLM **only** through the 4 soft tokens. The prompt never names it, and
the system prompt asks the policy not to name it either.

## Stage by stage

Startup loads every model once. This takes about 20–25 s (models already downloaded) and 8.3 GB of GPU memory. The demo
prints the parameter inventory:

```
learned parameters on the inference path:
  emotion2vec+ encoder                                            93.8M
  pooling                                                          0.1M
  emotion projector + head                                         0.5M
  linear probe                                                     0.0M
  speech recognition (openai/whisper-large-v3-turbo)             808.9M
  soft-prompt projector                                            8.7M
  LLM (Qwen/Qwen2.5-3B-Instruct)                                3085.9M
  LoRA adapter llm_sft_20261001_051606                            29.9M
  LoRA adapter llm_grpo_20261001_060830                           29.9M
  total                                                          4.058B  (budget 6B)
```

Both LoRA adapters (SFT, then GRPO) are merged into the LLM's weights at load time. They are
counted here anyway, at their unmerged size, so the total is an upper bound.

### Turn 1: `dia17_utt8`

```
[1 audio]    66720 samples = 4.17 s at 16000 Hz, mono float32
heard: "I know. At $4.99 for a pillow top queen set, who cares about the divorce?"   [2 ASR 585 ms]
[3 emotion]  emotion2vec+ frames 208x768 (50 Hz) -> pooled z (768,) -> projector h (256,) -> 7-way softmax   [173 ms]
OUTPUT 1     predicted emotion: anger (0.91)
emotion model hears: anger █████████· 0.91   neutral █········· 0.05   joy ·········· 0.02
[4 prompt]   h -> 4 soft tokens x 2048 (LLM embedding size), placed in a 156-token chat prompt with 0 context turns
[5 LLM]      53 tokens in 2239 ms (24 tokens/s, greedy)

assistant: It sounds like you're really frustrated with the cost. Divorce can be incredibly
stressful, and adding financial concerns to that mix must be overwhelming. Is there anything
specific about the divorce process or the cost of the pillow top queen set that's particularly
bothering you?   <- OUTPUT 2

latency: speech recognition 585 ms + emotion model 173 ms + reply generation 2239 ms = 3.00 s
         after the speaker stops, peak GPU memory 8.6 GB
```

1. **Audio.** The clip is decoded to 16 kHz mono (`librosa`); live recordings come from the
   microphone at 16 kHz directly. MELD's clip boundaries are loose: this clip, annotated
   "I know!", also contains the start of the next line, and Whisper transcribes all of it.
2. **Speech recognition.** Whisper large-v3-turbo (fp16) turns the audio into the transcript.
   The policy was trained on MELD's gold transcripts. At inference it sees ASR output instead,
   errors included.
3. **Emotion model.** This is `main_finetune_supcon_attention`
   ([RESULTS.md](../RESULTS.md#emotion-models)):
   - emotion2vec+ gives one 768-d frame per 20 ms (208 frames for 4.17 s);
   - attention pooling turns the frames into one utterance vector, `z`;
   - the projector maps `z` to the 256-d emotion latent `h`;
   - the linear probe saved with the checkpoint maps `h` to 7 emotion probabilities. Its
     argmax is **output 1**.
   Code: `emollm/emotion_features.py`.
4. **Prompt.** The soft-prompt projector maps `h` to 4 vectors of the LLM's embedding size
   (2048). They are written into 4 placeholder positions of the chat prompt, after the
   transcript, where the prompt says `Monica's vocal emotion:`. Code: `emollm/prompt.py`,
   `EmotionLLM.embed` in `emollm/model.py`.
5. **LLM.** Qwen2.5-3B-Instruct, with the SFT and GRPO LoRA adapters merged, decodes greedily
   (at most 96 new tokens). The result is **output 2**.

The first turn is slower than the rest because of CUDA warm-up and kernel compilation.

### Turn 2: `dia17_utt9`, with the first turn as context

```
[1 audio]    37344 samples = 2.33 s at 16000 Hz, mono float32
heard: "those babies will sell themselves."   [2 ASR 94 ms]
[3 emotion]  emotion2vec+ frames 116x768 (50 Hz) -> pooled z (768,) -> projector h (256,) -> 7-way softmax   [15 ms]
OUTPUT 1     predicted emotion: anger (0.89)
emotion model hears: anger █████████· 0.89   neutral ·········· 0.04   sadness ·········· 0.03
[4 prompt]   h -> 4 soft tokens x 2048 (LLM embedding size), placed in a 223-token chat prompt with 2 context turns
[5 LLM]      38 tokens in 1230 ms (31 tokens/s, greedy)

assistant: Those babies do sound like they could sell themselves, but it seems like the cost is
really weighing on you. Is there anything specific about the situation that's making you feel
this way?   <- OUTPUT 2

latency: speech recognition 94 ms + emotion model 15 ms + reply generation 1230 ms = 1.34 s
         after the speaker stops, peak GPU memory 8.6 GB
```

The context now holds Monica's first line and the assistant's first reply (2 turns), so the
prompt grows from 156 to 223 tokens. The transcript alone ("those babies will sell themselves")
reads as upbeat. The angry voice leads the policy to treat it as frustration rather than
enthusiasm, which is the behavior [RESULTS.md](../RESULTS.md#example-replies) measures at scale.

What this example also shows:
- **Imperfect replies.** The reply names the feeling outright ("it seems like the cost is really
  weighing on you").
- **Misread sarcasm.** It reads Monica's sarcasm as worry about money. Both are known failure
  modes (see [Limitations](../README.md#limitations)).

## The same path, offline

Training and evaluation use the same models, split into cached stages:
- `scripts/llm/extract_features.py` runs step 3 over a whole split and caches `h`, `z` and the
  predicted emotion.
- `scripts/llm/evaluate.py` and `scripts/llm/evaluate_pairwise.py` run steps 4–5 in batches on
  the cached features.

MELD's gold transcripts replace step 2 there. The emotion-recognition output is evaluated
separately by `scripts/emotion/evaluate.py` (see the [usage guide](usage.md)).
