# Talking to the assistant

`scripts/llm/chat.py` is a terminal interface to a trained policy. You speak; it hears both
*what* you said and *how* you said it, and replies.

Each turn runs four steps:
1. **Record** your voice (or read an audio file).
2. **Transcribe** it with Whisper.
3. **Hear the emotion.** The policy's own frozen emotion model turns the audio into the
   emotion features the policy was trained on. Its top-3 emotions are shown so you can see what
   it heard.
4. **Reply.** The policy answers, with the last few turns of the conversation as context.

With `--baseline`, the untrained LLM without emotion information answers too, for comparison.

```
> 
  recording... press Enter to stop
  heard: "You know what? There's no more left left."
  emotion model hears: anger ████████·· 0.85   neutral █········· 0.12   joy ·········· 0.02

  assistant: I can hear the frustration in your voice. It sounds like you're really upset about this situation. Can you tell me more about what's going on?
  no-emotion baseline: Ah, I see. Do you think you've covered everything then?
```

## Requirements

- **A trained policy run:** `runs/llm_grpo_<stamp>` or `runs/llm_sft_<stamp>`
  (see [usage §5](usage.md#5-the-emotion-conditioned-llm)).
- **Its emotion features.** The policy's feature cache must still exist (`emotion.cache_dir` in
  the run's `config.yaml`), so the demo can find the emotion model the policy was trained with.
  Otherwise pass `--emotion-checkpoint`.
- **A GPU with ~9 GB free.** Measured peak: 8.6 GB, or 14.8 GB with `--baseline`.
  - On an A100, a reply arrives a median 1.26 s after you stop speaking.
  - `--device cpu` works but takes 47–70 s per turn and 12 GB of RAM (measured on a shared
    EPYC 7742 node).
- **A microphone on the machine running the script**, plus PortAudio and `sounddevice`. These are
  only needed for live recording; audio files work without them.

  ```bash
  pip install sounddevice
  sudo apt install libportaudio2        # Debian/Ubuntu; macOS: brew install portaudio
  ```

The first run downloads Whisper (`openai/whisper-large-v3-turbo`, 1.6 GB) and, if not cached,
Qwen2.5-3B-Instruct and emotion2vec+.

## Running it

```bash
python scripts/llm/chat.py runs/llm_grpo_<stamp>
```

At the `>` prompt:

| Input | Action |
|---|---|
| Enter | Start recording; Enter again stops and sends the turn |
| a file path | Use that audio file as your turn (wav, flac, mp3, or a video with audio; resampled to 16 kHz mono) |
| `/reset` | Forget the conversation so far |
| `/quit` (or Ctrl-D) | Exit |

Speak for at least half a second, in English. Like MELD utterances, one or two sentences work
best. Only the first 20 s are used for emotion recognition (the emotion model's evaluation crop)
and the first 30 s for transcription (Whisper's window).

### Common variations

```bash
# show every stage: tensor shapes, per-stage latency, memory and the parameter inventory
python scripts/llm/chat.py runs/llm_grpo_<stamp> --trace

# compare with the base LLM that only sees the words
python scripts/llm/chat.py runs/llm_grpo_<stamp> --baseline

# correct transcription errors before the policy sees them
python scripts/llm/chat.py runs/llm_grpo_<stamp> --edit

# no speech recognition: type what you said (the emotion still comes from the audio)
python scripts/llm/chat.py runs/llm_grpo_<stamp> --asr none

# non-interactive: each file is one turn of a single conversation
python scripts/llm/chat.py runs/llm_grpo_<stamp> --wav turn1.wav turn2.wav --baseline

# pick a microphone
python scripts/llm/chat.py runs/llm_grpo_<stamp> --list-devices
python scripts/llm/chat.py runs/llm_grpo_<stamp> --input-device 2
```

### On a remote GPU server

A server has no microphone. Either run the demo on a local machine with a GPU, or record locally
and pass the files:

```bash
# local machine: record a 16 kHz mono clip (Linux: arecord; macOS: sox's rec), then copy it
arecord -f S16_LE -r 16000 -c 1 turn1.wav          # Ctrl-C to stop
rec -r 16000 -c 1 turn1.wav                        # macOS (brew install sox)
scp turn1.wav server:emotion-llm/

# server
python scripts/llm/chat.py runs/llm_grpo_<stamp> --wav turn1.wav
```

You can also type a copied file's path at the interactive prompt, so a conversation can continue
turn by turn.

## Options

| Option | Default | Meaning |
|---|---|---|
| `run` | required | Policy run directory |
| `--wav FILE ...` | – | Use these files as consecutive turns, then exit |
| `--asr MODEL` | openai/whisper-large-v3-turbo | Any Hugging Face speech-recognition model (`openai/whisper-small` is lighter); `none` to type transcripts |
| `--language` | en | ASR language; `auto` to detect |
| `--edit` | off | Confirm or correct each transcript |
| `--speaker` | You | Your name in the policy's prompt |
| `--baseline` | off | Also show the reply of the base LLM without emotion information |
| `--sample` | off (greedy) | Sample replies with the config's `generation.temperature` / `top_p` |
| `--emotion-checkpoint` | from the policy's feature cache | Emotion model to use. Must be the one the policy was trained with: a different model's latent means nothing to the policy |
| `--input-device`, `--list-devices` | system default | Microphone selection |
| `--log` | `runs/chat/<stamp>.jsonl` | Session transcript |
| `--save-audio` | off | Also save each recording next to the log |
| `--trace` | off | Print each stage (shapes, latency, peak memory), both outputs, and the learned-parameter count of the inference path ([walkthrough](walkthrough.md)) |
| `--device` | cuda | `cpu` works, but takes about a minute per turn |

Each line of the log holds:
- the turn number;
- the audio file, if any;
- the transcript;
- the predicted emotion and all seven probabilities;
- per-stage latency;
- the reply, and the baseline reply with `--baseline`.

## What to expect

- **The policy hears emotion only through the soft tokens.** Its prompt never names an emotion,
  so the bars show what the emotion model heard, not something the policy was told. Try one
  sentence said two ways (calmly, then angrily) and compare the replies.
- **The emotion model was trained on MELD:** acted, English sitcom speech, at 0.45–0.46 weighted
  F1 on MELD test. It is often wrong on real speech, and it favors neutral.
- **Replies to strongly negative speech are formulaic.** GRPO pushed the policy toward the
  judge's favorite empathetic template ("I can hear the frustration in your voice…"), so
  different angry utterances often get the same reply (see RESULTS.md).
- **Context is the last `data.context_turns` (5) turns,** including the assistant's own replies.
  Use `/reset` to start over.

## Troubleshooting

| Problem | Fix |
|---|---|
| `microphone input needs sounddevice and PortAudio` | Install both (above), or pass audio files |
| `Error querying device -1` / no default input device | `--list-devices`, then `--input-device <index>` |
| `cannot find the policy's feature cache` | Pass `--emotion-checkpoint runs/<emotion run>/best.pt` (the policy's emotion model) |
| CUDA out of memory | Drop `--baseline`, use `--asr openai/whisper-small`, or another GPU |
| Wrong transcript | `--edit`, or `--asr none` and type it |
| `(too short, try again)` | Recordings under 0.3 s are ignored; hold Enter-to-stop a little longer |
