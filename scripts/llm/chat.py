"""Talk to a trained emotion-conditioned policy from the terminal.

Each turn: record your voice (or give an audio file) -> speech recognition gives the
transcript -> the policy's frozen emotion model hears *how* you said it -> the policy replies,
with the last few turns of the conversation as context.

    python scripts/llm/chat.py runs/llm_grpo_<stamp>                       # microphone
    python scripts/llm/chat.py runs/llm_grpo_<stamp> --baseline            # also show the no-emotion LLM's reply
    python scripts/llm/chat.py runs/llm_grpo_<stamp> --asr none            # type what you said instead of ASR
    python scripts/llm/chat.py runs/llm_grpo_<stamp> --wav a.wav b.wav     # non-interactive: one turn per file

At the prompt: Enter starts recording and Enter again stops it; or type the path of an audio
file; `/reset` clears the conversation, `/quit` exits. See docs/chat.md.
"""
import argparse
import json
import logging
import resource
import sys
import time
from pathlib import Path

import numpy as np
import torch

from emollm.data import cache_paths
from emollm.emotion_features import EmotionFeaturizer
from emollm.model import load_llm, load_policy, resolve_saved_path
from emollm.prompt import Collator, no_emotion_messages
from emollm.utils import baseline_texts, greedy_kwargs, sample_completions
from emotion.config import load_config


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("run", help="policy run directory (runs/llm_grpo_<stamp> or runs/llm_sft_<stamp>)")
    ap.add_argument("--wav", nargs="+", help="audio files to use as consecutive turns, then exit")
    ap.add_argument("--asr", default="openai/whisper-large-v3-turbo",
                    help="Hugging Face speech-recognition model, or 'none' to type transcripts")
    ap.add_argument("--language", default="en", help="ASR language (default en; 'auto' to detect)")
    ap.add_argument("--edit", action="store_true", help="confirm or correct each ASR transcript")
    ap.add_argument("--speaker", default="You", help="your name in the prompt")
    ap.add_argument("--baseline", action="store_true",
                    help="also reply with the base LLM without emotion information, for comparison")
    ap.add_argument("--sample", action="store_true",
                    help="sample replies (config generation.temperature / top_p) instead of greedy")
    ap.add_argument("--emotion-checkpoint", help="emotion model (default: the one the policy's "
                                                 "features were extracted with)")
    ap.add_argument("--input-device", help="microphone (sounddevice index or name; see --list-devices)")
    ap.add_argument("--list-devices", action="store_true", help="list audio devices and exit")
    ap.add_argument("--log", help="JSONL transcript of the session (default: runs/chat/<stamp>.jsonl)")
    ap.add_argument("--save-audio", action="store_true", help="also save each recording next to --log")
    ap.add_argument("--trace", action="store_true",
                    help="show every stage of each turn (tensor shapes, latency, memory) and the "
                         "parameter count of the whole inference path")
    ap.add_argument("--device", default="cuda")
    return ap.parse_args()


def read_audio(path: str, sr: int) -> np.ndarray:
    import librosa  # decodes wav / flac / mp3 / mp4 (via ffmpeg) and resamples
    wav, _ = librosa.load(path, sr=sr, mono=True)
    return wav.astype(np.float32)


def record(sr: int, device=None) -> np.ndarray:
    """Records from the microphone between two presses of Enter."""
    try:
        import sounddevice as sd
    except (ImportError, OSError) as e:
        raise SystemExit(f"microphone input needs sounddevice and PortAudio ({e}); "
                         "pip install sounddevice (Linux: apt install libportaudio2), "
                         "or type the path of an audio file instead") from e
    chunks = []
    with sd.InputStream(samplerate=sr, channels=1, dtype="float32", device=device,
                        callback=lambda data, *_: chunks.append(data.copy())):
        input("  recording... press Enter to stop ")
    return np.concatenate(chunks)[:, 0] if chunks else np.zeros(0, np.float32)


def emotion_checkpoint(run: Path, cfg: dict, override: str | None) -> Path:
    if override:
        return Path(override)
    feats = cache_paths(cfg["emotion"]["cache_dir"], cfg["data"]["train_split"])["features"]
    if not Path(feats).exists():
        raise SystemExit(f"cannot find the policy's feature cache {feats} to look up its emotion "
                         "model; pass --emotion-checkpoint runs/<emotion run>/best.pt")
    ckpt = torch.load(feats, map_location="cpu", weights_only=False)["emotion_checkpoint"]
    return resolve_saved_path(ckpt, run)


def adapter_params(path) -> int:
    from safetensors.torch import load_file
    return sum(t.numel() for t in load_file(str(Path(path) / "adapter_model.safetensors")).values())


def inventory(policy, meta, featurize, asr) -> list[tuple[str, int]]:
    """Every learned parameter on the inference path. LoRA adapters are merged into the LLM at
    load time; they are counted separately here, at their unmerged size."""
    rows = list(featurize.num_params.items())
    if asr is not None:
        rows.append((f"speech recognition ({asr.model.name_or_path})",
                     sum(p.numel() for p in asr.model.parameters())))
    rows.append(("soft-prompt projector", sum(p.numel() for p in policy.projector.parameters())))
    rows.append((f"LLM ({meta['base_model']})", sum(p.numel() for p in policy.llm.parameters())))
    for a in meta["merged_adapters"] + [meta["adapter"]]:
        rows.append((f"LoRA adapter {Path(a).parent.name}", adapter_params(a)))
    return rows


class Timer:
    def __init__(self, device):
        self.device, self.t = device, time.perf_counter()

    def lap(self) -> float:
        if self.device.type == "cuda":
            torch.cuda.synchronize()
        now = time.perf_counter()
        dt, self.t = now - self.t, now
        return dt


def bar(p: float, width: int = 20) -> str:
    n = round(p * width)
    return "█" * n + "·" * (width - n)


def main():
    args = parse_args()
    logging.getLogger("httpx").setLevel(logging.WARNING)
    if args.list_devices:
        import sounddevice as sd
        print(sd.query_devices())
        return
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    run = Path(args.run)
    cfg = load_config(str(run / "config.yaml"))

    print("loading the policy...")
    policy, meta = load_policy(run, device)
    collate = Collator(policy.tokenizer, cfg, policy.k, policy.placeholder_id, "prompt")
    ckpt = emotion_checkpoint(run, cfg, args.emotion_checkpoint)
    print(f"loading the emotion model {ckpt}...")
    featurize = EmotionFeaturizer(ckpt, device)
    sr = featurize.sample_rate
    asr = None
    if args.asr != "none":
        from transformers import pipeline
        print(f"loading speech recognition {args.asr}...")
        asr = pipeline("automatic-speech-recognition", model=args.asr, device=device,
                       torch_dtype=torch.float16 if device.type == "cuda" else torch.float32)
    base = load_llm(cfg["policy"]["model"], device=device).eval() if args.baseline else None

    max_new = cfg["generation"]["max_new_tokens"]
    gen = (dict(do_sample=True, temperature=cfg["generation"]["temperature"],
                top_p=cfg["generation"]["top_p"], max_new_tokens=max_new)
           if args.sample else greedy_kwargs(max_new))
    log = Path(args.log or f"runs/chat/{time.strftime('%Y%m%d_%H%M%S')}.jsonl")
    log.parent.mkdir(parents=True, exist_ok=True)
    history: list[dict] = []
    files = list(args.wav or [])
    print(f"\npolicy {run} ({meta.get('stage', 'policy')}) | emotion feature '{cfg['emotion']['feature']}' "
          f"| context {cfg['data']['context_turns']} turns | log {log}")
    if args.trace:
        rows = inventory(policy, meta, featurize, asr)
        print("\nlearned parameters on the inference path:")
        for name, n in rows:
            print(f"  {name:<58} {n / 1e6:>9.1f}M")
        total = sum(n for _, n in rows)
        print(f"  {'total':<58} {total / 1e9:>9.3f}B  (budget {float(cfg['param_budget']) / 1e9:.0f}B)")
        if base is not None:
            print(f"  (--baseline loads a second, comparison-only LLM: +{sum(p.numel() for p in base.parameters()) / 1e9:.3f}B)")
        if device.type == "cuda":
            print(f"  GPU memory after loading: {torch.cuda.memory_allocated() / 1e9:.1f} GB "
                  f"({torch.cuda.get_device_name(device)})")
    if not files:
        print("Enter: record (Enter again to stop) | <path>: use an audio file | /reset | /quit")

    turn = 0
    while True:
        if args.wav:
            if not files:
                break
            src = files.pop(0)
            print(f"\n[{src}]")
        else:
            try:
                src = input("\n> ").strip()
            except EOFError:
                break
            if src == "/quit":
                break
            if src == "/reset":
                history.clear()
                print("  (conversation cleared)")
                continue
        if src:
            if not Path(src).exists():
                print(f"  no such file: {src}")
                continue
            wav = read_audio(src, sr)
        else:
            wav = record(sr, args.input_device)
        if len(wav) < 0.3 * sr:
            print("  (too short, try again)")
            continue
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        timer = Timer(device)
        lat = {}
        if args.trace:
            print(f"  [1 audio]    {len(wav)} samples = {len(wav) / sr:.2f} s at {sr} Hz, mono float32")

        if asr is not None:
            kw = {} if args.language == "auto" else {"generate_kwargs": {"language": args.language,
                                                                          "task": "transcribe"}}
            text = asr({"raw": wav, "sampling_rate": sr}, **kw)["text"].strip()
            lat["speech recognition"] = timer.lap()
            print(f'  heard: "{text}"' + (f"   [2 ASR {lat['speech recognition'] * 1e3:.0f} ms]" if args.trace else ""))
            if args.edit and not args.wav:
                text = input("  Enter to accept, or type a correction: ").strip() or text
        else:
            text = input("  what did you say? ").strip() if not args.wav else ""
        if not text:
            print("  (no words recognized, try again)")
            continue

        timer.lap()   # exclude typing time
        feats = featurize(wav)
        lat["emotion model"] = timer.lap()
        top = sorted(feats["probs"].items(), key=lambda kv: -kv[1])[:3]
        if args.trace:
            f = feats["frames"]
            print(f"  [3 emotion]  emotion2vec+ frames {f[1]}x{f[2]} (50 Hz) -> pooled z {tuple(feats['z'].shape)} "
                  f"-> projector h {tuple(feats['h'].shape)} -> 7-way softmax   [{lat['emotion model'] * 1e3:.0f} ms]")
            print(f"  OUTPUT 1     predicted emotion: {feats['emotion']} ({feats['probs'][feats['emotion']]:.2f})")
        print("  emotion model hears: " + "   ".join(f"{e} {bar(p, 10)} {p:.2f}" for e, p in top))

        item = {"speaker": args.speaker, "text": text, "context": list(history),
                "emotion_feat": feats[cfg["emotion"]["feature"]]}
        batch = collate([item])
        if args.trace:
            with torch.no_grad():
                soft = policy.projector(item["emotion_feat"][None].to(device))
            print(f"  [4 prompt]   {cfg['emotion']['feature']} -> {soft.shape[1]} soft tokens x {soft.shape[2]} "
                  f"(LLM embedding size), placed in a {batch['input_ids'].shape[1]}-token chat prompt "
                  f"with {len(item['context'][-cfg['data']['context_turns']:])} context turns")
        timer.lap()
        ids, mask, texts = sample_completions(policy, batch, **gen)
        reply = texts[0]
        lat["reply generation"] = timer.lap()
        if args.trace:
            n = int(mask.sum())
            print(f"  [5 LLM]      {n} tokens in {lat['reply generation'] * 1e3:.0f} ms "
                  f"({n / lat['reply generation']:.0f} tokens/s, {'sampled' if args.sample else 'greedy'})")
        print(f"\n  assistant: {reply}" + ("   <- OUTPUT 2" if args.trace else ""))
        if args.trace:
            total = sum(lat.values())
            mem = (f", peak GPU memory {torch.cuda.max_memory_allocated(device) / 1e9:.1f} GB"
                   if device.type == "cuda" else "")
            ram = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6   # KB on Linux
            print("  latency: " + " + ".join(f"{k} {v * 1e3:.0f} ms" for k, v in lat.items())
                  + f" = {total:.2f} s after the speaker stops{mem}, peak RAM {ram:.1f} GB")
        out = {"turn": turn, "audio": src or None, "text": text, "emotion": feats["emotion"],
               "emotion_probs": feats["probs"], "reply": reply,
               "latency_s": {k: round(v, 3) for k, v in lat.items()}}
        if base is not None:
            b = baseline_texts(base, policy.tokenizer, [no_emotion_messages(item, cfg)], 1, max_new)[0]
            print(f"  no-emotion baseline: {b}")
            out["baseline_reply"] = b
        if args.save_audio and not src:
            import soundfile as sf
            out["audio"] = str(log.with_name(f"{log.stem}_turn{turn}.wav"))
            sf.write(out["audio"], wav, sr)
        with open(log, "a") as f:
            f.write(json.dumps(out) + "\n")
        history += [{"speaker": args.speaker, "text": text}, {"speaker": "Assistant", "text": reply}]
        turn += 1
    print(f"\nsession saved to {log}" if turn else "")


if __name__ == "__main__":
    sys.exit(main())
