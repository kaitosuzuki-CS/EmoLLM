"""MELD transcripts, dialogue context, cached emotion features, and pseudo labels."""
import json
import re
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import Dataset

_ASCII = {"’": "'", "‘": "'", "“": '"', "”": '"', "—": " - ",
          "–": " - ", "…": "...", " ": " "}


def clean_text(s: str) -> str:
    """MELD transcripts store Windows-1252 punctuation as C1 control characters
    (e.g. U+0092 for a right quote). Decode those, then normalize to plain ASCII
    punctuation and collapse whitespace."""
    s = "".join(bytes([ord(c)]).decode("cp1252", errors="ignore") if 0x80 <= ord(c) <= 0x9F
                else c for c in str(s))
    for k, v in _ASCII.items():
        s = s.replace(k, v)
    return re.sub(r"\s+", " ", s).strip()


def load_transcripts(csv_path: str) -> pd.DataFrame:
    """One row per utterance: id (diaD_uttU, the name of the utterance's own audio clip),
    dialogue / utterance ids, speaker, cleaned text and ground-truth emotion."""
    df = pd.read_csv(Path(csv_path).expanduser())
    out = pd.DataFrame({
        "id": "dia" + df["Dialogue_ID"].astype(str) + "_utt" + df["Utterance_ID"].astype(str),
        "dialogue_id": df["Dialogue_ID"], "utterance_id": df["Utterance_ID"],
        "speaker": df["Speaker"].map(clean_text), "text": df["Utterance"].map(clean_text),
        "emotion": df["Emotion"],
    })
    return out.sort_values(["dialogue_id", "utterance_id"]).reset_index(drop=True)


def add_context(df: pd.DataFrame) -> pd.DataFrame:
    """`context`: all earlier utterances of the same dialogue as [{speaker, text}], in order.
    (Prompts truncate this to the last `data.context_turns` turns.)"""
    contexts = []
    for _, g in df.groupby("dialogue_id", sort=False):
        turns = []
        for r in g.itertuples():
            contexts.append((r.Index, list(turns)))
            turns.append({"speaker": r.speaker, "text": r.text})
    ctx = dict(contexts)
    df = df.copy()
    df["context"] = [ctx[i] for i in df.index]
    return df


def cache_paths(cache_dir: str, split: str) -> dict:
    d = Path(cache_dir).expanduser()
    return {"features": d / f"features_{split}.pt", "utterances": d / f"utterances_{split}.jsonl"}


def read_jsonl(path) -> list[dict]:
    path = Path(path).expanduser()
    if not path.exists():
        return []
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def pseudo_label_path(cfg: dict, split: str) -> Path:
    return Path(cfg["pseudo"]["out_dir"]).expanduser() / f"{split}.jsonl"


class UtteranceDataset(Dataset):
    """Utterances that have audio (hence an emotion feature), with transcript, dialogue
    context, ground-truth and predicted emotion, and optionally the cached pseudo-label
    response. Items are dicts; `emotion_feat` is the cached emotion latent."""

    def __init__(self, cfg: dict, split: str, require_response: bool = False,
                 limit: int | None = None):
        paths = cache_paths(cfg["emotion"]["cache_dir"], split)
        feats = torch.load(paths["features"])
        key = cfg["emotion"]["feature"]
        index = {k: i for i, k in enumerate(feats["ids"])}
        responses = {}
        if require_response:
            responses = {r["id"]: r["response"] for r in read_jsonl(pseudo_label_path(cfg, split))
                         if r.get("response")}
        self.items = []
        for u in read_jsonl(paths["utterances"]):
            if u["id"] not in index or (require_response and u["id"] not in responses):
                continue
            i = index[u["id"]]
            self.items.append({**u, "emotion_feat": feats[key][i].float(),
                               "pred_emotion": feats["pred_emotion"][i],
                               "response": responses.get(u["id"])})
            if limit and len(self.items) >= limit:
                break
        self.feature_dim = feats[key].shape[1]
        # which frozen emotion model produced these features (written by the extractor)
        self.emotion_checkpoint = feats.get("emotion_checkpoint")
        self.emotion_params = feats.get("emotion_params")

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        return self.items[i]
