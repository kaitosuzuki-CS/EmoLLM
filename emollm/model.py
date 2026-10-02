"""Emotion soft prompts for a LoRA-tuned causal LM.

The frozen emotion model's latent (cached per utterance) is projected to K vectors in the
LLM's embedding space. They replace K placeholder tokens in the prompt, so the LLM sees
    [system, context, transcript, ..., e_1 ... e_K, ...] -> response
Only the projector and the LoRA adapters train.
"""
import json
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

PLACEHOLDER_TOKEN = "<|fim_pad|>"   # any existing token works; its embedding is overwritten


class EmotionPromptProjector(nn.Module):
    """emotion latent (in_dim) -> K soft-prompt tokens (K, llm_dim).
    LayerNorm -> Linear(hidden) -> GELU -> Linear(K * llm_dim), plus a learned per-slot
    offset; each token is then normalized and scaled to the LLM's typical token-embedding
    RMS, so the soft tokens start in-distribution."""

    def __init__(self, in_dim: int, hidden: int, k: int, llm_dim: int, embed_rms: float):
        super().__init__()
        self.in_dim, self.hidden, self.k, self.llm_dim = in_dim, hidden, k, llm_dim
        self.net = nn.Sequential(nn.LayerNorm(in_dim), nn.Linear(in_dim, hidden), nn.GELU(),
                                 nn.Linear(hidden, k * llm_dim))
        self.slot = nn.Parameter(torch.randn(k, llm_dim) * 0.02)
        self.norm = nn.LayerNorm(llm_dim, elementwise_affine=False)
        self.scale = nn.Parameter(torch.tensor(float(embed_rms)))

    def forward(self, e: torch.Tensor) -> torch.Tensor:
        x = self.net(e).view(-1, self.k, self.llm_dim) + self.slot
        return self.norm(x) * self.scale

    def spec(self) -> dict:
        return {"in_dim": self.in_dim, "hidden": self.hidden, "k": self.k, "llm_dim": self.llm_dim}


def position_ids_from_mask(attention_mask: torch.Tensor) -> torch.Tensor:
    return (attention_mask.long().cumsum(-1) - 1).clamp(min=0)


class EmotionLLM(nn.Module):
    def __init__(self, llm, projector: EmotionPromptProjector, tokenizer):
        super().__init__()
        self.llm = llm
        self.projector = projector
        self.tokenizer = tokenizer
        self.placeholder_id = tokenizer.convert_tokens_to_ids(PLACEHOLDER_TOKEN)
        self.k = projector.k

    @property
    def device(self):
        return self.llm.get_input_embeddings().weight.device

    def embed(self, input_ids, emotion_mask, emotion, projector=None):
        emb = self.llm.get_input_embeddings()(input_ids)
        soft = (projector or self.projector)(emotion.to(emb.device, torch.float32)).to(emb.dtype)
        # rows each hold exactly K placeholders, in order, so a masked scatter fills them
        return emb.masked_scatter(emotion_mask.unsqueeze(-1), soft.reshape(-1, emb.shape[-1]))

    def forward(self, input_ids, attention_mask, emotion_mask, emotion, labels=None):
        d = self.device
        return self.llm(inputs_embeds=self.embed(input_ids.to(d), emotion_mask.to(d), emotion),
                        attention_mask=attention_mask.to(d),
                        position_ids=position_ids_from_mask(attention_mask.to(d)),
                        labels=None if labels is None else labels.to(d), use_cache=False)

    @torch.no_grad()
    def generate(self, batch: dict, **gen_kwargs) -> torch.Tensor:
        """Left-padded prompt batch -> generated token ids (B, new_tokens)."""
        d = self.device
        embeds = self.embed(batch["input_ids"].to(d), batch["emotion_mask"].to(d), batch["emotion"])
        return self.llm.generate(inputs_embeds=embeds, attention_mask=batch["attention_mask"].to(d),
                                 pad_token_id=self.tokenizer.pad_token_id, **gen_kwargs)

    def completion_logprobs(self, batch: dict, completion_ids, completion_mask, projector=None,
                            temperature: float = 1.0):
        """Per-token log-probs of `completion_ids` (B, L) following the left-padded prompts,
        from logits / temperature (the distribution the completions were sampled from).
        Only the last L+1 positions go through the LM head."""
        d = self.device
        prompt = self.embed(batch["input_ids"].to(d), batch["emotion_mask"].to(d),
                            batch["emotion"], projector)
        comp = completion_ids.to(d)
        embeds = torch.cat([prompt, self.llm.get_input_embeddings()(comp)], dim=1)
        mask = torch.cat([batch["attention_mask"].to(d), completion_mask.to(d).long()], dim=1)
        L = comp.shape[1]
        logits = self.llm(inputs_embeds=embeds, attention_mask=mask,
                          position_ids=position_ids_from_mask(mask), use_cache=False,
                          logits_to_keep=L + 1).logits[:, :-1].float() / temperature
        return torch.gather(F.log_softmax(logits, -1), 2, comp.unsqueeze(-1)).squeeze(-1)


def completion_mask_from_ids(ids: torch.Tensor, eos_ids: set[int]) -> torch.Tensor:
    """1 up to and including the first EOS token, 0 after (generate pads after EOS)."""
    is_eos = torch.zeros_like(ids, dtype=torch.bool)
    for e in eos_ids:
        is_eos |= ids == e
    after = (is_eos.long().cumsum(1) - is_eos.long()) > 0   # strictly after the first EOS
    return (~after).long()


# ---------------------------------------------------------------- loading / saving

def load_tokenizer(name: str):
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok


def load_llm(name: str, merge_adapters=(), device="cuda"):
    """Base causal LM in bf16 with the given LoRA adapters merged in (in order)."""
    from peft import PeftModel
    from transformers import AutoModelForCausalLM
    llm = AutoModelForCausalLM.from_pretrained(name, dtype=torch.bfloat16).to(device)
    for path in merge_adapters:
        llm = PeftModel.from_pretrained(llm, path).merge_and_unload()
    return llm


def add_lora(llm, lora_cfg: dict, dropout: float | None = None):
    from peft import LoraConfig, get_peft_model
    cfg = LoraConfig(r=lora_cfg["r"], lora_alpha=lora_cfg["alpha"],
                     lora_dropout=lora_cfg["dropout"] if dropout is None else dropout,
                     target_modules=list(lora_cfg["target_modules"]), task_type="CAUSAL_LM")
    return get_peft_model(llm, cfg)


def embedding_rms(llm) -> float:
    w = llm.get_input_embeddings().weight.detach()
    return float(w.float().pow(2).mean(-1).sqrt().median())


def new_projector(cfg: dict, in_dim: int, llm) -> EmotionPromptProjector:
    p = cfg["policy"]
    return EmotionPromptProjector(in_dim, p["projector_hidden"], p["num_emotion_tokens"],
                                  llm.config.hidden_size, embedding_rms(llm))


def load_projector(path) -> EmotionPromptProjector:
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    proj = EmotionPromptProjector(**ckpt["spec"], embed_rms=1.0)
    proj.load_state_dict(ckpt["state_dict"])
    return proj


def save_policy(run_dir: Path, policy: EmotionLLM, base_model: str, merged_adapters: list,
                tag: str = "", extra: dict | None = None):
    """Saves <run_dir>/<tag>adapter (LoRA), <tag>projector.pt and <tag>policy.json, which
    lists everything needed to rebuild the policy: base model, adapters to merge first
    (e.g. SFT under GRPO), then this adapter."""
    run_dir = Path(run_dir)
    adapter_dir = run_dir / f"{tag}adapter"
    policy.llm.save_pretrained(adapter_dir)
    torch.save({"state_dict": policy.projector.state_dict(), "spec": policy.projector.spec()},
               run_dir / f"{tag}projector.pt")
    meta = {"base_model": base_model, "merged_adapters": [str(a) for a in merged_adapters],
            "adapter": str(adapter_dir), "projector": str(run_dir / f"{tag}projector.pt"),
            **(extra or {})}
    (run_dir / f"{tag}policy.json").write_text(json.dumps(meta, indent=2))
    return meta


def resolve_saved_path(path: str, run_dir: Path) -> Path:
    """Paths in policy.json are relative to the directory training ran from. Resolve them
    from any working directory by trying the run directory's ancestors."""
    p = Path(path)
    if p.is_absolute() or p.exists():
        return p
    for base in run_dir.resolve().parents:
        if (base / p).exists():
            return base / p
    raise FileNotFoundError(f"{path} (from {run_dir}/policy.json) not found")


def load_policy(run_dir, device="cuda", tag: str = "") -> tuple[EmotionLLM, dict]:
    """Rebuilds a saved policy for inference: every adapter merged into the base."""
    run_dir = Path(run_dir)
    meta = json.loads((run_dir / f"{tag}policy.json").read_text())
    resolve = lambda p: str(resolve_saved_path(p, run_dir))  # noqa: E731
    meta["merged_adapters"] = [resolve(p) for p in meta["merged_adapters"]]
    meta["adapter"], meta["projector"] = resolve(meta["adapter"]), resolve(meta["projector"])
    tok = load_tokenizer(meta["base_model"])
    llm = load_llm(meta["base_model"], meta["merged_adapters"] + [meta["adapter"]], device)
    policy = EmotionLLM(llm, load_projector(meta["projector"]).to(device), tok)
    return policy.eval(), meta


def pipeline_params(emotion_params: int | None, policy: EmotionLLM) -> dict:
    """Parameter counts of the whole pipeline: frozen emotion model (recorded in the
    feature cache by scripts/llm/extract_features.py), soft-prompt projector,
    LLM (base + LoRA)."""
    if emotion_params is None:
        raise ValueError("feature cache has no emotion_params; re-run "
                         "scripts/llm/extract_features.py")
    emotion = emotion_params
    llm_total = sum(p.numel() for p in policy.llm.parameters())
    llm_trainable = sum(p.numel() for p in policy.llm.parameters() if p.requires_grad)
    proj = sum(p.numel() for p in policy.projector.parameters())
    return {"emotion_model": emotion, "projector": proj, "llm_total": llm_total,
            "lora_trainable": llm_trainable, "trainable": llm_trainable + proj,
            "total": (emotion or 0) + proj + llm_total}


def check_budget(params: dict, budget: float):
    total = params["total"]
    print(f"pipeline params: emotion model {params['emotion_model'] / 1e6:.1f}M + projector "
          f"{params['projector'] / 1e6:.1f}M + LLM {params['llm_total'] / 1e9:.3f}B "
          f"(LoRA {params['lora_trainable'] / 1e6:.1f}M trainable) = {total / 1e9:.3f}B "
          f"(budget {budget / 1e9:.1f}B)")
    if total >= budget:
        raise ValueError(f"pipeline has {total / 1e9:.2f}B params, over the {budget / 1e9:.1f}B budget")
