"""GRPO pieces: reward composition, group-relative advantages and batched log-probs.

reward = judge_weight    * (s_true - 1) / 9
       + contrast_weight * (s_true - s_cf) / 9
       + pmi_weight      * clip(PMI, +-pmi_clip)
where s_true / s_cf are judge scores (1-10) under the true / a counterfactual emotion and
PMI is the mean per-token log p(reply | true emotion) - log p(reply | other-emotion features).
"""
import torch

BATCH_KEYS = ("input_ids", "attention_mask", "emotion_mask", "emotion")


def rows(batch: dict, sl: slice) -> dict:
    return {k: batch[k][sl] for k in BATCH_KEYS}


def _score_tensor(scores: list) -> torch.Tensor:
    return torch.tensor([float("nan") if s is None else float(s) for s in scores])


def compose_rewards(true_scores: list, cf_scores: list | None, pmi: torch.Tensor | None,
                    rcfg: dict) -> tuple[torch.Tensor, dict]:
    """Total reward per sample (NaN where the true-label judge score is missing) and its
    unweighted components for logging."""
    s_true = _score_tensor(true_scores)
    parts = {"judge": (s_true - 1) / 9}
    total = rcfg.get("judge_weight", 1.0) * parts["judge"]
    if cf_scores is not None:
        s_cf = _score_tensor(cf_scores)
        parts["judge_cf"] = (s_cf - 1) / 9
        parts["contrast"] = ((s_true - s_cf) / 9).nan_to_num(0.0)  # missing either -> 0
        total = total + rcfg["contrast_weight"] * parts["contrast"]
    if pmi is not None:
        parts["pmi"] = pmi.float().cpu()
        parts["pmi_bonus"] = parts["pmi"].clamp(-rcfg["pmi_clip"], rcfg["pmi_clip"])
        total = total + rcfg["pmi_weight"] * parts["pmi_bonus"]
    return total, parts


def group_advantages(rewards, group: int):
    """rewards (B*G,) -> rewards, advantages (B*G,), share of zero-variance groups.
    Accepts a list of 1-10 judge scores (None = unparseable; plain judge reward) or a reward
    tensor with NaN for missing values. Missing rewards get their group's mean (zero advantage)."""
    if not torch.is_tensor(rewards):
        rewards = (_score_tensor(rewards) - 1) / 9
    r = rewards.float().view(-1, group)
    mean = torch.nanmean(r, dim=1, keepdim=True).nan_to_num(0.5)
    r = torch.where(r.isnan(), mean.expand_as(r), r)
    std = r.std(dim=1, keepdim=True, unbiased=False)
    adv = torch.where(std > 1e-6, (r - r.mean(1, keepdim=True)) / (std + 1e-4), torch.zeros_like(r))
    return r.flatten(), adv.flatten(), (std.squeeze(1) <= 1e-6).float().mean().item()


@torch.no_grad()
def logprobs(policy, batch, ids, mask, micro: int, temperature: float, projector=None):
    return torch.cat([policy.completion_logprobs(rows(batch, slice(i, i + micro)), ids[i:i + micro],
                                                 mask[i:i + micro], projector, temperature)
                      for i in range(0, ids.shape[0], micro)])
