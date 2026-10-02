import torch
import torch.nn as nn
import torch.nn.functional as F


class SupConLoss(nn.Module):
    """Supervised contrastive loss (Khosla et al., 2020), L_out variant.

    Each sample is one view; positives are the other in-batch samples with the
    same label. Anchors with no positive in the batch are skipped."""

    def __init__(self, temperature: float = 0.1, base_temperature: float | None = None):
        super().__init__()
        self.temperature = temperature
        self.base_temperature = base_temperature or temperature

    def forward(self, features: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        f = F.normalize(features.float(), dim=1)
        logits = f @ f.T / self.temperature
        self_mask = torch.eye(len(f), dtype=torch.bool, device=f.device)
        logits = logits.masked_fill(self_mask, float("-inf"))
        log_prob = logits - torch.logsumexp(logits, dim=1, keepdim=True)

        pos_mask = (labels[:, None] == labels[None, :]) & ~self_mask
        n_pos = pos_mask.sum(1)
        has_pos = n_pos > 0
        if not has_pos.any():
            return features.sum() * 0.0  # keeps the graph; no positives in this batch
        mean_log_prob_pos = log_prob.masked_fill(~pos_mask, 0.0).sum(1)[has_pos] / n_pos[has_pos]
        return -(self.temperature / self.base_temperature) * mean_log_prob_pos.mean()


class VarianceCovarianceLoss(nn.Module):
    """VICReg variance and covariance terms (Bardes et al., 2022) on a batch z (N, d):

        L_var = 1/d * sum_j max(0, gamma - std(z_j)),   std = sqrt(Var + eps)
        L_cov = 1/d * sum_{i != j} C(z)_{ij}^2,          C = covariance over the batch

    Returns (L_var, L_cov)."""

    def __init__(self, gamma: float = 1.0, eps: float = 1e-4):
        super().__init__()
        self.gamma = gamma
        self.eps = eps

    def forward(self, z: torch.Tensor):
        z = z.float()
        n, d = z.shape
        z = z - z.mean(dim=0)
        std = torch.sqrt(z.var(dim=0) + self.eps)
        var_loss = F.relu(self.gamma - std).mean()
        cov = (z.T @ z) / (n - 1)
        off_diag = cov - torch.diag(torch.diagonal(cov))
        cov_loss = off_diag.pow(2).sum() / d
        return var_loss, cov_loss


class SupConVarCovLoss(nn.Module):
    """supcon_weight * SupCon(normalize(h)) + var_weight * L_var(h) + cov_weight * L_cov(h).

    SupCon works on L2-normalized features; the variance/covariance terms act on the
    raw projector output h, as in VICReg (after normalization each dim's std is at
    most ~1/sqrt(d), so a gamma near 1 would be unreachable). The unweighted parts
    of the last call are kept in `self.parts` (detached) for logging."""

    def __init__(self, temperature: float, base_temperature: float | None = None,
                 supcon_weight: float = 1.0, var_weight: float = 0.0, cov_weight: float = 0.0,
                 gamma: float = 1.0, eps: float = 1e-4):
        super().__init__()
        self.supcon = SupConLoss(temperature, base_temperature)
        self.varcov = VarianceCovarianceLoss(gamma, eps)
        self.supcon_weight, self.var_weight, self.cov_weight = supcon_weight, var_weight, cov_weight
        self.parts = {}

    def forward(self, features: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        supcon = self.supcon(features, labels)
        var, cov = self.varcov(features)
        self.parts = {"supcon": supcon.detach(), "var": var.detach(), "cov": cov.detach()}
        return self.supcon_weight * supcon + self.var_weight * var + self.cov_weight * cov

    @classmethod
    def from_config(cls, loss_cfg: dict):
        return cls(loss_cfg["temperature"], loss_cfg["base_temperature"],
                   loss_cfg.get("supcon_weight", 1.0), loss_cfg.get("var_weight", 0.0),
                   loss_cfg.get("cov_weight", 0.0), loss_cfg.get("var_gamma", 1.0),
                   loss_cfg.get("var_eps", 1e-4))
