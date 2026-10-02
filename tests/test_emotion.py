import numpy as np
import torch

from emollm.data import clean_text
from emollm.pairwise import win_rate
from emotion.data import ClassBalancedBatchSampler, LengthBucketedBatchSampler, collate_waveforms
from emotion.losses import SupConLoss, VarianceCovarianceLoss
from emotion.model import AttentionPool, MeanPool


def test_clean_text_repairs_cp1252_punctuation():
    assert clean_text("company\x92s   transition\x85") == "company's transition..."


def test_collate_waveforms_padding_mask():
    batch = collate_waveforms([{"wav": torch.ones(3), "label": 0, "index": 0},
                               {"wav": torch.ones(5), "label": 1, "index": 1}])
    assert batch["wav"].shape == (2, 5)
    assert batch["padding_mask"].tolist() == [[False] * 3 + [True] * 2, [False] * 5]


def test_class_balanced_sampler():
    labels = [0] * 50 + [1] * 5 + [2] * 20
    batches = list(ClassBalancedBatchSampler(labels, classes_per_batch=3, samples_per_class=4))
    assert len(batches) == len(labels) // 12
    for b in batches:
        assert sorted(np.bincount(np.array(labels)[b], minlength=3)) == [4, 4, 4]


def test_length_bucketed_sampler_covers_and_groups():
    durations = np.random.default_rng(0).uniform(0.5, 10, size=400)
    batches = list(LengthBucketedBatchSampler(durations, batch_size=20, drop_last=True, seed=0))
    assert sorted(i for b in batches for i in b) == list(range(400))
    spread = np.mean([np.ptp(durations[b]) for b in batches])
    assert spread < np.ptp(durations) / 4   # clips in a batch have similar lengths


def test_attention_pool_starts_as_mean_pool():
    x, pad = torch.randn(3, 10, 768), torch.zeros(3, 10, dtype=torch.bool)
    pad[0, 6:] = True
    assert torch.allclose(AttentionPool(768)(x, pad), MeanPool()(x, pad), atol=1e-6)


def test_variance_covariance_matches_formula():
    torch.manual_seed(0)
    z = torch.randn(64, 16) * torch.rand(16) * 2
    v, c = VarianceCovarianceLoss(gamma=1.0, eps=1e-4)(z)
    zc = z - z.mean(0)
    C = zc.T @ zc / (len(z) - 1)
    std = torch.sqrt(torch.diagonal(C) + 1e-4)
    assert torch.isclose(v, torch.relu(1 - std).mean())
    assert torch.isclose(c, (C - torch.diag(torch.diagonal(C))).pow(2).sum() / 16)


def test_supcon_prefers_clustered_classes():
    labels = torch.tensor([0, 0, 1, 1])
    clustered = torch.tensor([[1.0, 0], [0.9, 0.1], [0, 1.0], [0.1, 0.9]])
    mixed = torch.tensor([[1.0, 0], [0, 1.0], [0.9, 0.1], [0.1, 0.9]])
    loss = SupConLoss(temperature=0.1)
    assert loss(clustered, labels) < loss(mixed, labels)


def test_win_rate():
    w = win_rate(["a", "a", "b", "tie"], n_boot=200)
    assert (w["wins"], w["losses"], w["ties"]) == (2, 1, 1) and w["win_rate"] == 0.625
    assert w["ci95"][0] <= 0.625 <= w["ci95"][1]
