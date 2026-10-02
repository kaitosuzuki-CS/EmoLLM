import math

import torch

from emollm.grpo import compose_rewards, group_advantages
from emollm.judge import EMOTIONS, VALENCE_FLIP, counterfactual
from emollm.model import completion_mask_from_ids

RCFG = {"judge_weight": 1.0, "contrast_weight": 0.5, "contrast_label": "valence",
        "pmi_weight": 0.5, "pmi_clip": 1.0}


def test_compose_rewards_components():
    total, parts = compose_rewards([10, 7, 7, None], [1, 7, None, 4],
                                   torch.tensor([0.2, -3.0, 0.0, 0.5]), RCFG)
    assert torch.allclose(total[:3], torch.tensor([1.6, 2 / 3 - 0.5, 2 / 3]), atol=1e-6)
    assert math.isnan(total[3])                                     # missing true score
    assert parts["contrast"].tolist() == [1.0, 0.0, 0.0, 0.0]       # missing cf score -> 0
    assert parts["pmi_bonus"].tolist()[1] == -1.0                   # clipped


def test_zero_weights_equal_plain_judge_reward():
    scores = [10, 7, 7, None]
    plain = group_advantages(scores, 4)
    composed = group_advantages(compose_rewards(scores, None, None, {"judge_weight": 1.0})[0], 4)
    assert torch.allclose(plain[0], composed[0]) and torch.allclose(plain[1], composed[1])


def test_group_advantages():
    rewards, adv, zero_std = group_advantages([10, 1, 5, None, 7, 7, 7, 7], 4)
    assert adv[3] == 0 and torch.all(adv[4:] == 0)           # missing value / constant group
    assert zero_std == 0.5
    assert abs(float(adv[:4].mean())) < 1e-5 and adv[0] > 0 > adv[1]


def test_counterfactual():
    for e in EMOTIONS:
        assert counterfactual("id", e, "valence") == VALENCE_FLIP[e] != e
        cf = counterfactual("id", e, "random", seed=1)
        assert cf != e and cf == counterfactual("id", e, "random", seed=1)  # stable


def test_completion_mask_stops_after_first_eos():
    ids = torch.tensor([[5, 6, 9, 0, 0], [9, 1, 1, 1, 1], [1, 2, 3, 4, 5]])
    mask = completion_mask_from_ids(ids, {9})
    assert mask.tolist() == [[1, 1, 1, 0, 0], [1, 0, 0, 0, 0], [1, 1, 1, 1, 1]]
