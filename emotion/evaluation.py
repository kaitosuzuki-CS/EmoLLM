"""Evaluation used during training and by scripts/emotion/evaluate.py: classification
metrics of a head, cosine kNN in the latent space, and a logistic-regression linear probe
(exportable as an nn.Linear)."""
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression

from emotion.data import EMOTIONS
from emotion.model import get_z
from emotion.trainer import classification_metrics, embed, knn_predict, per_class_f1


@torch.no_grad()
def classification_eval(model, loader, source, criterion, device):
    """Mean loss, predictions and labels of a classification head over a loader."""
    model.eval()
    total_loss, preds, labels = 0.0, [], []
    for batch in loader:
        y = batch["label"].to(device)
        logits, _ = model(get_z(batch, source, device))
        total_loss += criterion(logits, y).item() * len(y)
        preds.append(logits.argmax(1).cpu())
        labels.append(y.cpu())
    preds, labels = torch.cat(preds), torch.cat(labels)
    return total_loss / len(labels), preds.numpy(), labels.numpy()


@torch.no_grad()
def knn_eval(model, loaders, source, cfg, device):
    model.eval()
    e = cfg["eval"]
    bank, bank_y = embed(loaders["train_eval"], source, model, device)
    query, query_y = embed(loaders["val"], source, model, device)
    preds = knn_predict(bank, bank_y, query, e["knn_k"], e["knn_temperature"], len(EMOTIONS))
    return classification_metrics(query_y.numpy(), preds.numpy())


def linear_probe(model, loaders, source, cfg, device, save_path=None, meta=None):
    """Logistic regression on frozen, L2-normalized D-dim representations.

    With `save_path`, the probe is saved as an nn.Linear(D, num_classes) state dict
    (`state_dict`), applied as `probe(F.normalize(projector(z), dim=1))`; argmax of its
    logits reproduces the sklearn predictions. `meta` is stored alongside."""
    model.eval()
    fn = lambda z: F.normalize(model(z), dim=1)  # noqa: E731
    x_tr, y_tr = embed(loaders["train_eval"], source, fn, device)
    x_va, y_va = embed(loaders["val"], source, fn, device)
    clf = LogisticRegression(max_iter=2000, C=cfg["eval"]["probe_C"],
                             class_weight=cfg["eval"]["probe_class_weight"])
    clf.fit(x_tr.numpy(), y_tr.numpy())
    preds = clf.predict(x_va.numpy())
    metrics, per_class = classification_metrics(y_va.numpy(), preds), per_class_f1(y_va.numpy(), preds)
    if save_path is not None:
        probe = probe_to_linear(clf, x_tr.shape[1])
        with torch.no_grad():
            torch_preds = probe(x_va).argmax(1).numpy()
        assert (torch_preds == preds).all(), "exported probe disagrees with sklearn"
        torch.save({"state_dict": probe.state_dict(), "in_features": x_tr.shape[1],
                    "num_classes": len(EMOTIONS), "emotions": EMOTIONS,
                    "input": "F.normalize(projector(z), dim=1)", "metrics": metrics,
                    "per_class_f1": per_class, "probe_C": cfg["eval"]["probe_C"],
                    "probe_class_weight": cfg["eval"]["probe_class_weight"], **(meta or {})},
                   save_path)
        print(f"saved linear probe to {save_path}")
    return metrics, per_class


def probe_to_linear(clf: LogisticRegression, in_features: int) -> nn.Linear:
    """Copies a fitted multinomial LogisticRegression into nn.Linear with one output
    per emotion; a class absent from training gets a very negative bias."""
    probe = nn.Linear(in_features, len(EMOTIONS))
    with torch.no_grad():
        probe.weight.zero_()
        probe.bias.fill_(-1e4)
        rows = torch.as_tensor(clf.classes_, dtype=torch.long)
        probe.weight[rows] = torch.as_tensor(clf.coef_, dtype=torch.float32)
        probe.bias[rows] = torch.as_tensor(clf.intercept_, dtype=torch.float32)
    return probe
