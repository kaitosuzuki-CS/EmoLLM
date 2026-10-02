"""YAML configs with `base:` inheritance, CLI overrides and `${key}` interpolation.

    cfg = load_config("configs/emotion/finetune.yaml", ["train.lr=1e-3", "data_root=/data/meld"])

* A top-level `base: other.yaml` (relative to the file) is loaded first and recursively
  overridden by the file's own keys.
* Overrides are `dotted.key=value` strings; values are parsed as YAML.
* String values may reference other keys as `${dotted.key}` (e.g. `${data_root}/wav16k`).
  References are resolved after overrides, so overriding `data_root` moves every path.
"""
import copy
import re
from pathlib import Path

import yaml

_REF = re.compile(r"\$\{([^}]+)\}")


def _merge(base: dict, override: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _merge(out[k], v)
        else:
            out[k] = v
    return out


def _load_raw(path: Path) -> dict:
    with open(path) as f:
        cfg = yaml.safe_load(f) or {}
    if "base" in cfg:
        cfg = _merge(_load_raw(path.parent / cfg.pop("base")), cfg)
    return cfg


def _lookup(cfg: dict, dotted: str):
    node = cfg
    for part in dotted.split("."):
        if not isinstance(node, dict) or part not in node:
            raise KeyError(f"config reference ${{{dotted}}} not found")
        node = node[part]
    return node


def _resolve(node, root: dict, depth: int = 0):
    if depth > 10:
        raise ValueError("config references nest too deeply (cycle?)")
    if isinstance(node, dict):
        return {k: _resolve(v, root, depth) for k, v in node.items()}
    if isinstance(node, list):
        return [_resolve(v, root, depth) for v in node]
    if isinstance(node, str) and _REF.search(node):
        out = _REF.sub(lambda m: str(_lookup(root, m.group(1))), node)
        return _resolve(out, root, depth + 1)
    return node


def apply_overrides(cfg: dict, overrides: list[str]) -> dict:
    for item in overrides:
        key, value = item.split("=", 1)
        node = cfg
        *parents, leaf = key.split(".")
        for p in parents:
            node = node.setdefault(p, {})
        node[leaf] = yaml.safe_load(value)
    return cfg


def load_config(path: str, overrides: list[str] | None = None) -> dict:
    cfg = apply_overrides(_load_raw(Path(path)), overrides or [])
    return _resolve(cfg, cfg)
