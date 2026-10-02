import pytest

from emotion.config import load_config


def write(tmp_path, name, text):
    p = tmp_path / name
    p.write_text(text)
    return p


def test_base_inheritance_and_overrides(tmp_path):
    write(tmp_path, "base.yaml", "a: 1\nnested: {x: 1, y: 2}\n")
    child = write(tmp_path, "child.yaml", "base: base.yaml\nnested: {y: 3}\nb: hi\n")
    cfg = load_config(str(child), ["nested.x=10", "c=[1, 2]"])
    assert cfg == {"a": 1, "nested": {"x": 10, "y": 3}, "b": "hi", "c": [1, 2]}


def test_interpolation_resolves_after_overrides(tmp_path):
    p = write(tmp_path, "c.yaml", "root: data\npaths:\n  wav: ${root}/wav\n  nested: ${paths.wav}/train\n")
    assert load_config(str(p))["paths"] == {"wav": "data/wav", "nested": "data/wav/train"}
    assert load_config(str(p), ["root=/mnt/meld"])["paths"]["nested"] == "/mnt/meld/wav/train"


def test_missing_reference_raises(tmp_path):
    p = write(tmp_path, "c.yaml", "x: ${nope}\n")
    with pytest.raises(KeyError):
        load_config(str(p))


def test_repo_configs_resolve():
    for name in ["base", "classification", "supcon", "finetune", "finetune_supcon"]:
        cfg = load_config(f"configs/emotion/{name}.yaml", ["data_root=/x"])
        assert cfg["data"]["splits"]["dev"]["manifest"] == "/x/wav16k/dev_manifest.csv"
    llm = load_config("configs/llm.yaml", ["data_root=/x"])
    assert llm["pseudo"]["out_dir"] == "/x/llm/pseudo"
    assert isinstance(llm["param_budget"], float)  # 6.0e+9, not the string "6.0e9"
