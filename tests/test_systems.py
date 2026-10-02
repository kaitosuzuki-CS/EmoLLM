import pytest

from emollm.systems import parse_system


def test_baselines():
    s = parse_system("baseline:none")
    assert (s.kind, s.variant, s.name) == ("baseline", "none", "baseline_none")
    assert s.reply_cache("test").name == "replies_test_none.jsonl"
    with pytest.raises(ValueError):
        parse_system("baseline:shuffled")


def test_policy_and_shuffled(tmp_path):
    run = tmp_path / "llm_grpo_1"
    run.mkdir()
    (run / "policy.json").write_text("{}")
    s = parse_system(str(run))
    assert (s.kind, s.name) == ("policy", "llm_grpo_1")
    assert s.reply_cache("dev") == run / "replies_dev.jsonl"
    sh = parse_system(f"{run}:shuffled")
    assert sh.name == "llm_grpo_1_shuffled"
    assert sh.reply_cache("test") == run / "replies_test_shuffled.jsonl"
    with pytest.raises(ValueError):
        parse_system(f"{run}:other")


def test_reply_file_and_errors(tmp_path):
    f = tmp_path / "replies_mine.jsonl"
    f.write_text('{"id": "dia0_utt0", "reply": "hi"}\n')
    s = parse_system(str(f))
    assert (s.kind, s.name, s.reply_cache("test")) == ("file", "mine", f)
    with pytest.raises(FileNotFoundError):
        parse_system(str(tmp_path / "not_a_run"))
