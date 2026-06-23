"""Tests for the paraphrase cache key (no API calls)."""

from nzphish.data.paraphrase import ParaphraseConfig, _key


def test_cache_key_deterministic():
    cfg = ParaphraseConfig(n_paraphrases=4, temperature=0.9)
    a = _key("hello", cfg, seed=42)
    b = _key("hello", cfg, seed=42)
    assert a == b


def test_cache_key_changes_with_seed():
    cfg = ParaphraseConfig(n_paraphrases=4, temperature=0.9)
    assert _key("hello", cfg, seed=1) != _key("hello", cfg, seed=2)


def test_cache_key_changes_with_n_paraphrases():
    a = _key("hello", ParaphraseConfig(n_paraphrases=3), seed=42)
    b = _key("hello", ParaphraseConfig(n_paraphrases=4), seed=42)
    assert a != b


def test_cache_key_changes_with_text():
    cfg = ParaphraseConfig()
    assert _key("hello", cfg, seed=42) != _key("world", cfg, seed=42)
