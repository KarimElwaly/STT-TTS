"""Offline mode: run from the local cache, fail loudly when it is incomplete."""

import os

import pytest

from voicegw.common import offline
from voicegw.common.config import Settings
from voicegw.common.errors import hint_for


@pytest.fixture
def clean_env(monkeypatch):
    for key in offline._OFFLINE_ENV:
        monkeypatch.delenv(key, raising=False)
    return monkeypatch


# --- enabling --------------------------------------------------------------


def test_enable_sets_every_offline_env_var(clean_env):
    offline.enable()
    for key, value in offline._OFFLINE_ENV.items():
        assert os.environ[key] == value


def test_enable_patches_the_already_imported_hub_constant(clean_env):
    """The env var alone is ignored once huggingface_hub has been imported."""
    from huggingface_hub import constants

    clean_env.setattr(constants, "HF_HUB_OFFLINE", False, raising=False)
    offline.enable()
    assert constants.HF_HUB_OFFLINE is True


def test_is_enabled_reflects_the_env(clean_env):
    assert offline.is_enabled() is False
    offline.enable()
    assert offline.is_enabled() is True


def test_settings_offline_defaults_to_false(monkeypatch):
    # Must not depend on the ambient shell, which may have VOICEGW_OFFLINE set.
    monkeypatch.delenv("VOICEGW_OFFLINE", raising=False)
    assert Settings(_env_file=None).offline is False


def test_settings_reads_offline_from_env(monkeypatch):
    monkeypatch.setenv("VOICEGW_OFFLINE", "1")
    assert Settings().offline is True


# --- settings-backed knobs -------------------------------------------------
# These used to be read with os.environ at import time, so a value set in .env
# (which only pydantic-settings reads) was silently ignored.


def test_tts_step_count_comes_from_settings(monkeypatch):
    from voicegw.common import config
    from voicegw.engines import omnivoice_tts

    monkeypatch.setattr(config, "get_settings", lambda: Settings(tts_num_step=7))
    assert omnivoice_tts.default_num_step() == 7


def test_whisper_model_comes_from_settings(monkeypatch):
    from voicegw.common import config
    from voicegw.engines import faster_whisper_asr

    monkeypatch.setattr(config, "get_settings", lambda: Settings(whisper_model="medium"))
    assert faster_whisper_asr.pinned_model() == "medium"


def test_unset_whisper_model_falls_back_to_the_profile_default(monkeypatch):
    from voicegw.common import config
    from voicegw.engines import faster_whisper_asr

    monkeypatch.setattr(config, "get_settings", lambda: Settings(whisper_model=None))
    assert faster_whisper_asr.pinned_model() is None


def test_asset_list_follows_the_pinned_whisper_model(monkeypatch):
    from voicegw.engines import faster_whisper_asr

    monkeypatch.setattr(faster_whisper_asr, "pinned_model", lambda: "medium")
    repos = {a.key: a.repo_id for a in offline.assets()}
    assert repos["faster-whisper"] == "Systran/faster-whisper-medium"


# --- asset inventory -------------------------------------------------------


def test_assets_cover_both_profiles_and_are_unique():
    keys = [a.key for a in offline.assets()]
    assert {"cohere-asr", "omnivoice", "faster-whisper"} <= set(keys)
    assert len(keys) == len(set(keys))


def test_only_the_cohere_repo_is_gated():
    gated = {a.key for a in offline.assets() if a.gated}
    assert gated == {"cohere-asr"}


def test_every_asset_names_a_real_repo():
    for asset in offline.assets():
        assert "/" in asset.repo_id, asset


@pytest.mark.parametrize(
    "alias, expected",
    [
        ("small", "Systran/faster-whisper-small"),
        ("large-v3-turbo", "mobiuslabsgmbh/faster-whisper-large-v3-turbo"),
        # An explicit repo id must pass through untouched.
        ("Systran/faster-whisper-medium", "Systran/faster-whisper-medium"),
    ],
)
def test_whisper_aliases_resolve_to_repos(alias, expected):
    assert offline.whisper_repo(alias) == expected


def test_unknown_whisper_alias_passes_through():
    assert offline.whisper_repo("not-a-size") == "not-a-size"


# --- cache probing ---------------------------------------------------------


def test_is_cached_false_for_a_nonexistent_repo():
    assert offline.is_cached("voicegw/definitely-not-a-real-repo") is False


def test_missing_reports_uncached_assets(monkeypatch):
    monkeypatch.setattr(offline, "is_cached", lambda repo: False)
    monkeypatch.setattr(offline, "needs_external_audio_tokenizer", lambda: False)
    assert {a.key for a in offline.missing()} == {a.key for a in offline.assets()}


def test_missing_is_empty_when_everything_is_cached(monkeypatch):
    monkeypatch.setattr(offline, "is_cached", lambda repo: True)
    monkeypatch.setattr(offline, "needs_external_audio_tokenizer", lambda: False)
    assert offline.missing() == []


def test_missing_includes_the_tokenizer_only_when_unbundled(monkeypatch):
    monkeypatch.setattr(offline, "is_cached", lambda repo: False)
    monkeypatch.setattr(offline, "needs_external_audio_tokenizer", lambda: True)
    assert offline.AUDIO_TOKENIZER_REPO in {a.repo_id for a in offline.missing()}


# --- diagnostics -----------------------------------------------------------


def test_offline_cache_miss_hint_beats_the_generic_network_hint():
    """LocalEntryNotFoundError mentions disabled traffic; advice must be specific."""
    from huggingface_hub.errors import LocalEntryNotFoundError

    exc = LocalEntryNotFoundError(
        "Cannot find the requested files in the disk cache and outgoing traffic "
        "has been disabled. To enable hf.co look-ups, set HF_HUB_OFFLINE=0."
    )
    assert "voicegw fetch" in hint_for(exc)


def test_hint_matches_on_exception_type_alone():
    class LocalEntryNotFoundError(Exception):
        pass

    assert "voicegw fetch" in hint_for(LocalEntryNotFoundError("no detail"))


def test_a_genuine_network_error_still_gets_the_network_hint():
    hint = hint_for(OSError("Max retries exceeded with url: /api/models"))
    assert "network" in hint
