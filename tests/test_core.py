from __future__ import annotations

import logging

import pytest

from transcripteur_whisper.core.logging import RedactingFormatter
from transcripteur_whisper.core.paths import AppPaths
from transcripteur_whisper.core.settings import SettingsStore


def test_user_paths_and_assets(tmp_path):
    paths = AppPaths.create(tmp_path)
    assert all(p.is_dir() for p in (paths.config, paths.models, paths.logs, paths.temp, paths.results))
    assert paths.asset_path("icon.ico").is_file()


def test_settings_never_persist_credentials(tmp_path):
    store = SettingsStore(AppPaths.create(tmp_path))
    store.set("theme", "dark")
    with pytest.raises(ValueError):
        store.set("api_key", "test-value")
    assert "test-value" not in store.path.read_text()
    assert SettingsStore(AppPaths.create(tmp_path)).get("theme") == "dark"


def test_log_redaction_including_exception():
    formatter = RedactingFormatter("%(message)s")
    token = "sk-" + "not-a-real-credential"
    try:
        raise RuntimeError(token)
    except RuntimeError:
        import sys

        record = logging.LogRecord("test", logging.ERROR, "", 0, "Request failed: %s", (token,), sys.exc_info())
    output = formatter.format(record)
    assert token not in output
    assert "RuntimeError" in output
