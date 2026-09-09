"""Small allowlisted preference store. Credentials and voiceprints are never stored here."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .paths import AppPaths


class SettingsStore:
    ALLOWED = {
        "theme",
        "mode",
        "local_model",
        "api_model",
        "language",
        "output_type",
        "microphone_key",
        "speaker_key",
        "diarization_enabled",
        "self_speaker_name",
        "expected_speakers",
        "clustering_threshold",
    }

    def __init__(self, paths: AppPaths):
        self.path: Path = paths.config / "preferences.json"
        self.values: dict[str, Any] = {}
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                self.values = {k: v for k, v in data.items() if k in self.ALLOWED}
        except (OSError, ValueError):
            pass

    def get(self, key: str, default: Any = None) -> Any:
        return self.values.get(key, default)

    def set(self, key: str, value: Any) -> None:
        if key not in self.ALLOWED:
            raise ValueError("Ce paramètre ne peut pas être mémorisé.")
        self.values[key] = value
        self.save()

    def save(self) -> None:
        temporary = self.path.with_suffix(".tmp")
        temporary.write_text(json.dumps(self.values, ensure_ascii=False, indent=2), encoding="utf-8")
        temporary.replace(self.path)
