"""Validated lookup for preserved native-recording source tracks."""
from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

_RECORDING_RE = re.compile(r"native_recording_([0-9a-f]{32})\.wav\Z")


@dataclass(frozen=True)
class RecordingSources:
    recording_id: str
    microphone_path: Path | None
    system_path: Path | None
    microphone_offset: float
    system_offset: float
    sample_rate: int


def _safe_child(directory: Path, value: str | None) -> Path | None:
    if not value:
        return None
    candidate = (directory / value).resolve()
    root = directory.resolve()
    if not candidate.is_relative_to(root) or not candidate.is_file():
        return None
    return candidate


def find_recording_sources(paths: Any, mixed_audio: Path) -> RecordingSources | None:
    source = Path(mixed_audio).resolve()
    match = _RECORDING_RE.fullmatch(source.name)
    if not match:
        return None
    recording_id = match.group(1)
    directory = Path(paths.results) / "recording_sources" / recording_id
    manifest = directory / "recording.json"
    try:
        payload = json.loads(manifest.read_text(encoding="utf-8"))
        if int(payload.get("schema_version", 0)) != 1 or payload.get("recording_id") != recording_id:
            return None
        mic = _safe_child(directory, payload.get("microphone_file"))
        system = _safe_child(directory, payload.get("system_file"))
        # Never silently analyze only half of an originally two-track meeting.
        if ((payload.get("microphone_file") and mic is None)
                or (payload.get("system_file") and system is None)):
            return None
        if mic is None and system is None:
            return None
        mic_offset = float(payload.get("microphone_offset") or 0.0)
        system_offset = float(payload.get("system_offset") or 0.0)
        if not all(math.isfinite(value) and value >= 0 for value in (mic_offset, system_offset)):
            return None
        return RecordingSources(
            recording_id=recording_id,
            microphone_path=mic,
            system_path=system,
            microphone_offset=mic_offset,
            system_offset=system_offset,
            sample_rate=max(1, int(payload.get("sample_rate") or 48000)),
        )
    except (OSError, ValueError, TypeError, AttributeError):
        return None
