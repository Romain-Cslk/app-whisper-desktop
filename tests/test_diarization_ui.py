from __future__ import annotations

from types import SimpleNamespace

import pytest
from PySide6.QtWidgets import QMessageBox

from tests.test_desktop_ui import desktop as _base_desktop
from transcripteur_whisper.core.settings import SettingsStore


@pytest.fixture
def desktop(qtbot, tmp_path):
    """Reuse the audited desktop fixture without importing it under the same public name."""
    yield from _base_desktop.__wrapped__(qtbot, tmp_path)


def test_diarization_is_opt_in_and_self_name_persists(desktop):
    assert desktop.diarization.isChecked() is False
    assert desktop.self_speaker_name.isEnabled() is False
    desktop.diarization.setChecked(True)
    desktop.self_speaker_name.setText("Alice Example")
    desktop._save_preferences()
    restored = SettingsStore(desktop.paths)
    assert restored.get("diarization_enabled") is True
    assert restored.get("self_speaker_name") == "Alice Example"


def test_native_recording_preserves_sources_only_when_diarization_was_enabled(desktop, qtbot):
    from transcripteur_whisper.models.audio_devices import AudioDevice, AudioDeviceSnapshot

    mic = AudioDevice("USB mic", "USB mic", 2, "USB mic", "input", "WASAPI", 1, 48000, True)
    panel = desktop.recording_panel
    panel.update_devices(AudioDeviceSnapshot((mic,), ()))
    panel.speaker_enabled.setChecked(False)

    desktop.diarization.setChecked(False)
    panel.start_recording()
    qtbot.waitUntil(lambda: desktop.recording.is_active and not panel._transition)
    assert "preserve_sources" not in desktop.recording.started[-1]
    panel.stop_recording()
    qtbot.waitUntil(lambda: not panel.busy)

    desktop.diarization.setChecked(True)
    panel.start_recording()
    qtbot.waitUntil(lambda: desktop.recording.is_active and not panel._transition)
    assert desktop.recording.started[-1]["preserve_sources"] is True
    panel.stop_recording()
    qtbot.waitUntil(lambda: not panel.busy)


def test_missing_diarization_assets_fail_closed_before_job_submission(desktop, qtbot, tmp_path, monkeypatch):
    warnings = []
    monkeypatch.setattr(QMessageBox, "warning", lambda *args: warnings.append(args[2]))
    desktop.diarization_models = SimpleNamespace(available=lambda: False)
    desktop.diarization.setChecked(True)
    meeting = tmp_path / "meeting.wav"
    meeting.write_bytes(b"media")
    desktop.files.add_paths([meeting])
    desktop.start_transcription()
    qtbot.waitUntil(lambda: not desktop._active)
    assert warnings and "prepare_diarization_assets.py" in warnings[0]
    assert desktop.jobs.submitted == []
