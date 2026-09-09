from __future__ import annotations

from types import SimpleNamespace

from PySide6.QtWidgets import QFileDialog

from transcripteur_whisper.ui.widgets.recording_panel import RecordingPanel
from transcripteur_whisper.ui.workers import TaskRunner


class IdleRecordingService:
    is_active = False


def test_completed_recording_can_be_exported_without_modifying_source(qtbot, tmp_path, monkeypatch):
    runner = TaskRunner()
    panel = RecordingPanel(runner, IdleRecordingService())
    qtbot.addWidget(panel)
    panel.show()

    source = tmp_path / "captured.wav"
    original = b"RIFF-test-recording-content"
    source.write_bytes(original)
    destination = tmp_path / "exports" / "meeting.wav"

    panel._stopped(SimpleNamespace(path=source))
    assert panel.export_button.isVisible()
    assert panel._last_recording_path == source

    monkeypatch.setattr(
        QFileDialog,
        "getSaveFileName",
        lambda *args, **kwargs: (str(destination), "Audio WAV (*.wav)"),
    )
    panel.export_button.click()
    qtbot.waitUntil(lambda: runner.active_count == 0 and destination.is_file(), timeout=5000)

    assert destination.read_bytes() == original
    assert source.read_bytes() == original
    assert "WAV exporté" in panel.status.text()


def test_export_adds_wav_suffix_when_user_omits_it(qtbot, tmp_path, monkeypatch):
    runner = TaskRunner()
    panel = RecordingPanel(runner, IdleRecordingService())
    qtbot.addWidget(panel)

    source = tmp_path / "captured.wav"
    source.write_bytes(b"audio")
    selected = tmp_path / "copie-sans-extension"

    panel._stopped(SimpleNamespace(path=source))
    monkeypatch.setattr(
        QFileDialog,
        "getSaveFileName",
        lambda *args, **kwargs: (str(selected), "Audio WAV (*.wav)"),
    )
    panel.export_last_recording()
    expected = selected.with_suffix(".wav")
    qtbot.waitUntil(lambda: runner.active_count == 0 and expected.is_file(), timeout=5000)
    assert expected.read_bytes() == b"audio"
