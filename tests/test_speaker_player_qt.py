# ruff: noqa: E402
"""Qt multimedia lifecycle tests. They do not certify an audible hardware output."""
from __future__ import annotations

import struct
import wave

import pytest

pytest.importorskip('PySide6.QtMultimedia')
from transcripteur_whisper.services.speaker_audio import AudioExcerpt
from transcripteur_whisper.ui.widgets.speaker_player import SpeakerExcerptPlayer


def test_missing_file_is_explicit_and_no_playback_starts(qtbot, tmp_path):
    player = SpeakerExcerptPlayer()
    errors = []
    player.failed.connect(errors.append)
    player.play_excerpt(AudioExcerpt(tmp_path / 'absent.wav', 0, 1, 0, 'a', False))
    assert errors and player._clip is None
    player.close()


def test_real_wav_load_seek_and_auto_stop(qtbot, tmp_path):
    wav = tmp_path / 'silence.wav'
    with wave.open(str(wav), 'wb') as stream:
        stream.setparams((1, 2, 16000, 0, 'NONE', 'not compressed'))
        stream.writeframes(struct.pack('<h', 0) * (16000 * 2))
    player = SpeakerExcerptPlayer()
    errors = []
    player.failed.connect(errors.append)
    player.play_excerpt(AudioExcerpt(wav, 0.5, 0.8, 0.5, 'a', False))
    qtbot.waitUntil(lambda: player._clip is None or bool(errors), timeout=18000)
    player.close()
    assert not errors, f'Qt media backend must support WAV for this release: {errors}'
    assert player.player.source().isEmpty()
