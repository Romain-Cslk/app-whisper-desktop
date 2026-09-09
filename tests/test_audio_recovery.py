from __future__ import annotations

import os
import struct
import time
from unittest import mock

import numpy as np
import pytest
import soundfile as sf

from transcripteur_whisper.audio import native_recorder as native
from transcripteur_whisper.audio.recovery import RecordingRecoveryError, recover_wav


def session_in(tmp_path, **kwargs):
    return native._RecordingSession(
        "b" * 32,
        tmp_path / ("native_recording_" + "b" * 32 + ".wav"),
        0,
        "Microphone",
        "speaker-id",
        "Speakers",
        **kwargs,
    )


def test_old_interrupted_journals_are_never_deleted_on_launch(tmp_path):
    journal = tmp_path / (".native_recording_" + "b" * 32 + "_microphone.wav")
    sf.write(journal, np.ones((240, 2), dtype=np.float32) * 0.1, 48000, subtype="PCM_16")
    old = time.time() - 90 * 24 * 3600
    os.utime(journal, (old, old))
    native.NativeRecorder(tmp_path)
    assert journal.exists()
    assert sf.info(journal).frames == 240


def test_recovery_repairs_stale_wav_header_and_preserves_original(tmp_path):
    journal = tmp_path / "interrupted.wav"
    sf.write(journal, np.ones((1200, 2), dtype=np.float32) * 0.25, 48000, subtype="PCM_16")
    contents = bytearray(journal.read_bytes())
    data_header = contents.index(b"data")
    contents[data_header + 4 : data_header + 8] = struct.pack("<I", 0)
    contents[4:8] = struct.pack("<I", data_header)
    journal.write_bytes(contents)
    recovered = recover_wav(journal, tmp_path / "results")
    assert sf.info(recovered).frames == 1200
    assert journal.read_bytes() == contents
    values, _ = sf.read(recovered)
    assert np.allclose(values, 0.25)


def test_empty_or_invalid_journal_reports_error_without_destroying_source(tmp_path):
    journal = tmp_path / "broken.wav"
    journal.write_bytes(b"incomplete header")
    with pytest.raises(RecordingRecoveryError):
        recover_wav(journal, tmp_path / "results")
    assert journal.exists()
    assert not list((tmp_path / "results").glob("*.wav"))


def test_source_loss_drains_its_journal_and_keeps_other_source_running(tmp_path):
    session = session_in(tmp_path)
    session._start_ts = time.monotonic()
    session._start_writer_threads()
    session._sources["microphone"].audio_queue.put(np.ones((300, 2), dtype="float32") * 0.1)
    session._sources["system"].audio_queue.put(np.ones((600, 2), dtype="float32") * 0.2)
    session._mark_source_failure("microphone", "disconnected", RuntimeError("USB removed"))
    assert session._sources["microphone"].writer_done.wait(2)
    assert not session._sources["system"].reader_done.is_set()
    session._sources["system"].audio_queue.put(np.ones((600, 2), dtype="float32") * 0.2)
    snapshot = session.levels_snapshot()
    assert snapshot["degraded"]
    assert snapshot["sources"]["microphone"]["failed"]
    assert snapshot["sources"]["system"]["available"]
    result = session.stop()
    assert sf.info(result.path).frames == 1200
    assert all(not source.path.exists() for source in session._sources.values())


def test_mix_failure_salvages_longest_journal_before_deleting_sources(tmp_path):
    session = session_in(tmp_path)
    session._start_ts = time.monotonic()
    for key, frames in (("microphone", 240), ("system", 480)):
        sf.write(session._sources[key].path, np.zeros((frames, 2)), 48000, subtype="PCM_16")
    with mock.patch.object(session, "_mix_sources", side_effect=RuntimeError("mix error")):
        result = session.stop()
    assert sf.info(result.path).frames == 480
    assert all(not source.path.exists() for source in session._sources.values())


@pytest.mark.parametrize("enabled", ["microphone", "system"])
def test_single_source_capture_writes_only_enabled_journal(tmp_path, enabled):
    session = session_in(
        tmp_path, microphone_enabled=enabled == "microphone", system_enabled=enabled == "system"
    )
    session._reset_runtime_state()
    session._start_ts = time.monotonic()
    session._start_writer_threads()
    session._sources[enabled].audio_queue.put(np.ones((480, 2), dtype="float32") * 0.2)
    levels = session.levels_snapshot()
    assert levels["available_sources"] == 1
    assert not levels["degraded"]
    result = session.stop()
    assert sf.info(result.path).frames == 480


def test_failed_stream_close_is_retained_and_prevents_portaudio_reset(tmp_path):
    session = session_in(tmp_path)
    session._mic_stream = mock.Mock(active=False, closed=False)
    session._mic_stream.close.side_effect = RuntimeError("driver blocked")
    with pytest.raises(RuntimeError):
        session._close_microphone_stream()
    assert session._mic_stream is not None
    assert session.has_live_workers()


def test_failed_start_cleanup_preserves_samples_already_written(tmp_path):
    session = session_in(tmp_path)
    source = session._sources["microphone"]
    sf.write(source.path, np.zeros((480, 2)), 48000, subtype="PCM_16")
    source.frames_written = 480
    with pytest.raises(native.NativeRecorderError, match="récupérables"):
        session._cleanup_failed_attempt()
    assert sf.info(source.path).frames == 480


def test_mix_keeps_source_timing_offset_and_gains(tmp_path):
    session = session_in(tmp_path)
    session._start_ts = 10.0
    system = session._sources["system"]
    microphone = session._sources["microphone"]
    system.first_frame_monotonic = 10.0
    microphone.first_frame_monotonic = 10.01
    system.gain, microphone.gain = 0.5, 1.0
    sf.write(system.path, np.ones((480, 2)) * 0.2, 48000, subtype="PCM_16")
    sf.write(microphone.path, np.ones((480, 2)) * 0.3, 48000, subtype="PCM_16")
    session._mix_sources()
    data, _ = sf.read(session.output_path)
    assert len(data) == 960
    assert np.allclose(data[:480], 0.1, atol=0.0001)
    assert np.allclose(data[480:], 0.3, atol=0.0001)


@pytest.mark.parametrize("operation", ["stop", "_cleanup_failed_attempt"])
def test_blocked_soundcard_reader_is_never_released_concurrently_and_stop_can_retry(tmp_path, operation):
    session = session_in(tmp_path)
    session._start_ts = time.monotonic()
    reader = mock.Mock()
    reader.is_alive.return_value = True
    session._system_reader_thread = reader
    context = mock.MagicMock()
    session._system_context = session._system_recorder = context
    source = session._sources["system"]
    sf.write(source.path, np.zeros((480, 2)), 48000, subtype="PCM_16")
    source.frames_written = 480
    with pytest.raises(native.NativeRecorderError):
        getattr(session, operation)()
    context.__exit__.assert_not_called()
    assert source.path.exists()
    assert session.has_live_workers()
    # Once the driver releases the reader, another stop safely closes on the
    # coordinator thread and finalizes the samples captured before the failure.
    reader.is_alive.return_value = False
    result = session.stop()
    context.__exit__.assert_called_once()
    assert sf.info(result.path).frames == 480
