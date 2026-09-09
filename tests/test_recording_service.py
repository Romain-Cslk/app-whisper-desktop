from __future__ import annotations

import threading
from types import SimpleNamespace

import numpy as np
import pytest
import soundfile as sf

from tests.test_audio_devices import FakeSoundCard, FakeSoundDevice, endpoint, pa_device
from transcripteur_whisper.audio.native_recorder import NativeRecordingResult
from transcripteur_whisper.services.audio_device_service import AudioDeviceService
from transcripteur_whisper.services.recording_service import RecordingService


class FakeRecorder:
    def __init__(self, directory):
        self.directory = directory
        self.live = False
        self.calls = []
        self.unavailable = []
        self.thread_ids = []
        self.started = None
        self.release = None

    def start(self, mic, speaker, **kwargs):
        self.calls.append((mic, speaker, kwargs))
        self.thread_ids.append(threading.get_ident())
        if self.started:
            self.started.set()
            assert self.release.wait(2)
        self.live = True
        self.result = NativeRecordingResult(
            "a" * 32, self.directory / ("native_recording_" + "a" * 32 + ".wav"), 1.0
        )
        sf.write(self.result.path, np.zeros((480, 2)), 48000, subtype="PCM_16")
        return self.result

    def stop(self, recording_id=None):
        self.thread_ids.append(threading.get_ident())
        self.live = False
        return self.result

    def has_live_capture(self):
        return self.live

    def source_unavailable(self, key):
        self.unavailable.append(key)

    def get_levels(self, recording_id):
        return {"recording_id": recording_id}


@pytest.fixture
def recording(tmp_path):
    sd, sc = FakeSoundDevice(), FakeSoundCard()
    devices = AudioDeviceService(sounddevice=sd, soundcard=sc)
    paths = SimpleNamespace(temp=tmp_path / "temp", results=tmp_path / "results")
    service = RecordingService(paths, devices)
    recorder = FakeRecorder(service._journal_dir)
    service._recorder = recorder
    yield service, recorder, sd, sc
    service.close()


def test_start_resolves_persisted_identity_to_fresh_portaudio_index(recording):
    service, recorder, sd, sc = recording
    previous = service.devices.snapshot()
    selected = previous.inputs[0].stable_key
    sc.microphones.insert(0, endpoint("new", "New microphone"))
    sd.current.insert(0, pa_device("New microphone"))
    first = service.start(selected, previous.outputs[0].stable_key)
    assert recorder.calls[0][0] == "sd:1"
    assert service.start(selected).resumed
    assert len(recorder.calls) == 1
    result = service.stop()
    assert result.path.parent == service.paths.results
    assert result.path.exists()
    assert not first.path.exists()
    assert service.stop() == result
    assert len(set(recorder.thread_ids)) == 1
    assert recorder.thread_ids[0] != threading.get_ident()


def test_disconnect_marks_only_lost_active_source_and_does_not_switch(recording):
    service, recorder, sd, sc = recording
    service.start()
    sc.microphones = [endpoint("new", "Other microphone")]
    sd.current = [pa_device("Other microphone")]
    snapshot = service.devices.snapshot()
    service.device_snapshot_changed(snapshot)
    assert recorder.unavailable == ["microphone"]
    assert recorder.live
    assert len(recorder.calls) == 1
    assert sd.terminations == 1


def test_microphone_and_system_can_be_disabled_independently(recording):
    service, recorder, sd, sc = recording
    sc.microphones = []
    sd.current = []
    service.start(microphone_enabled=False, system_enabled=True)
    assert recorder.calls[0][0] is None
    assert recorder.calls[0][2] == dict(microphone_enabled=False, system_enabled=True)


def test_scan_cannot_reinitialize_portaudio_while_stream_is_opening(recording):
    service, recorder, sd, sc = recording
    recorder.started, recorder.release = threading.Event(), threading.Event()
    errors = []

    def start():
        try:
            service.start()
        except Exception as exc:
            errors.append(exc)

    opener = threading.Thread(target=start)
    opener.start()
    assert recorder.started.wait(2)
    scanner = threading.Thread(target=service.devices.snapshot)
    scanner.start()
    recorder.release.set()
    opener.join(2)
    scanner.join(2)
    assert not errors
    assert not opener.is_alive() and not scanner.is_alive()
    assert sd.terminations == 1


def test_failed_final_copy_keeps_audio_recoverable(recording, monkeypatch):
    service, recorder, sd, sc = recording
    started = service.start()

    def fail(*args):
        raise OSError("disk full")

    monkeypatch.setattr("transcripteur_whisper.services.recording_service.shutil.copyfile", fail)
    with pytest.raises(OSError, match="disk full"):
        service.stop()
    assert not service.is_active
    assert started.path in service.recoverable_recordings()


def test_final_copy_can_be_retried_after_disk_error(recording, monkeypatch):
    service, recorder, sd, sc = recording
    started = service.start()

    def fail(*args):
        raise OSError("disk full")

    with monkeypatch.context() as patch:
        patch.setattr("transcripteur_whisper.services.recording_service.shutil.copyfile", fail)
        with pytest.raises(OSError):
            service.stop()
    result = service.stop()
    assert result.path.is_file()
    assert not started.path.exists()
    assert not list(service.paths.results.glob("*.partial"))
    assert len(recorder.thread_ids) == 2  # Only one backend stop.


def test_dead_active_result_does_not_prevent_another_start(recording):
    service, recorder, sd, sc = recording
    service.start()
    recorder.live = False
    result = service.start()
    assert not result.resumed
    assert len(recorder.calls) == 2


def test_delayed_old_snapshot_cannot_mark_a_new_capture_source_lost(recording):
    service, recorder, sd, sc = recording
    # The monitor enumerates before a new USB microphone appears. Its callback
    # is delayed until after start() independently discovers and opens that mic.
    old_snapshot = service.devices.snapshot()
    sc.microphones = [endpoint("new", "New USB microphone")]
    sc.default_input = sc.microphones[0]
    sd.current = [pa_device("New USB microphone")]
    service.start()
    service.device_snapshot_changed(old_snapshot)
    assert recorder.unavailable == []
    assert service.is_active
