from __future__ import annotations

import threading
from types import SimpleNamespace

import pytest

from transcripteur_whisper.models.audio_devices import select_device
from transcripteur_whisper.services.audio_device_service import AudioDeviceError, AudioDeviceService


def endpoint(identifier="mic-usb", name="Microphone USB", channels=1):
    return SimpleNamespace(id=identifier, name=name, channels=channels)


def pa_device(name="Microphone USB", channels=1):
    return dict(name=name, max_input_channels=channels, hostapi=0, default_samplerate=48000)


class FakeSoundDevice:
    def __init__(self):
        self.current = [pa_device()]
        self.cached = list(self.current)
        self.refreshes = 0
        self.terminations = 0
        self.threads = []
        self._initialized = 1

    def _terminate(self):
        self.terminations += 1
        self._initialized -= 1

    def _initialize(self):
        self.refreshes += 1
        self._initialized += 1
        self.cached = list(self.current)

    def query_devices(self):
        self.threads.append(threading.get_ident())
        return self.cached

    def query_hostapis(self):
        return [dict(name="Windows WASAPI")]


class FakeSoundCard:
    def __init__(self):
        self.microphones = [endpoint()]
        self.speakers = [endpoint("speakers", "Speakers", 2)]
        self.default_input = self.microphones[0]
        self.default_output = self.speakers[0]

    def all_microphones(self, include_loopback=False):
        assert not include_loopback
        return self.microphones

    def all_speakers(self):
        return self.speakers

    def default_microphone(self):
        return self.default_input

    def default_speaker(self):
        return self.default_output


@pytest.fixture
def audio_backends():
    sd, sc = FakeSoundDevice(), FakeSoundCard()
    return AudioDeviceService(sounddevice=sd, soundcard=sc), sd, sc


def test_idle_portaudio_cache_refresh_and_current_index_resolution(audio_backends):
    service, sd, sc = audio_backends
    first = service.snapshot().inputs[0]
    # The existing microphone changes index after another device is inserted.
    sc.microphones.insert(0, endpoint("new", "Bluetooth microphone"))
    sd.current.insert(0, pa_device("Bluetooth microphone"))
    current = service.snapshot()
    assert service.resolve(first.stable_key, "input", snapshot=current).backend_index == 1
    assert current.find(first.stable_key).backend_id == "mic-usb"
    assert sd.refreshes == 2
    service.snapshot()
    assert sd.refreshes == 2  # Unchanged polls never cycle PortAudio.


def test_hotplug_during_capture_lists_endpoints_without_reinitializing(audio_backends):
    service, sd, sc = audio_backends
    initial = service.snapshot()
    service.set_capture_active(True)
    sc.microphones = [endpoint("new", "Bluetooth microphone")]
    sc.default_input = sc.microphones[0]
    sd.current = [pa_device("Bluetooth microphone")]
    while_recording = service.snapshot(force_refresh=True)
    assert len(while_recording.inputs) == 1
    assert while_recording.inputs[0].backend_id == "new"
    assert while_recording.inputs[0].backend_index is None
    assert while_recording.find(initial.inputs[0].stable_key) is None
    assert sd.terminations == 1
    service.set_capture_active(False)
    after = service.snapshot()
    assert after.inputs[0].backend_index == 0
    assert sd.terminations == 2


def test_default_device_change_is_detected_without_changing_explicit_selection(audio_backends):
    service, sd, sc = audio_backends
    selected = service.snapshot().inputs[0].stable_key
    sc.microphones.append(endpoint("bt", "Bluetooth microphone"))
    sd.current.append(pa_device("Bluetooth microphone"))
    sc.default_input = sc.microphones[-1]
    changed = service.snapshot()
    assert select_device(changed.inputs, selected) == selected
    assert select_device(changed.inputs, "removed") == changed.inputs[0].stable_key
    assert changed.inputs[0].is_default


def test_unplug_all_devices_and_no_default_is_a_valid_empty_snapshot(audio_backends):
    service, sd, sc = audio_backends
    initial = service.snapshot()
    sc.microphones = []
    sc.speakers = []
    sc.default_input = sc.default_output = None
    sd.current = []
    empty = service.snapshot()
    assert not empty.all
    assert select_device(empty.inputs, initial.inputs[0].stable_key) is None
    with pytest.raises(AudioDeviceError, match="plus disponible"):
        service.resolve(initial.inputs[0].stable_key, "input", snapshot=empty)


def test_ambiguous_same_name_microphones_never_pick_first_index(audio_backends):
    service, sd, sc = audio_backends
    sc.microphones.append(endpoint("second-identical-usb", "Microphone USB"))
    sd.current.append(pa_device())
    snapshot = service.snapshot()
    assert len({device.stable_key for device in snapshot.inputs}) == 2
    assert all(device.backend_index is None for device in snapshot.inputs)
    with pytest.raises(AudioDeviceError, match="ambiguïté"):
        service.resolve(snapshot.inputs[0].stable_key, "input", snapshot=snapshot)


def test_enumeration_failure_preserves_previous_snapshot(audio_backends, monkeypatch):
    service, sd, sc = audio_backends
    previous = service.snapshot()

    def fail():
        raise RuntimeError("endpoint enumeration unavailable")

    monkeypatch.setattr(sc, "all_speakers", fail)
    with pytest.raises(AudioDeviceError):
        service.snapshot()
    assert service.current_snapshot is previous


def test_portaudio_initialization_failure_can_retry_without_overtermination(audio_backends, monkeypatch):
    service, sd, sc = audio_backends
    real_initialize = sd._initialize

    def fail():
        raise RuntimeError("PortAudio temporarily unavailable")

    monkeypatch.setattr(sd, "_initialize", fail)
    with pytest.raises(AudioDeviceError):
        service.snapshot()
    assert sd._initialized == 0
    monkeypatch.setattr(sd, "_initialize", real_initialize)
    assert service.snapshot().inputs
    assert sd.terminations == 1
    assert sd._initialized == 1


def test_channel_count_disambiguates_names_only_when_both_sides_are_unique(audio_backends):
    service, sd, sc = audio_backends
    sc.microphones.append(endpoint("stereo-usb", "Microphone USB", 2))
    sd.current.append(pa_device(channels=2))
    snapshot = service.snapshot()
    assert {device.backend_index for device in snapshot.inputs} == {0, 1}
