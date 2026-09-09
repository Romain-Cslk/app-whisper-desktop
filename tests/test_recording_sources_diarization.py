from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import soundfile as sf

from tests.test_audio_devices import FakeSoundCard, FakeSoundDevice
from transcripteur_whisper.audio.native_recorder import NativeRecordingResult
from transcripteur_whisper.services.audio_device_service import AudioDeviceService
from transcripteur_whisper.services.recording_service import RecordingService
from transcripteur_whisper.services.recording_sources import find_recording_sources


class SourceRecorder:
    def __init__(self, directory):
        self.directory = directory
        self.live = False
        self.kwargs = None

    def start(self, mic, speaker, **kwargs):
        self.kwargs = kwargs
        self.live = True
        recording_id = "c" * 32
        mixed = self.directory / f"native_recording_{recording_id}.wav"
        microphone = self.directory / f".native_recording_{recording_id}_microphone.wav"
        system = self.directory / f".native_recording_{recording_id}_system.wav"
        sf.write(mixed, np.zeros((4800, 2), dtype=np.float32), 48000, subtype="PCM_16")
        sf.write(microphone, np.zeros((4800, 2), dtype=np.float32), 48000, subtype="PCM_16")
        sf.write(system, np.zeros((4800, 2), dtype=np.float32), 48000, subtype="PCM_16")
        self.result = NativeRecordingResult(
            recording_id,
            mixed,
            0.1,
            microphone_source_path=microphone if kwargs.get("preserve_sources") else None,
            system_source_path=system if kwargs.get("preserve_sources") else None,
            microphone_offset=0.03,
            system_offset=0.01,
            sample_rate=48000,
        )
        return self.result

    def stop(self, recording_id=None):
        self.live = False
        return self.result

    def has_live_capture(self):
        return self.live

    def get_levels(self, recording_id):
        return None


def test_preserved_tracks_are_transactionally_copied_and_discoverable(tmp_path):
    devices = AudioDeviceService(sounddevice=FakeSoundDevice(), soundcard=FakeSoundCard())
    paths = SimpleNamespace(temp=tmp_path / "temp", results=tmp_path / "results")
    service = RecordingService(paths, devices)
    recorder = SourceRecorder(service._journal_dir)
    service._recorder = recorder
    try:
        service.start(preserve_sources=True)
        result = service.stop()
        assert recorder.kwargs["preserve_sources"] is True
        manifest = paths.results / "recording_sources" / result.recording_id / "recording.json"
        payload = json.loads(manifest.read_text(encoding="utf-8"))
        assert payload["microphone_offset"] == 0.03
        assert payload["system_offset"] == 0.01
        found = find_recording_sources(paths, result.path)
        assert found is not None
        assert found.microphone_path.is_file()
        assert found.system_path.is_file()
    finally:
        service.close()
