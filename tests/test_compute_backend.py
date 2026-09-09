from __future__ import annotations

from pathlib import Path

import pytest

from transcripteur_whisper.models.jobs import ValidationError
from transcripteur_whisper.models.transcription import TranscriptionOptions
from transcripteur_whisper.services import compute_backend


class FakeCTranslate2:
    def __init__(self, count=1, supported=None):
        self.count = count
        self.supported = supported or {"float16", "float32"}

    def get_cuda_device_count(self):
        return self.count

    def get_supported_compute_types(self, device):
        assert device == "cuda"
        return self.supported


class FakeSherpa:
    __version__ = "1.13.7+cuda12.cudnn9"


def test_transcription_options_default_to_cpu_and_reject_unknown_device():
    options = TranscriptionOptions()
    assert options.compute_device == "cpu"
    options.validate("")
    with pytest.raises(ValidationError):
        TranscriptionOptions(compute_device="metal").validate("")


def test_cuda_backend_prefers_float16(monkeypatch):
    monkeypatch.setattr(compute_backend, "_load_ctranslate2", lambda: FakeCTranslate2())
    monkeypatch.setattr(compute_backend, "_load_sherpa_onnx", lambda: FakeSherpa())
    info = compute_backend.validate_compute_backend("cuda", require_diarization=True)
    assert info.device == "cuda"
    assert info.whisper_compute_type == "float16"
    assert info.sherpa_provider == "cuda"
    assert info.cuda_device_count == 1


def test_cuda_backend_rejects_cpu_only_sherpa(monkeypatch):
    class CpuSherpa:
        __version__ = "1.13.7"

    monkeypatch.setattr(compute_backend, "_load_ctranslate2", lambda: FakeCTranslate2())
    monkeypatch.setattr(compute_backend, "_load_sherpa_onnx", lambda: CpuSherpa())
    with pytest.raises(compute_backend.ComputeBackendError, match="version CPU"):
        compute_backend.validate_compute_backend("cuda", require_diarization=True)


def test_cuda_backend_rejects_missing_gpu(monkeypatch):
    monkeypatch.setattr(compute_backend, "_load_ctranslate2", lambda: FakeCTranslate2(count=0))
    with pytest.raises(compute_backend.ComputeBackendError, match="Aucun GPU NVIDIA"):
        compute_backend.validate_compute_backend("cuda")


def test_diarization_worker_propagates_cuda_provider(monkeypatch, tmp_path):
    from transcripteur_whisper.services import diarization_worker

    segmentation = tmp_path / "segmentation.onnx"
    embedding = tmp_path / "embedding.onnx"
    audio = tmp_path / "meeting.wav"
    for path in (segmentation, embedding, audio):
        path.write_bytes(b"test")

    calls = []

    def fake_process_one(path, **kwargs):
        calls.append((path, kwargs))
        return [], []

    monkeypatch.setattr(diarization_worker, "_process_one", fake_process_one)
    result = diarization_worker.process_request(
        {
            "schema_version": 1,
            "model_set_id": diarization_worker.MODEL_SET_ID,
            "audio_path": str(audio),
            "expected_speakers": 0,
            "clustering_threshold": 0.60,
            "provider": "cuda",
            "segmentation_model": str(segmentation),
            "embedding_model": str(embedding),
        }
    )
    assert result.model_set_id == diarization_worker.MODEL_SET_ID
    assert calls and calls[0][1]["provider"] == "cuda"


def test_diarizer_builds_segmentation_and_embedding_with_cuda(monkeypatch):
    import sys
    from types import ModuleType, SimpleNamespace

    from transcripteur_whisper.services import diarization_worker

    captured = {}
    fake = ModuleType("sherpa_onnx")

    class SegmentationConfig:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

    class EmbeddingConfig:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

    class DiarizationConfig:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

        def validate(self):
            return True

    def diarizer(config):
        captured["config"] = config
        return config

    fake.OfflineSpeakerSegmentationPyannoteModelConfig = lambda **kwargs: SimpleNamespace(**kwargs)
    fake.OfflineSpeakerSegmentationModelConfig = SegmentationConfig
    fake.SpeakerEmbeddingExtractorConfig = EmbeddingConfig
    fake.FastClusteringConfig = lambda **kwargs: SimpleNamespace(**kwargs)
    fake.OfflineSpeakerDiarizationConfig = DiarizationConfig
    fake.OfflineSpeakerDiarization = diarizer
    monkeypatch.setitem(sys.modules, "sherpa_onnx", fake)

    diarization_worker._make_diarizer(
        Path("segmentation.onnx"),
        Path("embedding.onnx"),
        -1,
        provider="cuda",
    )
    config = captured["config"]
    assert config.segmentation.provider == "cuda"
    assert config.embedding.provider == "cuda"
