from __future__ import annotations

from transcripteur_whisper.core.paths import AppPaths
from transcripteur_whisper.core.settings import SettingsStore
from transcripteur_whisper.services.compute_backend import GPU_REPAIR_VERSION


def test_compute_device_can_be_persisted(tmp_path):
    paths = AppPaths.create(tmp_path)
    settings = SettingsStore(paths)
    settings.set("compute_device", "cuda")
    assert SettingsStore(paths).get("compute_device") == "cuda"


def test_gpu_repair_version_is_current():
    assert GPU_REPAIR_VERSION == "1.0.1"
