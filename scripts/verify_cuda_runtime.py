"""Strict runtime/package preflight; the exact sherpa ORT is validated separately."""
from __future__ import annotations

import json
from importlib.metadata import PackageNotFoundError, version

from transcripteur_whisper.services.compute_backend import (
    preload_sherpa_cuda_dependencies,
    validate_compute_backend,
)

EXPECTED = {
    "faster-whisper": "1.2.1",
    "ctranslate2": "4.6.3",
    "nvidia-cuda-runtime-cu12": "12.8.90",
    "nvidia-cublas-cu12": "12.8.5.5",
    "nvidia-cudnn-cu12": "9.8.0.87",
    "nvidia-cufft-cu12": "11.3.3.83",
    "nvidia-nvjitlink-cu12": "12.8.93",
    "nvidia-cuda-nvrtc-cu12": "12.8.93",
    "nvidia-curand-cu12": "10.3.9.90",
    "sherpa-onnx": "1.13.7+cuda12.cudnn9",
}


def main() -> int:
    installed: dict[str, str] = {}
    for package, expected in EXPECTED.items():
        try:
            actual = version(package)
        except PackageNotFoundError as exc:
            raise RuntimeError(f"Paquet requis absent : {package}") from exc
        installed[package] = actual
        if actual != expected:
            raise RuntimeError(f"Version inattendue pour {package}: {actual} (attendu {expected})")

    dlls = preload_sherpa_cuda_dependencies("cuda")
    info = validate_compute_backend("cuda", require_diarization=True)
    payload = {
        "ok": True,
        "packages": installed,
        "preloaded_dlls": {name: str(path) for name, path in dlls.items()},
        "cuda_device_count": info.cuda_device_count,
        "whisper_compute_type": info.whisper_compute_type,
        "sherpa_version": info.sherpa_version,
        "note": "Le ORT sherpa 1.24.4 est vérifié dans un processus séparé avant le smoke worker.",
    }
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
