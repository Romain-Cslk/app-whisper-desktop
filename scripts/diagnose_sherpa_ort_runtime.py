"""Verify the exact ONNX Runtime DLL shipped by the sherpa CUDA wheel on Windows."""
from __future__ import annotations

import json
import os
from pathlib import Path

from transcripteur_whisper.services.compute_backend import (
    _loaded_module_path,
    _sherpa_onnxruntime_path,
    prepare_sherpa_windows_runtime,
)


def _runtime_value(module) -> str:
    value = getattr(module, "onnxruntime_version", "")
    if callable(value):
        value = value()
    return str(value or "")


def main() -> int:
    if os.name != "nt":
        raise RuntimeError("Ce diagnostic est destiné à Windows.")

    before = _loaded_module_path("onnxruntime.dll")
    if before is not None:
        raise RuntimeError(
            "Le processus de diagnostic a déjà chargé onnxruntime.dll avant le contrôle sherpa : "
            f"{before}"
        )

    expected = _sherpa_onnxruntime_path()
    resolved = prepare_sherpa_windows_runtime("cuda")
    if resolved is None or resolved != expected:
        raise RuntimeError(f"Runtime ONNX sherpa mal résolu : {resolved} (attendu {expected})")

    # Import only after the exact 1.24.4 DLL has been loaded by full path.
    import sherpa_onnx

    reported = _runtime_value(sherpa_onnx)
    if reported and reported != "1.24.4":
        raise RuntimeError(
            f"sherpa_onnx annonce ONNX Runtime {reported}, alors que 1.24.4 est requis."
        )

    payload = {
        "ok": True,
        "expected_onnxruntime": str(expected),
        "loaded_onnxruntime": str(_loaded_module_path("onnxruntime.dll")),
        "sherpa_reported_onnxruntime": reported or "1.24.4 (API vérifiée par ctypes)",
        "expected_ort_api": 24,
        "sherpa_version": str(getattr(sherpa_onnx, "__version__", "")),
    }
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
