"""Isolated worker that extracts one speaker embedding per selected audio excerpt.

It deliberately runs in a fresh process before Qt/faster-whisper are imported so
sherpa-onnx and ONNX Runtime can load their CUDA runtime without DLL collisions.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np


def _normalise(vector: Any) -> list[float] | None:
    if vector is None:
        return None
    value = np.asarray(vector, dtype=np.float32).reshape(-1)
    if value.size < 16 or not np.isfinite(value).all():
        return None
    norm = float(np.linalg.norm(value))
    if not math.isfinite(norm) or norm <= 1e-8:
        return None
    return [float(item) for item in value / norm]


def worker_main(request_path: Path, response_path: Path) -> int:
    response: dict[str, Any] = {"ok": False, "vectors": []}
    try:
        payload = json.loads(Path(request_path).read_text(encoding="utf-8"))
        model_path = Path(str(payload["embedding_model"])).resolve()
        provider = str(payload.get("provider") or "cpu").lower()
        if provider not in {"cpu", "cuda"}:
            raise ValueError("Provider de calcul vocal invalide.")
        clips = payload.get("clips")
        if not isinstance(clips, list) or not clips:
            raise ValueError("Aucun extrait vocal à analyser.")

        from .diarization_worker import _decode_audio, _embedding, _make_extractor

        extractor = _make_extractor(model_path, provider)
        cache: dict[Path, tuple[np.ndarray, int]] = {}
        vectors: list[list[float] | None] = []
        errors: list[str] = []

        for index, item in enumerate(clips):
            try:
                source = Path(str(item["path"])).expanduser().resolve()
                start = float(item["start"])
                end = float(item["end"])
                if not source.is_file() or not math.isfinite(start) or not math.isfinite(end) or end <= start or start < 0:
                    raise ValueError("Extrait audio invalide.")
                if source not in cache:
                    cache[source] = _decode_audio(source)
                samples, sample_rate = cache[source]
                left = max(0, round(start * sample_rate))
                right = min(samples.size, round(end * sample_rate))
                if right <= left:
                    raise ValueError("Extrait hors limites.")
                vector = _normalise(_embedding(extractor, samples[left:right], sample_rate))
                if vector is None:
                    raise ValueError("Empreinte vocale inexploitable.")
                vectors.append(vector)
                errors.append("")
            except Exception as exc:
                vectors.append(None)
                errors.append(f"Extrait {index + 1}: {exc}")

        response.update(ok=True, vectors=vectors, errors=errors)
        code = 0
    except Exception as exc:
        response["error"] = str(exc)
        code = 1

    Path(response_path).parent.mkdir(parents=True, exist_ok=True)
    Path(response_path).write_text(
        json.dumps(response, ensure_ascii=False, allow_nan=False), encoding="utf-8"
    )
    return code


if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise SystemExit("Usage: speaker_sample_worker REQUEST_JSON RESPONSE_JSON")
    raise SystemExit(worker_main(Path(sys.argv[1]), Path(sys.argv[2])))
