"""Extract reviewed speaker samples in a clean worker process."""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import uuid
from pathlib import Path
from typing import Any, Iterable

from .diarization_model_service import DiarizationModelService
from .speaker_audio import AudioExcerpt


class SpeakerSampleError(RuntimeError):
    pass


class SpeakerSampleService:
    def __init__(self, paths: Any) -> None:
        self.paths = paths
        self.models = DiarizationModelService(paths)

    @staticmethod
    def _worker_command(request: Path, response: Path) -> list[str]:
        if getattr(sys, "frozen", False):
            return [sys.executable, "--speaker-sample-worker", str(request), str(response)]
        return [sys.executable, "-m", "transcripteur_whisper", "--speaker-sample-worker", str(request), str(response)]

    def extract(
        self,
        excerpts: Iterable[AudioExcerpt],
        *,
        compute_device: str = "cpu",
    ) -> list[tuple[float, ...] | None]:
        values = list(excerpts)
        if not values:
            return []
        device = str(compute_device or "cpu").lower()
        if device not in {"cpu", "cuda"}:
            raise ValueError("Périphérique de calcul vocal inconnu.")
        _segmentation, embedding = self.models.ensure()
        work = Path(self.paths.temp) / f"speaker_samples_{uuid.uuid4().hex}"
        work.mkdir(parents=True, exist_ok=False)
        request = work / "request.json"
        response = work / "response.json"
        try:
            request.write_text(json.dumps({
                "schema_version": 1,
                "provider": "cuda" if device == "cuda" else "cpu",
                "embedding_model": str(embedding.resolve()),
                "clips": [
                    {"path": str(Path(item.path).resolve()), "start": item.start, "end": item.end}
                    for item in values
                ],
            }, ensure_ascii=False), encoding="utf-8")
            completed = subprocess.run(
                self._worker_command(request, response),
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
                timeout=300,
                creationflags=(getattr(subprocess, "CREATE_NO_WINDOW", 0) if os.name == "nt" else 0),
                check=False,
            )
            payload = None
            if response.is_file():
                try:
                    payload = json.loads(response.read_text(encoding="utf-8"))
                except (OSError, ValueError):
                    pass
            if completed.returncode != 0 or not isinstance(payload, dict) or not payload.get("ok"):
                details = completed.stderr.decode("utf-8", errors="replace")[-3000:]
                message = str(payload.get("error") if isinstance(payload, dict) else "") or details
                raise SpeakerSampleError("L'analyse des extraits vocaux a échoué. " + message.strip())
            vectors = payload.get("vectors")
            if not isinstance(vectors, list) or len(vectors) != len(values):
                raise SpeakerSampleError("Réponse incomplète du moteur d'empreintes vocales.")
            output: list[tuple[float, ...] | None] = []
            for vector in vectors:
                output.append(tuple(float(x) for x in vector) if isinstance(vector, list) else None)
            return output
        finally:
            shutil.rmtree(work, ignore_errors=True)
