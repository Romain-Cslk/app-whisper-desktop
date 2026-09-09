"""Main-process orchestration of the isolated diarization worker."""
from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import time
import uuid
from pathlib import Path
from typing import Any, Callable

from ..models.speaker import DiarizationResult
from .diarization_model_service import MODEL_SET_ID, DiarizationModelService
from .diarization_options import DEFAULT_CLUSTERING_THRESHOLD, validate_diarization_options
from .recording_sources import find_recording_sources


class DiarizationError(RuntimeError):
    pass


class DiarizationService:
    def __init__(self, paths: Any, *, models: DiarizationModelService | None = None) -> None:
        self.paths = paths
        self.models = models if models is not None else DiarizationModelService(paths)

    @staticmethod
    def _worker_command(request: Path, response: Path) -> list[str]:
        if getattr(sys, "frozen", False):
            return [sys.executable, "--diarization-worker", str(request), str(response)]
        return [sys.executable, "-m", "transcripteur_whisper", "--diarization-worker", str(request), str(response)]

    def run(
        self,
        source: Path,
        *,
        job_id: str,
        file_index: int,
        expected_speakers: int = 0,
        clustering_threshold: float = DEFAULT_CLUSTERING_THRESHOLD,
        cancel_check: Callable[[], None],
        progress: Callable[[float], None] | None = None,
        log: Callable[[str], None] | None = None,
    ) -> DiarizationResult:
        expected_speakers, clustering_threshold = validate_diarization_options(
            expected_speakers, clustering_threshold
        )
        if not re.fullmatch(r"[0-9a-f]{32}", job_id) or type(file_index) is not int or file_index < 0:
            raise ValueError("Identifiant de traitement invalide.")
        cancel_check()
        progress = progress or (lambda _value: None)
        log = log or (lambda _message: None)
        log("Préparation des modèles de diarisation…")
        segmentation, embedding = self.models.ensure(
            cancel_check,
            lambda value: progress(min(0.20, value * 0.20)),
            log,
        )
        work = Path(self.paths.temp) / job_id / f"diarization_{file_index:03d}_{uuid.uuid4().hex[:8]}"
        work.mkdir(parents=True, exist_ok=True)
        request = work / "request.json"
        response = work / "response.json"
        stderr_path = work / "worker-stderr.log"
        payload: dict[str, Any] = {
            "schema_version": 1,
            "action": "diarize",
            "model_set_id": MODEL_SET_ID,
            "audio_path": str(Path(source).resolve()),
            "expected_speakers": expected_speakers,
            "clustering_threshold": clustering_threshold,
            "segmentation_model": str(segmentation.resolve()),
            "embedding_model": str(embedding.resolve()),
        }
        native = find_recording_sources(self.paths, source)
        if native is not None:
            if native.microphone_path:
                payload["microphone_source"] = str(native.microphone_path)
                payload["microphone_offset"] = native.microphone_offset
            if native.system_path:
                payload["system_source"] = str(native.system_path)
                payload["system_offset"] = native.system_offset
            log("Diarisation : piste microphone personnelle et piste système séparées détectées.")
        else:
            log("Diarisation : fichier mixé, identification de « moi » uniquement via profil vocal confirmé.")
        try:
            request.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
            command = self._worker_command(request, response)
            creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0) if os.name == "nt" else 0
            log("Analyse locale des locuteurs dans un processus isolé…")
            with stderr_path.open("wb") as stderr:
                process = subprocess.Popen(
                    command,
                    stdin=subprocess.DEVNULL,
                    stdout=subprocess.DEVNULL,
                    stderr=stderr,
                    creationflags=creationflags,
                )
                started = time.monotonic()
                try:
                    while process.poll() is None:
                        cancel_check()
                        # No fixed short timeout: long meetings can legitimately take minutes.
                        # Four hours is only a dead-worker safety ceiling.
                        if time.monotonic() - started > 4 * 3600:
                            raise DiarizationError("Le moteur de diarisation a dépassé la limite de sécurité de 4 heures.")
                        elapsed = time.monotonic() - started
                        progress(min(0.92, 0.20 + elapsed / 300.0 * 0.60))
                        time.sleep(0.10)
                except BaseException:
                    if process.poll() is None:
                        process.terminate()
                        try:
                            process.wait(timeout=3)
                        except subprocess.TimeoutExpired:
                            process.kill()
                            process.wait(timeout=3)
                    raise
            cancel_check()
            payload_out = None
            if response.is_file():
                try:
                    payload_out = json.loads(response.read_text(encoding="utf-8"))
                except (OSError, ValueError):
                    pass
            if isinstance(payload_out, dict) and not payload_out.get("ok"):
                raise DiarizationError(str(payload_out.get("error") or "Erreur de diarisation inconnue."))
            if process.returncode != 0 or not isinstance(payload_out, dict):
                details = stderr_path.read_text(encoding="utf-8", errors="replace")[-4000:]
                raise DiarizationError("Le processus de diarisation a echoue ou sa reponse est illisible. " + details.strip())
            result = DiarizationResult.from_dict(payload_out["result"])
            if result.model_set_id != MODEL_SET_ID:
                raise DiarizationError("Le worker a utilisé un espace d'empreintes vocales incompatible.")
            progress(1.0)
            return result
        finally:
            # This response contains raw embeddings. Do not retain it for the rest
            # of a document-generation job, or after a cancelled/failed worker.
            shutil.rmtree(work, ignore_errors=True)

    def probe_worker(self, segmentation: Path, embedding: Path, directory: Path) -> dict[str, Any]:
        directory.mkdir(parents=True, exist_ok=True)
        request = directory / "probe-request.json"
        response = directory / "probe-response.json"
        request.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "action": "probe",
                    "segmentation_model": str(segmentation.resolve()),
                    "embedding_model": str(embedding.resolve()),
                }
            ),
            encoding="utf-8",
        )
        completed = subprocess.run(
            self._worker_command(request, response),
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            timeout=180,
            creationflags=(getattr(subprocess, "CREATE_NO_WINDOW", 0) if os.name == "nt" else 0),
            check=False,
        )
        if completed.returncode != 0 or not response.is_file():
            raise DiarizationError(
                "Smoke test sherpa-onnx échoué : "
                + completed.stderr.decode("utf-8", errors="replace")[-3000:]
            )
        payload = json.loads(response.read_text(encoding="utf-8"))
        if not payload.get("ok"):
            raise DiarizationError(str(payload.get("error") or "Probe sherpa-onnx échoué."))
        return payload
