"""Job ownership, cancellation, persistent history and desktop exports."""
from __future__ import annotations

import json
import logging
import re
import shutil
import threading
import uuid
import zipfile
from concurrent.futures import Future
from copy import deepcopy
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

from ..core.config import (
    MAX_API_JOB_WORKERS,
    MAX_FILES_PER_JOB,
    MAX_JOB_LOG_ENTRIES,
    MAX_JOB_UPLOAD_BYTES,
    MAX_QUEUED_JOBS,
    MAX_UPLOAD_BYTES,
)
from ..models.jobs import TERMINAL_STATUSES, JobCancelled, JobError, ProgressReporter, ValidationError
from ..models.transcription import TranscriptionOptions
from .executor import DaemonJobExecutor
from .files import safe_output_name, safe_upload_name, write_text_atomic
from .transcription_service import TranscriptionService

logger = logging.getLogger(__name__)


def timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def redact(message: str, secret: str = "") -> str:
    if secret:
        message = message.replace(secret, "[clé masquée]")
    return re.sub(r"\bsk-[A-Za-z0-9_-]{8,}", "[clé masquée]", message)


class JobService:
    def __init__(self, paths, *, transcription=None, load_history: bool = True):
        self.paths = paths
        self._lock = threading.RLock()
        self._jobs: dict[str, dict] = {}
        self._futures: dict[str, Future] = {}
        self._closed = False
        self._engine = transcription or TranscriptionService(paths)
        self._local = DaemonJobExecutor(1, "whisper-local")
        self._api = DaemonJobExecutor(MAX_API_JOB_WORKERS, "whisper-api")
        if load_history:
            self.load_history()

    @property
    def busy(self) -> bool:
        with self._lock:
            return any(job["status"] not in TERMINAL_STATUSES for job in self._jobs.values())

    def submit(self, files: list[Path], options: TranscriptionOptions, api_key: str = "") -> str:
        api_key = api_key.strip()
        options.validate(api_key)
        if not files or len(files) > MAX_FILES_PER_JOB:
            raise ValidationError(f"Sélectionnez entre 1 et {MAX_FILES_PER_JOB} fichiers.")
        metadata = []
        total_size = 0
        for index, raw_path in enumerate(files, start=1):
            path = Path(raw_path).expanduser().resolve()
            stored = safe_upload_name(path.name, index)
            if not path.is_file():
                raise ValidationError(f"Fichier introuvable : {path.name}")
            size = path.stat().st_size
            total_size += size
            if not size:
                raise ValidationError(f"Fichier vide : {path.name}")
            if size > MAX_UPLOAD_BYTES or total_size > MAX_JOB_UPLOAD_BYTES:
                raise ValidationError("La taille des fichiers dépasse la limite de traitement.")
            metadata.append({"name": path.name, "path": str(path), "output_stem": Path(stored).stem,
                             "size": size, "status": "queued", "stage": "queued", "progress": 0.0,
                             "out_path": None, "transcription_path": None, "document_path": None,
                             "transcription_available": False, "document_available": False,
                             "segment_index": 0, "segment_count": 0, "error": None})
        with self._lock:
            if self._closed:
                raise JobError("L'application est en cours d'arrêt.")
            if sum(j["status"] not in TERMINAL_STATUSES for j in self._jobs.values()) >= MAX_QUEUED_JOBS:
                raise JobError("Trop de traitements en attente. Réessayez plus tard.")
            job_id = uuid.uuid4().hex
            job = {"id": job_id, "job_id": job_id, "status": "pending", "progress": 0.0,
                   "created_at": timestamp(), "finished_at": None, "logs": [], "files": metadata,
                   "cancel_requested": False, "options": asdict(options), "mode": options.mode,
                   "use_api": options.use_api, "model": options.model, "lang": options.language,
                   "output_type": options.output_type, "output_name": safe_output_name(options.output_name),
                   "model_download_progress": 0.0}
            self._jobs[job_id] = job
            try:
                self._persist(job_id, required=True)
                executor = self._api if options.use_api else self._local
                future = executor.submit(self._run, job_id, options, api_key)
                self._futures[job_id] = future
                future.add_done_callback(lambda completed: self._forget_future(job_id, completed))
            except BaseException:
                self._jobs.pop(job_id, None)
                raise
        return job_id

    def snapshot(self, job_id: str) -> dict:
        with self._lock:
            if job_id not in self._jobs:
                raise JobError("Traitement introuvable.")
            result = deepcopy(self._jobs[job_id])
        result.pop("cancel_requested", None)
        result["has_transcriptions"] = any(f["transcription_available"] for f in result["files"])
        result["has_documents"] = any(f["document_available"] for f in result["files"])
        for metadata in result["files"]:
            metadata.pop("path", None)
            metadata.pop("output_stem", None)
            metadata["output_name"] = Path(metadata["out_path"]).name if metadata.get("out_path") else None
        return result

    def history(self) -> list[dict]:
        with self._lock:
            identifiers = sorted(self._jobs, key=lambda value: self._jobs[value]["created_at"], reverse=True)
        return [self.snapshot(value) for value in identifiers]

    def cancel(self, job_id: str) -> dict:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                raise JobError("Traitement introuvable.")
            if job["status"] in TERMINAL_STATUSES:
                return self.snapshot(job_id)
            job["cancel_requested"] = True
            future = self._futures.get(job_id)
            cancelled = bool(future and future.cancel())
            job["status"] = "cancelled" if cancelled else "cancelling"
            job["logs"].append("Annulation demandée. Les résultats déjà obtenus sont conservés.")
            if cancelled:
                self._mark_unfinished(job, "cancelled")
                job["finished_at"] = timestamp()
        self._persist(job_id)
        return self.snapshot(job_id)

    def shutdown(self, wait: bool = False) -> None:
        with self._lock:
            self._closed = True
            active = [key for key, job in self._jobs.items() if job["status"] not in TERMINAL_STATUSES]
        for identifier in active:
            self.cancel(identifier)
        self._local.shutdown(wait=wait, cancel_futures=True)
        self._api.shutdown(wait=wait, cancel_futures=True)

    def export(self, job_id: str, destination: Path, kind: str = "result") -> Path:
        job = self.snapshot(job_id)
        if kind not in {"result", "transcription", "document"}:
            raise ValidationError("Type d'export inconnu.")
        key = {"result": "out_path", "transcription": "transcription_path", "document": "document_path"}[kind]
        destination = Path(destination).expanduser().resolve()
        result_root = (self.paths.results / job_id).resolve()
        files = []
        for metadata in job["files"]:
            keys = ("transcription_path", "document_path") if destination.suffix.lower() == ".zip" else (key,)
            for source_key in keys:
                raw = metadata.get(source_key)
                if raw:
                    source = Path(raw).resolve()
                    if source.is_relative_to(result_root) and source.is_file() and source not in files:
                        files.append(source)
        if not files:
            raise JobError("Aucun résultat disponible pour cet export.")
        if destination in files:
            return destination
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.suffix.lower() == ".zip":
            temporary = destination.with_name(f".{destination.name}.{uuid.uuid4().hex}.tmp")
            try:
                with zipfile.ZipFile(temporary, "w", zipfile.ZIP_DEFLATED) as archive:
                    for source in files:
                        archive.write(source, source.name)
                temporary.replace(destination)
            finally:
                temporary.unlink(missing_ok=True)
        else:
            content = files[0].read_text(encoding="utf-8") if len(files) == 1 else "\n\n".join(
                f"===== {source.name} =====\n{source.read_text(encoding='utf-8')}" for source in files)
            write_text_atomic(destination, content)
        return destination

    def _run(self, job_id: str, options: TranscriptionOptions, api_key: str) -> None:
        def check() -> None:
            with self._lock:
                if self._jobs[job_id]["cancel_requested"]:
                    raise JobCancelled()

        def log(message: str) -> None:
            with self._lock:
                messages = self._jobs[job_id]["logs"]
                messages.append(redact(message, api_key))
                del messages[:-MAX_JOB_LOG_ENTRIES]

        def update_file(index: int, **changes) -> None:
            with self._lock:
                if changes.get("error"):
                    changes["error"] = redact(str(changes["error"]), api_key)
                self._jobs[job_id]["files"][index].update(changes)
            if "out_path" in changes or "status" in changes:
                self._persist(job_id)

        def update_job(**changes) -> None:
            with self._lock:
                self._jobs[job_id].update(changes)

        try:
            check()
            update_job(status="running")
            log(f"Mode {'OpenAI' if options.use_api else 'local'} · modèle {options.model}")
            self._engine.run(self._jobs[job_id], options, api_key,
                             ProgressReporter(check, log, update_file, update_job))
            check()
            with self._lock:
                statuses = [item["status"] for item in self._jobs[job_id]["files"]]
            status = "done" if all(item == "done" for item in statuses) else (
                "partial" if any(item in {"done", "partial"} for item in statuses) else "error")
            update_job(status=status, progress=1.0)
            log("Traitement terminé. Les résultats disponibles sont conservés dans votre dossier utilisateur.")
        except JobCancelled:
            with self._lock:
                self._mark_unfinished(self._jobs[job_id], "cancelled")
            update_job(status="cancelled")
            log("Traitement annulé.")
        except Exception as exc:
            error = redact(str(exc), api_key)
            with self._lock:
                self._mark_unfinished(self._jobs[job_id], "error", error)
            update_job(status="error")
            log(f"Échec du traitement : {error}")
        finally:
            update_job(finished_at=timestamp())
            self._persist(job_id)
            shutil.rmtree(self.paths.temp / job_id, ignore_errors=True)

    @staticmethod
    def _mark_unfinished(job: dict, status: str, error: str | None = None) -> None:
        for metadata in job["files"]:
            if metadata["status"] in {"queued", "running"}:
                metadata.update(status=status, stage=status, error=error)

    def _forget_future(self, job_id: str, completed: Future) -> None:
        with self._lock:
            if self._futures.get(job_id) is completed:
                self._futures.pop(job_id, None)

    def _persist(self, job_id: str, required: bool = False) -> None:
        with self._lock:
            manifest = deepcopy(self._jobs[job_id])
        # Source paths and credentials never belong in persistent history.
        manifest.pop("cancel_requested", None)
        for metadata in manifest["files"]:
            metadata.pop("path", None)
        try:
            write_text_atomic(self.paths.results / job_id / "job.json",
                              json.dumps(manifest, ensure_ascii=False, indent=2))
        except OSError:
            if required:
                raise
            logger.warning("Impossible de sauvegarder l'état du traitement %s", job_id)

    def load_history(self) -> list[dict]:
        """Read persistent manifests in a caller-owned worker at UI startup."""
        for manifest in self.paths.results.glob("*/job.json"):
            if not re.fullmatch(r"[0-9a-f]{32}", manifest.parent.name):
                continue
            try:
                job = json.loads(manifest.read_text(encoding="utf-8"))
                if (job["id"] != manifest.parent.name or not isinstance(job.get("files"), list)
                        or not isinstance(job.get("logs"), list) or not isinstance(job.get("created_at"), str)):
                    continue
                with self._lock:
                    if job["id"] in self._jobs:
                        continue
                for metadata in job["files"]:
                    metadata.setdefault("name", "Résultat")
                    metadata.setdefault("status", "cancelled")
                    metadata.setdefault("progress", 0.0)
                    metadata.setdefault("stage", metadata["status"])
                    for key, availability in (("transcription_path", "transcription_available"),
                                              ("document_path", "document_available")):
                        raw = metadata.get(key)
                        exists = bool(raw and Path(raw).resolve().is_relative_to(manifest.parent.resolve())
                                      and Path(raw).is_file())
                        metadata[availability] = exists
                        if not exists:
                            metadata[key] = None
                    metadata["out_path"] = metadata.get("document_path") or metadata.get("transcription_path")
                if job["status"] not in TERMINAL_STATUSES:
                    job.update(status="cancelled", finished_at=timestamp())
                    self._mark_unfinished(job, "cancelled")
                    job["logs"].append("Traitement interrompu lors de la fermeture précédente. Résultats conservés.")
                job["cancel_requested"] = False
                with self._lock:
                    if job["id"] in self._jobs:
                        continue
                    self._jobs[job["id"]] = job
                # The UUID folder contains derived conversion files only.
                shutil.rmtree(self.paths.temp / job["id"], ignore_errors=True)
            except (OSError, ValueError, KeyError, TypeError, AttributeError):
                logger.warning("Historique de traitement illisible : %s", manifest.parent.name)
        return self.history()
