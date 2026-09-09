"""Persistent, recoverable model downloads with cooperative cancellation."""
from __future__ import annotations

import fnmatch
import os
import shutil
import sys
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

from tqdm.auto import tqdm

from ..core.config import MODEL_APPROX_SIZE, REPO_MAP
from ..models.jobs import JobCancelled, JobError, ValidationError
from .files import write_text_atomic


def snapshot_download(*, repo_id: str, local_dir: str, allow_patterns: list[str], tqdm_class) -> None:
    """A sequential snapshot avoids Hugging Face's non-daemon download pool.

    Metadata pins every file to one commit. HTTP downloads still use the Hub's
    existing cache, integrity checks, incomplete files and resumption support.
    Bootstrap also sets this default before any Whisper/Hub imports; the lazy
    import keeps standalone service callers on the same cancellable HTTP path.
    """
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    from huggingface_hub import HfApi, hf_hub_download

    information = HfApi().model_info(repo_id, files_metadata=True, timeout=10)
    for sibling in information.siblings or []:
        filename = sibling.rfilename
        if any(fnmatch.fnmatchcase(filename, pattern) for pattern in allow_patterns):
            # Construction invokes the cancellation check before metadata/file I/O.
            with tqdm_class(total=0):
                hf_hub_download(repo_id=repo_id, filename=filename, revision=information.sha,
                                local_dir=local_dir, tqdm_class=tqdm_class, etag_timeout=10)


def directory_size(path: Path) -> int:
    try:
        return sum(p.stat().st_size for p in path.rglob("*") if p.is_file())
    except OSError:
        return 0


def snapshot_try_candidates(repo_or_list, local_dir: Path, cancel_check: Callable[[], None],
                            log: Callable[[str], None]) -> None:
    class JobAwareTqdm(tqdm):
        def __init__(self, *args, **kwargs):
            kwargs.pop("name", None)
            # Progress goes to the Qt view, never to a missing windowed stderr.
            kwargs["disable"] = True
            super().__init__(*args, **kwargs)
            cancel_check()

        def update(self, n=1):
            cancel_check()
            return super().update(n)

        def __iter__(self):
            for item in super().__iter__():
                cancel_check()
                yield item

    candidates = list(repo_or_list) if isinstance(repo_or_list, (list, tuple)) else [repo_or_list]
    last_error = None
    for repository in candidates:
        cancel_check()
        try:
            log(f"Téléchargement depuis {repository}…")
            snapshot_download(repo_id=repository, local_dir=str(local_dir),
                              allow_patterns=["config.json", "model.bin", "tokenizer.json", "vocabulary.*", "preprocessor_config.json"],
                              tqdm_class=JobAwareTqdm)
            cancel_check()
            return
        except JobCancelled:
            raise
        except Exception as exc:
            last_error = exc
            log(f"Échec du dépôt {repository} : {type(exc).__name__}. Nouvel essai si disponible.")
    raise JobError("Le téléchargement a échoué. Vérifiez la connexion et l'espace disque, puis réessayez.") from last_error


class ModelService:
    def __init__(self, paths):
        self.paths = paths
        self._lock = threading.Lock()

    def model_path(self, name: str) -> Path:
        if name not in REPO_MAP:
            raise ValidationError("Modèle local inconnu.")
        return self.paths.models / name

    @staticmethod
    def _complete(path: Path) -> bool:
        return all((path / name).is_file() and (path / name).stat().st_size > 0
                   for name in ("model.bin", "config.json", "tokenizer.json"))

    def available(self, name: str) -> bool:
        path = self.model_path(name)
        return (path / ".download_complete").is_file() and self._complete(path)

    def download(self, name: str, cancel_check: Callable[[], None] = lambda: None,
                 progress: Callable[[float], None] = lambda value: None,
                 log: Callable[[str], None] = lambda message: None) -> Path:
        target = self.model_path(name)
        while not self._lock.acquire(timeout=0.2):
            cancel_check()
        try:
            cancel_check()
            if self.available(name):
                progress(1.0)
                return target
            target.mkdir(parents=True, exist_ok=True)
            marker = target / ".download_complete"
            marker.unlink(missing_ok=True)
            # WHISPER_MODELS_DIR may already point at the checkout's models
            # directory, including via a junction/symlink. Adopting those files
            # must never open the sole original in write/truncate mode.
            legacy = Path(__file__).resolve().parents[3] / "models" / name
            legacy_ready = not getattr(sys, "frozen", False) and self._complete(legacy)
            if legacy_ready and target.samefile(legacy):
                cancel_check()
                write_text_atomic(marker, datetime.now(timezone.utc).isoformat())
                progress(1.0)
                log(f"Modèle {name} déjà présent dans le dossier configuré.")
                return target
            estimated = MODEL_APPROX_SIZE.get(name, 1024**3)
            needed = max(0, estimated - directory_size(target))
            if shutil.disk_usage(target).free < needed + 100 * 1024**2:
                raise JobError("Espace disque insuffisant pour télécharger ce modèle.")
            stop = threading.Event()

            def monitor() -> None:
                while not stop.wait(0.5):
                    progress(min(0.99, directory_size(target) / max(1, estimated)))

            monitor_thread = threading.Thread(target=monitor, name="model-progress", daemon=True)
            monitor_thread.start()
            try:
                # Migration from this source checkout only; never ship these models.
                if legacy_ready:
                    log(f"Import du modèle {name} déjà présent dans le projet…")
                    for source in legacy.iterdir():
                        if not source.is_file() or source.name not in {"model.bin", "config.json", "tokenizer.json", "vocabulary.txt", "vocabulary.json", "preprocessor_config.json"}:
                            continue
                        destination = target / source.name
                        if destination.exists() and source.samefile(destination):
                            continue
                        with source.open("rb") as reader, destination.open("wb") as writer:
                            while chunk := reader.read(1024 * 1024):
                                cancel_check()
                                writer.write(chunk)
                else:
                    snapshot_try_candidates(REPO_MAP[name], target, cancel_check, log)
                cancel_check()
                if not self._complete(target):
                    raise JobError("Le téléchargement du modèle est incomplet. Réessayez pour le reprendre.")
                write_text_atomic(marker, datetime.now(timezone.utc).isoformat())
                progress(1.0)
                log(f"Modèle {name} disponible.")
                return target
            finally:
                stop.set()
                monitor_thread.join(timeout=1)
        finally:
            self._lock.release()
