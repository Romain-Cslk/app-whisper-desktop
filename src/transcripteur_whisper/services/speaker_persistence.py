"""Locked, recoverable updates of speaker profiles, review metadata and text.

The journal never contains clear voice embeddings: callers supply DPAPI bytes.
A prepared transaction is rolled back after a crash; a committed one is cleaned.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
import threading
import time
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator, Mapping

_LOCKS: dict[str, threading.RLock] = {}
_LOCKS_GUARD = threading.Lock()
_DEPTH = threading.local()


def digest(data: bytes | None) -> str | None:
    return hashlib.sha256(data).hexdigest() if data is not None else None


def read_optional(path: Path) -> bytes | None:
    try:
        return path.read_bytes()
    except FileNotFoundError:
        return None


def atomic_bytes(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


@contextmanager
def speaker_lock(directory: Path, timeout: float = 10.0) -> Iterator[None]:
    """Reentrant thread lock plus OS lock; the OS releases it on process exit."""
    directory = Path(directory).resolve()
    directory.mkdir(parents=True, exist_ok=True)
    key = str(directory)
    with _LOCKS_GUARD:
        lock = _LOCKS.setdefault(key, threading.RLock())
    with lock:
        depths = getattr(_DEPTH, "values", None)
        if depths is None:
            depths = _DEPTH.values = {}
        if depths.get(key, 0):
            depths[key] += 1
            try:
                yield
            finally:
                depths[key] -= 1
            return
        stream = (directory / ".speaker-store.lock").open("a+b")
        locked = False
        try:
            stream.seek(0, os.SEEK_END)
            if stream.tell() == 0:
                stream.write(b"0")
                stream.flush()
            until = time.monotonic() + timeout
            while not locked:
                try:
                    stream.seek(0)
                    if os.name == "nt":
                        import msvcrt
                        msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
                    else:
                        import fcntl
                        fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                    locked = True
                except OSError as exc:
                    if time.monotonic() >= until:
                        raise RuntimeError("La base des locuteurs est occup\u00e9e. R\u00e9essayez.") from exc
                    time.sleep(0.05)
            depths[key] = 1
            try:
                yield
            finally:
                depths.pop(key, None)
        finally:
            if locked:
                stream.seek(0)
                if os.name == "nt":
                    import msvcrt
                    msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
                else:
                    import fcntl
                    fcntl.flock(stream.fileno(), fcntl.LOCK_UN)
            stream.close()


class SpeakerTransaction:
    """Application-level atomicity, with recovery for interrupted multi-file writes.

    Public entry points take the same speaker lock. Files edited outside the app
    are never silently overwritten during rollback/recovery.
    """

    def __init__(self, speakers: Path, results: Path) -> None:
        self.speakers = Path(speakers).resolve()
        self.results = Path(results).resolve()
        self.root = self.speakers / "transactions"

    def _target(self, path: Path) -> Path:
        target = Path(path).resolve()
        if target.is_relative_to(self.root):
            raise ValueError("Une transaction ne peut pas cibler son propre journal.")
        if target in (self.speakers, self.results) or not (
            target.is_relative_to(self.speakers) or target.is_relative_to(self.results)
        ):
            raise ValueError("Chemin de transaction des locuteurs invalide.")
        return target

    def _rollback(self, folder: Path, manifest: dict) -> None:
        entries = manifest.get("entries")
        if not isinstance(entries, list) or not entries:
            raise RuntimeError("Journal de transaction invalide.")
        # Validate everything before writing anything on recovery.
        pending = []
        for index, entry in enumerate(entries):
            target = self._target(Path(entry["target"]))
            current = read_optional(target)
            if digest(current) not in {entry["before_hash"], entry["after_hash"]}:
                raise RuntimeError(
                    f"R\u00e9cup\u00e9ration suspendue : {target.name} a \u00e9t\u00e9 modifi\u00e9 hors de l'application. "
                    f"Journal conserv\u00e9 dans {folder}."
                )
            before = (folder / f"{index}.before").read_bytes() if entry["existed"] else None
            if digest(before) != entry["before_hash"]:
                raise RuntimeError("Sauvegarde de transaction corrompue ; aucun \u00e9crasement effectu\u00e9.")
            pending.append((target, before))
        for target, before in reversed(pending):
            if before is None:
                target.unlink(missing_ok=True)
            elif read_optional(target) != before:
                atomic_bytes(target, before)

    def recover(self) -> None:
        with speaker_lock(self.speakers):
            if not self.root.is_dir():
                return
            for folder in sorted(self.root.iterdir()):
                if not folder.is_dir() or folder.is_symlink():
                    continue
                manifest_path = folder / "manifest.json"
                if not manifest_path.is_file():
                    # Nothing was published before the manifest existed.
                    shutil.rmtree(folder)
                    continue
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                if manifest.get("schema_version") != 1:
                    raise RuntimeError("Version de journal de transaction inconnue.")
                status = manifest.get("status")
                if status == "prepared":
                    self._rollback(folder, manifest)
                elif status != "committed":
                    raise RuntimeError("\u00c9tat du journal de transaction inconnu.")
                shutil.rmtree(folder)

    def commit(self, updates: Mapping[Path, bytes | None]) -> None:
        if not updates:
            return
        with speaker_lock(self.speakers):
            self.recover()
            normalized = [(self._target(path), data) for path, data in updates.items()]
            if len({p for p, _ in normalized}) != len(normalized):
                raise ValueError("Une transaction cible deux fois le m\u00eame fichier.")
            if any(data is not None and not isinstance(data, bytes) for _, data in normalized):
                raise TypeError("Les donn\u00e9es de transaction doivent \u00eatre des octets.")
            folder = self.root / uuid.uuid4().hex
            folder.mkdir(parents=True, exist_ok=False)
            manifest = {"schema_version": 1, "status": "prepared", "entries": []}
            try:
                for index, (target, data) in enumerate(normalized):
                    before = read_optional(target)
                    if before is not None:
                        atomic_bytes(folder / f"{index}.before", before)
                    if data is not None:
                        atomic_bytes(folder / f"{index}.after", data)
                    manifest["entries"].append({
                        "target": str(target), "existed": before is not None,
                        "before_hash": digest(before), "after_hash": digest(data),
                    })
                atomic_bytes(folder / "manifest.json", json.dumps(manifest).encode("utf-8"))
                for index, (target, _data) in enumerate(normalized):
                    if digest(read_optional(target)) != manifest["entries"][index]["before_hash"]:
                        raise RuntimeError(f"{target.name} a chang\u00e9 pendant la validation. Rouvrez la revue.")
                    if _data is None:
                        target.unlink(missing_ok=True)
                    else:
                        atomic_bytes(target, (folder / f"{index}.after").read_bytes())
                manifest["status"] = "committed"
                atomic_bytes(folder / "manifest.json", json.dumps(manifest).encode("utf-8"))
            except Exception:
                if (folder / "manifest.json").is_file():
                    self._rollback(folder, manifest)
                shutil.rmtree(folder)
                raise
            # A cleanup error must not invite the user to replay a committed enrollment.
            try:
                shutil.rmtree(folder)
            except OSError:
                pass
