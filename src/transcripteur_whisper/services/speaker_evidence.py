"""Verifiable local evidence for stored speaker embeddings.

Profiles keep their biometric vectors in the existing DPAPI store. This sidecar binds a
stable fingerprint of each vector to a human-review status and, when available, an encrypted
WAV excerpt that can be replayed later. In strict mode, only human-approved GOOD embeddings
with replayable audio are allowed to participate in automatic identity matching.
"""
from __future__ import annotations

import hashlib
import io
import json
import math
import re
import wave
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from .dpapi import WindowsDpapiCodec
from .speaker_audio import AudioExcerpt
from .speaker_persistence import SpeakerTransaction, atomic_bytes, speaker_lock

EVIDENCE_SCHEMA_VERSION = 1
EVIDENCE_STATUSES = {"UNVERIFIED", "GOOD", "BAD", "UNSURE", "QUARANTINED"}
MATCHABLE_STATUSES = {"GOOD"}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def embedding_key(vector: Iterable[float]) -> str:
    value = np.asarray(tuple(vector), dtype=np.float32)
    if value.ndim != 1 or value.size < 16 or not np.isfinite(value).all():
        raise ValueError("Empreinte vocale invalide.")
    norm = float(np.linalg.norm(value))
    if not math.isfinite(norm) or norm <= 1e-8:
        raise ValueError("Empreinte vocale vide.")
    value = np.ascontiguousarray(value / norm, dtype=np.float32)
    return hashlib.sha256(value.tobytes()).hexdigest()


def _wav_bytes(samples: np.ndarray, sample_rate: int) -> bytes:
    samples = np.asarray(samples, dtype=np.float32).reshape(-1)
    if not samples.size or sample_rate <= 0:
        raise ValueError("Extrait audio vide.")
    samples = np.clip(np.nan_to_num(samples, nan=0.0, posinf=0.0, neginf=0.0), -1.0, 1.0)
    pcm = np.asarray(np.rint(samples * 32767.0), dtype="<i2")
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(int(sample_rate))
        output.writeframes(pcm.tobytes())
    return buffer.getvalue()


@dataclass(frozen=True)
class EvidenceRecordRequest:
    profile_id: str
    embedding: tuple[float, ...] | list[float]
    status: str
    excerpt: AudioExcerpt | None = None
    metadata: dict[str, Any] | None = None


class SpeakerEvidenceStore:
    def __init__(self, paths: Any, *, codec: Any | None = None) -> None:
        self.speakers = Path(paths.speakers).resolve()
        self.speakers.mkdir(parents=True, exist_ok=True)
        self.path = self.speakers / "profile_evidence.v1.dpapi"
        self.audio_root = self.speakers / "evidence"
        self.codec = codec if codec is not None else WindowsDpapiCodec()
        self.transaction = SpeakerTransaction(self.speakers, paths.results)
        self.transaction.recover()

    def _empty(self) -> dict[str, Any]:
        return {
            "schema_version": EVIDENCE_SCHEMA_VERSION,
            "strict_verification": False,
            "profiles": {},
        }

    def _load_payload(self) -> dict[str, Any]:
        if not self.path.is_file():
            return self._empty()
        try:
            payload = json.loads(self.codec.unprotect(self.path.read_bytes()).decode("utf-8"))
        except Exception as exc:
            raise RuntimeError("La base de preuves vocales est illisible.") from exc
        if (
            not isinstance(payload, dict)
            or payload.get("schema_version") != EVIDENCE_SCHEMA_VERSION
            or not isinstance(payload.get("profiles"), dict)
        ):
            raise RuntimeError("Base de preuves vocales invalide.")
        return payload

    def _encode(self, payload: dict[str, Any]) -> bytes:
        return self.codec.protect(json.dumps(
            payload, ensure_ascii=False, separators=(",", ":"), allow_nan=False
        ).encode("utf-8"))

    def strict_active(self) -> bool:
        with speaker_lock(self.speakers):
            self.transaction.recover()
            return bool(self._load_payload().get("strict_verification"))

    def _audio_path(self, profile_id: str, key: str) -> Path:
        if not re.fullmatch(r"[0-9a-f]{32}", profile_id) or not re.fullmatch(r"[0-9a-f]{64}", key):
            raise ValueError("Identifiant de preuve vocale invalide.")
        path = (self.audio_root / profile_id / f"{key}.wav.dpapi").resolve()
        if not path.is_relative_to(self.audio_root.resolve()):
            raise ValueError("Chemin de preuve vocale invalide.")
        return path

    def _record_for(self, payload: dict[str, Any], profile_id: str, key: str) -> dict[str, Any] | None:
        profile = payload.get("profiles", {}).get(profile_id)
        if not isinstance(profile, dict):
            return None
        records = profile.get("embeddings")
        if not isinstance(records, dict):
            return None
        record = records.get(key)
        return record if isinstance(record, dict) else None

    def eligible_vectors(
        self, profile_id: str, vectors: Iterable[Iterable[float]],
    ) -> list[tuple[float, ...]]:
        values = [tuple(float(x) for x in vector) for vector in vectors]
        with speaker_lock(self.speakers):
            self.transaction.recover()
            payload = self._load_payload()
            if not payload.get("strict_verification"):
                return values
            output: list[tuple[float, ...]] = []
            for vector in values:
                key = embedding_key(vector)
                record = self._record_for(payload, profile_id, key)
                if not record or record.get("status") not in MATCHABLE_STATUSES:
                    continue
                try:
                    audio = self._audio_path(profile_id, key)
                except ValueError:
                    continue
                if audio.is_file():
                    output.append(vector)
            return output

    def prepare_quarantine(self, profiles: Iterable[Any]) -> bytes:
        """Enable strict mode and quarantine every legacy vector lacking replayable evidence."""
        payload = deepcopy(self._load_payload())
        payload["strict_verification"] = True
        now = _now()
        for profile in profiles:
            entry = payload["profiles"].setdefault(profile.profile_id, {
                "name": profile.name,
                "model_set_id": profile.model_set_id,
                "embeddings": {},
            })
            entry["name"] = profile.name
            entry["model_set_id"] = profile.model_set_id
            records = entry.setdefault("embeddings", {})
            for vector in profile.embeddings:
                key = embedding_key(vector)
                existing = records.get(key)
                if isinstance(existing, dict):
                    # Never downgrade a replayable/manual record.
                    continue
                records[key] = {
                    "status": "QUARANTINED",
                    "reason": "legacy_without_replayable_audio",
                    "created_at": now,
                    "updated_at": now,
                }
        return self._encode(payload)

    @staticmethod
    def _read_clip(excerpt: AudioExcerpt, cache: dict[Path, tuple[np.ndarray, int]]) -> tuple[bytes, float]:
        from .diarization_worker import _decode_audio

        source = Path(excerpt.path).resolve()
        if source not in cache:
            cache[source] = _decode_audio(source)
        samples, sample_rate = cache[source]
        left = max(0, round(float(excerpt.start) * sample_rate))
        right = min(samples.size, round(float(excerpt.end) * sample_rate))
        if right <= left:
            raise ValueError("Extrait audio hors limites.")
        clip = samples[left:right]
        if not clip.size or float(np.max(np.abs(clip))) < 1e-5:
            raise ValueError("Extrait audio silencieux.")
        return _wav_bytes(clip, sample_rate), clip.size / sample_rate

    def prepare_records(
        self, requests: Iterable[EvidenceRecordRequest],
    ) -> tuple[bytes, dict[Path, bytes]]:
        """Prepare encrypted metadata and audio bytes without publishing any file."""
        payload = deepcopy(self._load_payload())
        payload["strict_verification"] = True
        cache: dict[Path, tuple[np.ndarray, int]] = {}
        audio_updates: dict[Path, bytes] = {}

        for request in requests:
            status = str(request.status).upper()
            if status not in EVIDENCE_STATUSES:
                raise ValueError("Statut de preuve vocale invalide.")
            key = embedding_key(request.embedding)
            profile = payload["profiles"].setdefault(request.profile_id, {
                "name": "",
                "model_set_id": "",
                "embeddings": {},
            })
            records = profile.setdefault("embeddings", {})
            previous = records.get(key) if isinstance(records.get(key), dict) else {}
            now = _now()
            record = dict(previous)
            record.update({
                "status": status,
                "updated_at": now,
            })
            record.setdefault("created_at", now)

            if request.metadata:
                for field in (
                    "profile_name", "model_set_id", "review_id", "source", "source_name",
                    "historical_name", "link_similarity", "actual_name", "note",
                ):
                    if field in request.metadata:
                        value = request.metadata[field]
                        if value is not None:
                            record[field] = value
                if request.metadata.get("profile_name"):
                    profile["name"] = str(request.metadata["profile_name"])
                if request.metadata.get("model_set_id"):
                    profile["model_set_id"] = str(request.metadata["model_set_id"])

            if request.excerpt is not None:
                wav, duration = self._read_clip(request.excerpt, cache)
                audio_path = self._audio_path(request.profile_id, key)
                audio_updates[audio_path] = self.codec.protect(wav)
                record.update({
                    "audio_file": str(audio_path.relative_to(self.speakers)).replace("\\", "/"),
                    "audio_duration": float(duration),
                    "audio_sha256": hashlib.sha256(wav).hexdigest(),
                })

            # Only GOOD evidence can identify a person, and it must remain replayable.
            if status in MATCHABLE_STATUSES:
                audio_path = self._audio_path(request.profile_id, key)
                if audio_path not in audio_updates and not audio_path.is_file():
                    raise ValueError(
                        "Une empreinte GOOD doit avoir un extrait audio vérifiable."
                    )
            records[key] = record

        return self._encode(payload), audio_updates

    def save_records(self, requests: Iterable[EvidenceRecordRequest]) -> None:
        with speaker_lock(self.speakers):
            self.transaction.recover()
            protected, audio_updates = self.prepare_records(requests)
            updates: dict[Path, bytes | None] = {self.path: protected}
            updates.update(audio_updates)
            self.transaction.commit(updates)

    def save_quarantine(self, profiles: Iterable[Any]) -> None:
        with speaker_lock(self.speakers):
            self.transaction.recover()
            atomic_bytes(self.path, self.prepare_quarantine(profiles))

    def record(self, profile_id: str, vector: Iterable[float]) -> dict[str, Any] | None:
        key = embedding_key(vector)
        with speaker_lock(self.speakers):
            self.transaction.recover()
            payload = self._load_payload()
            record = self._record_for(payload, profile_id, key)
            return deepcopy(record) if record else None

    def materialize_audio(
        self, profile_id: str, vector: Iterable[float], destination: Path,
    ) -> AudioExcerpt | None:
        """Decrypt one verification clip to a caller-owned temporary WAV for playback."""
        key = embedding_key(vector)
        with speaker_lock(self.speakers):
            self.transaction.recover()
            payload = self._load_payload()
            record = self._record_for(payload, profile_id, key)
            if not record:
                return None
            audio_path = self._audio_path(profile_id, key)
            if not audio_path.is_file():
                return None
            try:
                wav = self.codec.unprotect(audio_path.read_bytes())
            except Exception as exc:
                raise RuntimeError("Extrait de vérification vocal illisible.") from exc
            if record.get("audio_sha256") and hashlib.sha256(wav).hexdigest() != record["audio_sha256"]:
                raise RuntimeError("Extrait de vérification vocal corrompu.")
            destination = Path(destination).resolve()
            destination.parent.mkdir(parents=True, exist_ok=True)
            atomic_bytes(destination, wav)
            duration = float(record.get("audio_duration") or 0.0)
            if not math.isfinite(duration) or duration <= 0:
                duration = 6.0
            return AudioExcerpt(destination, 0.0, duration, 0.0, key[:12], True, False)

    def profile_files(self, profile_id: str) -> list[Path]:
        try:
            folder = (self.audio_root / profile_id).resolve()
        except OSError:
            return []
        if not folder.is_relative_to(self.audio_root.resolve()) or not folder.is_dir():
            return []
        return [item.resolve() for item in folder.glob("*.wav.dpapi") if item.is_file()]
