"""Encrypted review metadata, reversible grouping and transactional application."""
from __future__ import annotations

import json
import re
from copy import deepcopy
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from ..models.speaker import DiarizationResult, SpeakerMatch
from ..models.transcript import StructuredTranscript
from .dpapi import WindowsDpapiCodec
from .recording_sources import find_recording_sources
from .speaker_audio import AudioExcerpt, excerpts_for
from .speaker_persistence import SpeakerTransaction, atomic_bytes, digest, speaker_lock
from .speaker_profile_service import SpeakerProfileService
from .speaker_review_state import SpeakerReviewState

PENDING_BIOMETRIC_RETENTION_DAYS = 7


def _json_bytes(payload: dict[str, Any]) -> bytes:
    return (json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode("utf-8")


class SpeakerReviewStore:
    SCHEMA = 2

    def __init__(self, paths: Any, *, codec: Any | None = None) -> None:
        self.speakers = Path(paths.speakers)
        self.root = self.speakers / "pending"
        self.root.mkdir(parents=True, exist_ok=True)
        self.codec = codec if codec is not None else WindowsDpapiCodec()
        self.transaction = SpeakerTransaction(self.speakers, paths.results)
        self.transaction.recover()
        self.purge_expired()

    @staticmethod
    def _now() -> datetime:
        return datetime.now(timezone.utc)

    def _path(self, review_id: str) -> Path:
        if not re.fullmatch(r"[0-9a-fA-F-]{1,160}", review_id):
            raise ValueError("Identifiant de revue invalide.")
        path = self.root / f"{review_id}.dpapi"
        if not path.resolve().is_relative_to(self.root.resolve()):
            raise ValueError("Chemin de revue invalide.")
        return path

    def encode(self, payload: dict[str, Any]) -> bytes:
        body = {key: deepcopy(value) for key, value in payload.items() if not key.startswith("_")}
        body["schema_version"] = self.SCHEMA
        now = self._now().isoformat()
        body.setdefault("created_at", now)
        body["updated_at"] = now
        return self.codec.protect(_json_bytes(body))

    def save(self, review_id: str, payload: dict[str, Any]) -> None:
        with speaker_lock(self.speakers):
            self.transaction.recover()
            atomic_bytes(self._path(review_id), self.encode(payload))

    def load(self, review_id: str) -> dict[str, Any]:
        with speaker_lock(self.speakers):
            self.transaction.recover()
            target = self._path(review_id)
            if not target.is_file():
                raise FileNotFoundError("La revue des locuteurs n'est plus disponible.")
            try:
                payload = json.loads(self.codec.unprotect(target.read_bytes()).decode("utf-8"))
            except Exception as exc:
                raise RuntimeError("La revue chiffr\u00e9e des locuteurs est illisible.") from exc
            if not isinstance(payload, dict) or payload.get("schema_version") not in {1, self.SCHEMA}:
                raise RuntimeError("Version de revue des locuteurs non prise en charge.")
            try:
                created = datetime.fromisoformat(str(payload.get("created_at") or ""))
                if created.tzinfo is None:
                    created = created.replace(tzinfo=timezone.utc)
            except ValueError as exc:
                raise RuntimeError("Date de revue des locuteurs invalide.") from exc
            if (created < self._now() - timedelta(days=PENDING_BIOMETRIC_RETENTION_DAYS)
                    and not payload.get("biometric_samples_cleared")):
                self.clear_samples(payload)
                payload["biometric_samples_expired"] = True
                self.save(review_id, payload)
            return payload

    @staticmethod
    def clear_samples(payload: dict[str, Any]) -> None:
        for cluster in payload.get("clusters") or []:
            cluster["embeddings"] = []
            cluster["biometric_samples_available"] = False
        payload["biometric_samples_cleared"] = True

    def purge_expired(self) -> None:
        # Keep text-editing/group metadata after expiration, not biometric samples.
        for candidate in self.root.glob("*.dpapi"):
            try:
                self.load(candidate.stem)
            except (OSError, RuntimeError, ValueError):
                # Foreign/corrupt DPAPI reviews are never used. Remove only old ones.
                try:
                    age = self._now().timestamp() - candidate.stat().st_mtime
                    if age > PENDING_BIOMETRIC_RETENTION_DAYS * 86400:
                        with speaker_lock(self.speakers):
                            candidate.unlink(missing_ok=True)
                except OSError:
                    pass

    def delete(self, review_id: str) -> None:
        with speaker_lock(self.speakers):
            self.transaction.recover()
            self._path(review_id).unlink(missing_ok=True)


class SpeakerReviewService:
    def __init__(self, paths: Any, *, profiles: SpeakerProfileService | None = None,
                 store: SpeakerReviewStore | None = None) -> None:
        self.paths = paths
        self.profiles = profiles if profiles is not None else SpeakerProfileService(paths)
        self.store = store if store is not None else SpeakerReviewStore(paths)
        self.transaction = SpeakerTransaction(paths.speakers, paths.results)
        self.transaction.recover()

    def _paths(self, payload: dict[str, Any]) -> tuple[Path, Path]:
        job_id = str(payload["job_id"])
        if not re.fullmatch(r"[0-9a-f]{32}", job_id):
            raise ValueError("Identifiant de traitement invalide.")
        root = Path(self.paths.results).resolve()
        job_dir = (root / job_id).resolve()
        if not job_dir.is_dir() or not job_dir.is_relative_to(root):
            raise RuntimeError("Le dossier de r\u00e9sultat n'existe plus ou son chemin est invalide.")
        output = []
        for key in ("structured_filename", "transcript_filename"):
            name = str(payload[key])
            if not name or name in {".", ".."} or any(char in name for char in "/\\"):
                raise ValueError("Nom de r\u00e9sultat invalide.")
            path = (job_dir / name).resolve()
            if not path.is_relative_to(job_dir):
                raise ValueError("Chemin de revue invalide.")
            output.append(path)
        if output[0] == output[1]:
            raise ValueError("Le texte et sa structure doivent \u00eatre deux fichiers distincts.")
        return output[0], output[1]

    def _audio_sources(self, source: Path | None) -> dict[str, Any]:
        if source is None:
            return {}
        source = Path(source).resolve()
        sources = {"mixed": {"path": str(source), "offset": 0.0}}
        native = find_recording_sources(self.paths, source)
        if native is not None:
            if native.microphone_path:
                sources["self"] = {"path": str(native.microphone_path), "offset": native.microphone_offset}
            if native.system_path:
                sources["system"] = {"path": str(native.system_path), "offset": native.system_offset}
        return sources

    def create_review(
        self, *, review_id: str, job_id: str, file_index: int, diarization: DiarizationResult,
        labels: dict[str, str], matches: dict[str, SpeakerMatch], structured_filename: str,
        transcript_filename: str, source_audio: Path | None = None,
    ) -> None:
        clusters = []
        first_seen = {}
        for turn in diarization.turns:
            first_seen[turn.cluster_id] = min(first_seen.get(turn.cluster_id, turn.start), turn.start)
        for cluster in sorted(diarization.clusters, key=lambda item: first_seen.get(item.cluster_id, float("inf"))):
            match = matches.get(cluster.cluster_id)
            clusters.append({
                **cluster.to_dict(), "current_name": labels.get(cluster.cluster_id, cluster.cluster_id),
                "auto_profile_id": match.profile_id if match else None,
                "auto_name": match.name if match else None, "auto_score": match.score if match else 0.0,
                "auto_margin": match.margin if match else 0.0, "auto_accepted": bool(match and match.accepted),
                "auto_reason": match.reason if match else "",
            })
        payload = {
            "job_id": job_id, "file_index": int(file_index), "model_set_id": diarization.model_set_id,
            "structured_filename": structured_filename, "transcript_filename": transcript_filename,
            "clusters": clusters, "turns": [turn.to_dict() for turn in diarization.turns],
            "audio_sources": self._audio_sources(source_audio),
        }
        self._paths(payload)
        SpeakerReviewState(payload)  # Fail before writing malformed cluster data.
        self.store.save(review_id, payload)

    def transcript(self, review_id: str) -> StructuredTranscript:
        payload = self.store.load(review_id)
        structured, _ = self._paths(payload)
        return StructuredTranscript.from_dict(json.loads(structured.read_text(encoding="utf-8")))

    def load_review(self, review_id: str) -> dict[str, Any]:
        with speaker_lock(Path(self.paths.speakers)):
            payload = self.store.load(review_id)
            structured_path, transcript_path = self._paths(payload)
            structured_bytes = structured_path.read_bytes()
            structured = StructuredTranscript.from_dict(json.loads(structured_bytes.decode("utf-8")))
            # Upgrade legacy 1.1 reviews in memory; no source path was saved then.
            if not payload.get("turns"):
                payload["turns"] = [{
                    "start": segment.start, "end": segment.end,
                    "cluster_id": segment.speaker_original_key or segment.speaker_key,
                    "source": segment.speaker_source or "mixed",
                } for segment in structured.segments if segment.speaker_key]
            if not payload.get("audio_sources"):
                try:
                    manifest = json.loads((structured_path.parent / "job.json").read_text(encoding="utf-8"))
                    name = str(manifest["files"][int(payload["file_index"])]["name"])
                    if re.fullmatch(r"native_recording_[0-9a-f]{32}\.wav", name):
                        payload["audio_sources"] = self._audio_sources(Path(self.paths.results) / name)
                except (OSError, ValueError, KeyError, IndexError, TypeError):
                    pass
            payload["_revision"] = digest(self.store._path(review_id).read_bytes())
            payload["_transcript_revision"] = digest(structured_bytes + b"\0" + transcript_path.read_bytes())
            payload["_has_generated_document"] = any(
                path.suffix.lower() == ".txt" and path != transcript_path for path in structured_path.parent.iterdir()
            )
            return payload

    def excerpts(self, payload: dict[str, Any], members: tuple[str, ...],
                 *, source_override: Path | None = None) -> list[AudioExcerpt]:
        return excerpts_for(payload, members, source_override=source_override)

    def apply(
        self, review_id: str, names: dict[str, str], remember: set[str], *, self_speaker_name: str,
        groups: list[dict[str, Any]] | None = None, expected_revision: str | None = None,
        expected_transcript_revision: str | None = None, source_override: Path | None = None,
    ) -> Path:
        with speaker_lock(Path(self.paths.speakers)):
            self.transaction.recover()
            payload = self.load_review(review_id)
            if expected_revision is not None and payload["_revision"] != expected_revision:
                raise RuntimeError("Cette revue a chang\u00e9 dans une autre fen\u00eatre. Rouvrez-la avant de valider.")
            if (expected_transcript_revision is not None
                    and payload["_transcript_revision"] != expected_transcript_revision):
                raise RuntimeError("La transcription a \u00e9t\u00e9 modifi\u00e9e entre-temps. Rouvrez la revue.")
            state = SpeakerReviewState(payload)
            if groups is not None:
                state.set_groups(groups)
            state.set_choices(names, remember)
            structured_path, transcript_path = self._paths(payload)
            structured = StructuredTranscript.from_dict(json.loads(structured_path.read_text(encoding="utf-8")))
            updated = state.preview(structured)
            enrollments = []
            for group in state.ordered_groups():
                if group.remember:
                    embeddings = state.embeddings(group.group_id)
                    if not embeddings:
                        raise ValueError(f"Aucune empreinte exploitable pour {group.name}. Relancez la diarisation.")
                    enrollments.append({
                        "name": group.name, "model_set_id": str(payload["model_set_id"]),
                        "embeddings": embeddings,
                        "is_self": group.name.casefold() == self_speaker_name.strip().casefold(),
                    })
            # All name/model/dimension/consent validation and encryption precede ANY write.
            updates = {}
            if enrollments:
                updates[self.profiles.path] = self.profiles.prepare_enrollments(enrollments)[0]
            payload["groups"] = state.serialized_groups()
            mapping = {member: group for group in state.groups.values() for member in group.members}
            for cluster in payload["clusters"]:
                cluster["current_name"] = mapping[cluster["cluster_id"]].name
            if source_override is not None:
                path = Path(source_override).resolve()
                if not path.is_file():
                    raise FileNotFoundError("L'audio choisi est introuvable.")
                payload["audio_sources"] = self._audio_sources(path)
            SpeakerReviewStore.clear_samples(payload)
            payload["reviewed_at"] = self.store._now().isoformat()
            updates[structured_path] = _json_bytes(updated.to_dict())
            text = updated.speaker_text()
            updates[transcript_path] = (text + ("\n" if text else "")).encode("utf-8")
            updates[self.store._path(review_id)] = self.store.encode(payload)
            self.transaction.commit(updates)
            return transcript_path
