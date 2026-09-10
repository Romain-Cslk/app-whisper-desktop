"""Encrypted, multimodal voice profiles; automatic matching is always read-only."""
from __future__ import annotations

import json
import math
import unicodedata
import uuid
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from ..models.speaker import SpeakerCluster, SpeakerMatch, SpeakerProfile
from .dpapi import WindowsDpapiCodec
from .speaker_persistence import SpeakerTransaction, atomic_bytes, speaker_lock

PROFILE_SCHEMA_VERSION = 1  # Existing DPAPI profiles remain readable.
DEFAULT_AUTO_THRESHOLD = 0.76
DEFAULT_AUTO_MARGIN = 0.04
MIN_AUTO_CLUSTER_SECONDS = 5.0
MIN_AUTO_QUERY_EMBEDDINGS = 2
MIN_PROFILE_EMBEDDINGS_FOR_AUTO = 2
MAX_PROFILE_EMBEDDINGS = 64
MAX_NAME_LENGTH = 80
AUTO_REFERENCE_TOP_K = 3
AUTO_REQUIRED_SUPPORT_RATIO = 0.75
AUTO_SAMPLE_FLOOR_DELTA = 0.06
PROFILE_TRUST_MIN_EMBEDDINGS = 8
PROFILE_TRUST_KEEP_RATIO = 0.80
ENROLLMENT_GUARD_MIN_PROFILE_EMBEDDINGS = 6
ENROLLMENT_GUARD_MIN_INCOMING_EMBEDDINGS = 2
ENROLLMENT_SIMILARITY_FLOOR = 0.68
ENROLLMENT_REQUIRED_SUPPORT_RATIO = 0.60


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _normalise(vector: Iterable[float]) -> np.ndarray:
    value = np.asarray(tuple(vector), dtype=np.float32)
    if value.ndim != 1 or value.size < 16 or not np.isfinite(value).all():
        raise ValueError("Empreinte vocale invalide.")
    norm = float(np.linalg.norm(value))
    if not math.isfinite(norm) or norm <= 1e-8:
        raise ValueError("Empreinte vocale vide.")
    return value / norm


def _normalised_tuple(vector: Iterable[float]) -> tuple[float, ...]:
    return tuple(float(v) for v in _normalise(vector))


def _validated_vectors(embeddings: Iterable[Iterable[float]]) -> list[tuple[float, ...]]:
    vectors = [_normalised_tuple(value) for value in embeddings]
    if not vectors:
        raise ValueError("Aucune empreinte vocale exploitable à mémoriser.")
    if any(len(value) != len(vectors[0]) for value in vectors):
        raise ValueError("Dimensions d'empreintes vocales incohérentes.")
    return vectors


def _topk_reference_score(query: np.ndarray, references: np.ndarray, k: int = AUTO_REFERENCE_TOP_K) -> np.ndarray:
    """Return one score per query using several close references, never a single lucky maximum.

    Taking the maximum over a large profile bank biases identification towards profiles
    containing many embeddings and lets one contaminated reference dominate. Averaging
    the nearest few references makes the score depend on repeated support instead.
    """
    similarities = np.clip(query @ references.T, -1.0, 1.0)
    take = min(max(1, int(k)), references.shape[0])
    if take == references.shape[0]:
        return similarities.mean(axis=1)
    nearest = np.partition(similarities, similarities.shape[1] - take, axis=1)[:, -take:]
    return nearest.mean(axis=1)


def _trusted_reference_bank(vectors: list[tuple[float, ...]]) -> list[tuple[float, ...]]:
    """Ignore isolated legacy samples at match time without deleting stored biometric data.

    Dense profiles may already contain a few wrong or acoustically broken samples. We keep
    all stored vectors intact, but automatic recognition uses the 80% with the strongest
    local support inside that profile. Sparse profiles are left untouched.
    """
    if len(vectors) < PROFILE_TRUST_MIN_EMBEDDINGS:
        return vectors
    matrix = np.asarray(vectors, dtype=np.float32)
    similarity = np.clip(matrix @ matrix.T, -1.0, 1.0)
    np.fill_diagonal(similarity, -2.0)
    take = min(AUTO_REFERENCE_TOP_K, len(vectors) - 1)
    nearest = np.partition(similarity, similarity.shape[1] - take, axis=1)[:, -take:]
    support = nearest.mean(axis=1)
    keep_count = max(PROFILE_TRUST_MIN_EMBEDDINGS, int(math.ceil(len(vectors) * PROFILE_TRUST_KEEP_RATIO)))
    keep_count = min(keep_count, len(vectors))
    order = np.argsort(-support, kind="stable")[:keep_count]
    return [vectors[int(index)] for index in sorted(order)]


def _diverse_bank(vectors: list[tuple[float, ...]], limit: int = MAX_PROFILE_EMBEDDINGS):
    """Bound storage while preserving legitimate acoustic variation.

    Matching no longer trusts every stored vector equally: _trusted_reference_bank filters
    isolated samples before identification. Storage therefore remains backwards-compatible
    and keeps confirmed variation until the hard capacity is reached.
    """
    if len(vectors) <= limit:
        return vectors
    matrix = np.asarray(vectors, dtype=np.float32)
    chosen = [0]
    nearest = np.clip(matrix @ matrix[0], -1.0, 1.0)
    nearest[0] = 2.0
    while len(chosen) < limit:
        index = int(np.argmin(nearest))
        chosen.append(index)
        nearest = np.maximum(nearest, np.clip(matrix @ matrix[index], -1.0, 1.0))
        nearest[chosen] = 2.0
    return [vectors[index] for index in sorted(chosen)]


def _validate_enrollment_against_existing(
    name: str,
    current: list[tuple[float, ...]],
    incoming: list[tuple[float, ...]],
) -> None:
    """Reject enrichment that would likely contaminate an established voice profile."""
    if (
        len(current) < ENROLLMENT_GUARD_MIN_PROFILE_EMBEDDINGS
        or len(incoming) < ENROLLMENT_GUARD_MIN_INCOMING_EMBEDDINGS
    ):
        return
    trusted = _trusted_reference_bank(current)
    references = np.asarray(trusted, dtype=np.float32)
    query = np.asarray(incoming, dtype=np.float32)
    if references.shape[1] != query.shape[1]:
        raise ValueError("Dimension incompatible avec le profil vocal existant.")
    scores = _topk_reference_score(query, references)
    required = max(2, int(math.ceil(len(scores) * ENROLLMENT_REQUIRED_SUPPORT_RATIO)))
    supported = int(np.sum(scores >= ENROLLMENT_SIMILARITY_FLOOR))
    if supported < required:
        raise ValueError(
            f"Les nouvelles empreintes ne sont pas assez cohérentes avec le profil {name} "
            f"({supported}/{len(scores)} extraits compatibles, minimum {required}). "
            "La mémorisation est refusée pour éviter de contaminer ce profil."
        )


class SpeakerProfileService:
    def __init__(self, paths: Any, *, codec: Any | None = None) -> None:
        self.directory = Path(paths.speakers)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.path = self.directory / "profiles.v1.dpapi"
        self.codec = codec if codec is not None else WindowsDpapiCodec()
        self.transaction = SpeakerTransaction(self.directory, paths.results)
        self.transaction.recover()

    @staticmethod
    def validate_name(name: str) -> str:
        if not isinstance(name, str):
            raise ValueError("Le nom du locuteur doit être un texte.")
        if any(unicodedata.category(char) in {"Cc", "Cf"} for char in name):
            raise ValueError("Le nom du locuteur contient un caractère de contrôle.")
        clean = " ".join(name.strip().split())
        if not clean:
            raise ValueError("Le nom du locuteur est vide.")
        if len(clean) > MAX_NAME_LENGTH:
            raise ValueError(f"Le nom du locuteur est limité à {MAX_NAME_LENGTH} caractères.")
        return clean

    def _load_payload(self) -> dict[str, Any]:
        if not self.path.exists():
            return {"schema_version": PROFILE_SCHEMA_VERSION, "profiles": []}
        try:
            payload = json.loads(self.codec.unprotect(self.path.read_bytes()).decode("utf-8"))
        except Exception as exc:
            raise RuntimeError("La base chiffrée des voix connues est illisible.") from exc
        if not isinstance(payload, dict) or payload.get("schema_version") != PROFILE_SCHEMA_VERSION:
            raise RuntimeError("Version de base des voix connues non prise en charge.")
        if not isinstance(payload.get("profiles"), list):
            raise RuntimeError("Base des voix connues invalide.")
        seen_ids, seen_names = set(), set()
        try:
            for item in payload["profiles"]:
                profile = SpeakerProfile.from_dict(item)
                name = self.validate_name(profile.name)
                if profile.profile_id in seen_ids or name.casefold() in seen_names or not profile.model_set_id:
                    raise ValueError("Identité de profil dupliquée ou invalide.")
                _validated_vectors(profile.embeddings)
                seen_ids.add(profile.profile_id)
                seen_names.add(name.casefold())
        except (TypeError, KeyError, ValueError) as exc:
            raise RuntimeError("Un profil vocal est corrompu ; aucune donnée n'a été écrasée.") from exc
        return payload

    def _encode(self, payload: dict[str, Any]) -> bytes:
        return self.codec.protect(json.dumps(
            payload, ensure_ascii=False, separators=(",", ":"), allow_nan=False
        ).encode("utf-8"))

    def _save_payload(self, payload: dict[str, Any]) -> None:
        atomic_bytes(self.path, self._encode(payload))

    def list_profiles(self) -> tuple[SpeakerProfile, ...]:
        with speaker_lock(self.directory):
            self.transaction.recover()
            profiles = [SpeakerProfile.from_dict(item) for item in self._load_payload()["profiles"]]
        return tuple(sorted(profiles, key=lambda item: item.name.casefold()))

    def prepare_enrollments(self, requests: Iterable[dict[str, Any]]) -> tuple[bytes, tuple[SpeakerProfile, ...]]:
        """Validate/encrypt the ENTIRE batch without changing any persistent file.

        Caller must hold speaker_lock until its transaction commits these bytes.
        """
        payload = deepcopy(self._load_payload())
        enrolled = []
        for request in requests:
            name = self.validate_name(request["name"])
            model = str(request["model_set_id"])
            if not model or len(model) > 200:
                raise ValueError("Identifiant de modèle vocal invalide.")
            incoming = _validated_vectors(request["embeddings"])
            existing = next((item for item in payload["profiles"]
                             if item["name"].casefold() == name.casefold()), None)
            now = _now()
            if existing is not None:
                if existing["model_set_id"] != model:
                    raise ValueError(
                        f"Le profil {name} utilise un autre modèle vocal. "
                        "Supprimez/réenrôlez ce profil ou choisissez un autre nom ; aucun mélange n'est permis."
                    )
                current = _validated_vectors(existing["embeddings"])
                if len(current[0]) != len(incoming[0]):
                    raise ValueError("Dimension incompatible avec le profil vocal existant.")
                _validate_enrollment_against_existing(name, current, incoming)
                bank = _diverse_bank(current + incoming)
                existing.update(
                    name=name, is_self=bool(request.get("is_self") or existing.get("is_self")),
                    embeddings=[list(item) for item in bank], updated_at=now,
                )
                profile = SpeakerProfile.from_dict(existing)
            else:
                profile = SpeakerProfile(
                    uuid.uuid4().hex, name, bool(request.get("is_self")), model,
                    tuple(_diverse_bank(incoming)), now, now,
                )
                payload["profiles"].append(profile.to_dict())
            enrolled.append(profile)
        return self._encode(payload), tuple(enrolled)

    def enroll(self, name: str, model_set_id: str, embeddings: Iterable[Iterable[float]],
               *, is_self: bool = False) -> SpeakerProfile:
        with speaker_lock(self.directory):
            self.transaction.recover()
            protected, profiles = self.prepare_enrollments([{
                "name": name, "model_set_id": model_set_id, "embeddings": embeddings, "is_self": is_self,
            }])
            atomic_bytes(self.path, protected)
            return profiles[0]

    def rename(self, profile_id: str, name: str) -> SpeakerProfile:
        clean = self.validate_name(name)
        with speaker_lock(self.directory):
            self.transaction.recover()
            payload = self._load_payload()
            if any(item["profile_id"] != profile_id and item["name"].casefold() == clean.casefold()
                   for item in payload["profiles"]):
                raise ValueError("Un profil porte déjà ce nom.")
            for item in payload["profiles"]:
                if item["profile_id"] == profile_id:
                    item.update(name=clean, updated_at=_now())
                    self._save_payload(payload)
                    return SpeakerProfile.from_dict(item)
        raise KeyError("Profil vocal introuvable.")

    def delete(self, profile_id: str) -> None:
        with speaker_lock(self.directory):
            self.transaction.recover()
            payload = self._load_payload()
            remaining = [item for item in payload["profiles"] if item["profile_id"] != profile_id]
            if len(remaining) == len(payload["profiles"]):
                raise KeyError("Profil vocal introuvable.")
            payload["profiles"] = remaining
            self._save_payload(payload)

    def delete_all(self) -> None:
        with speaker_lock(self.directory):
            self.transaction.recover()
            updates = {self.path: None}
            for candidate in (self.directory / "pending").glob("*.dpapi"):
                try:
                    payload = json.loads(self.codec.unprotect(candidate.read_bytes()).decode("utf-8"))
                    if not isinstance(payload, dict) or payload.get("schema_version") not in {1, 2}:
                        raise ValueError("Unknown review schema")
                    for cluster in payload.get("clusters") or []:
                        cluster["embeddings"] = []
                        cluster["biometric_samples_available"] = False
                    payload["biometric_samples_cleared"] = True
                except (ValueError, TypeError, KeyError, RuntimeError):
                    updates[candidate] = None
                else:
                    updates[candidate] = self.codec.protect(json.dumps(
                        payload, ensure_ascii=False, allow_nan=False
                    ).encode("utf-8"))
            self.transaction.commit(updates)

    def match_cluster_for_model(
        self, cluster: SpeakerCluster, model_set_id: str, *,
        threshold: float = DEFAULT_AUTO_THRESHOLD,
        margin_required: float = DEFAULT_AUTO_MARGIN,
    ) -> SpeakerMatch:
        """Conservative multi-embedding identification designed to minimize false names.

        A wrong automatic name is more damaging than leaving a cluster as Intervenant N.
        Recognition therefore requires repeated support from several query samples, a clear
        lead over competing profiles, and multiple agreeing references inside each profile.
        """

        def rejected(reason: str, *, name: str | None = None, profile_id: str | None = None,
                     is_self: bool = False, score: float = 0.0, margin: float = 0.0) -> SpeakerMatch:
            return SpeakerMatch(profile_id, name, is_self, score, margin, False, reason)

        if cluster.source == "self":
            return SpeakerMatch(None, None, True, 1.0, 1.0, False, "microphone personnel")

        if not math.isfinite(cluster.duration) or cluster.duration < MIN_AUTO_CLUSTER_SECONDS:
            return rejected("durée de parole insuffisante")

        if len(cluster.embeddings) < MIN_AUTO_QUERY_EMBEDDINGS:
            return rejected("pas assez d'extraits vocaux propres")

        if not (
            math.isfinite(threshold)
            and 0 <= threshold <= 1
            and math.isfinite(margin_required)
            and 0 <= margin_required <= 1
        ):
            raise ValueError("Seuil d'identification invalide.")

        try:
            query = np.asarray(_validated_vectors(cluster.embeddings), dtype=np.float32)
        except (TypeError, ValueError):
            return rejected("empreintes de requête invalides")

        candidates = []
        for profile in self.list_profiles():
            if profile.model_set_id != model_set_id:
                continue
            if len(profile.embeddings) < MIN_PROFILE_EMBEDDINGS_FOR_AUTO:
                continue
            try:
                stored = _validated_vectors(profile.embeddings)
                trusted = _trusted_reference_bank(stored)
                references = np.asarray(trusted, dtype=np.float32)
            except (TypeError, ValueError):
                continue
            if references.ndim != 2 or references.shape[1] != query.shape[1]:
                continue

            per_query = _topk_reference_score(query, references)
            ordered = np.sort(per_query)
            if len(ordered) >= 5:
                robust_score = float(np.mean(ordered[1:]))
            else:
                robust_score = float(np.mean(ordered))
            candidates.append({
                "profile": profile,
                "per_query": per_query,
                "score": robust_score,
                "mean": float(np.mean(per_query)),
                "trusted_references": len(trusted),
            })

        if not candidates:
            return rejected("aucun profil compatible suffisamment documenté")

        matrix = np.stack([candidate["per_query"] for candidate in candidates], axis=1)
        winners = np.argmax(matrix, axis=1)
        for index, candidate in enumerate(candidates):
            candidate["votes"] = int(np.sum(winners == index))

        candidates.sort(key=lambda item: (
            -item["score"], -item["votes"], -item["mean"], item["profile"].profile_id
        ))

        best = candidates[0]
        profile = best["profile"]
        similarities = best["per_query"]
        score = float(best["score"])
        second_score = float(candidates[1]["score"]) if len(candidates) > 1 else 0.0
        margin = score - second_score
        sample_count = len(similarities)
        required_support = max(2, int(math.ceil(sample_count * AUTO_REQUIRED_SUPPORT_RATIO)))
        vote_count = int(best["votes"])
        sample_floor = max(0.0, threshold - AUTO_SAMPLE_FLOOR_DELTA)
        quality_count = int(np.sum(similarities >= sample_floor))

        accepted = bool(
            score >= threshold
            and margin >= margin_required
            and vote_count >= required_support
            and quality_count >= required_support
        )

        if accepted:
            reason = (
                f"identification prudente: score={score:.3f}, marge={margin:.3f}, "
                f"votes={vote_count}/{sample_count}, extraits_valides={quality_count}/{sample_count}, "
                f"references_fiables={best['trusted_references']}"
            )
        elif score < threshold:
            reason = f"similarité globale insuffisante: score={score:.3f} < {threshold:.3f}"
        elif margin < margin_required:
            reason = (
                f"identité ambiguë entre plusieurs profils: marge={margin:.3f} < {margin_required:.3f}"
            )
        elif vote_count < required_support:
            reason = (
                f"accord insuffisant entre extraits: {vote_count}/{sample_count}, minimum={required_support}"
            )
        else:
            reason = (
                f"pas assez d'extraits cohérents: {quality_count}/{sample_count}, minimum={required_support}"
            )

        if accepted:
            return SpeakerMatch(
                profile.profile_id, profile.name, profile.is_self, score, margin, True, reason
            )
        return rejected(
            reason,
            name=profile.name,
            profile_id=profile.profile_id,
            is_self=profile.is_self,
            score=score,
            margin=margin,
        )
