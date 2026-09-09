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
DEFAULT_AUTO_THRESHOLD = 0.74
DEFAULT_AUTO_MARGIN = 0.015
MIN_AUTO_CLUSTER_SECONDS = 5.0
MIN_AUTO_QUERY_EMBEDDINGS = 2
MIN_PROFILE_EMBEDDINGS_FOR_AUTO = 2
MAX_PROFILE_EMBEDDINGS = 64
MAX_NAME_LENGTH = 80


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
        raise ValueError("Aucune empreinte vocale exploitable \u00e0 m\u00e9moriser.")
    if any(len(value) != len(vectors[0]) for value in vectors):
        raise ValueError("Dimensions d'empreintes vocales incoh\u00e9rentes.")
    return vectors


def _diverse_bank(vectors: list[tuple[float, ...]], limit: int = MAX_PROFILE_EMBEDDINGS):
    """Bound storage without a centroid or FIFO eviction of rare voice variants.

    Deterministic farthest-first selection maximizes coverage of confirmed modes.
    Below the limit, EVERY confirmed sample is retained, including legacy ones.
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
            raise ValueError("Le nom du locuteur doit \u00eatre un texte.")
        if any(unicodedata.category(char) in {"Cc", "Cf"} for char in name):
            raise ValueError("Le nom du locuteur contient un caract\u00e8re de contr\u00f4le.")
        clean = " ".join(name.strip().split())
        if not clean:
            raise ValueError("Le nom du locuteur est vide.")
        if len(clean) > MAX_NAME_LENGTH:
            raise ValueError(f"Le nom du locuteur est limit\u00e9 \u00e0 {MAX_NAME_LENGTH} caract\u00e8res.")
        return clean

    def _load_payload(self) -> dict[str, Any]:
        if not self.path.exists():
            return {"schema_version": PROFILE_SCHEMA_VERSION, "profiles": []}
        try:
            payload = json.loads(self.codec.unprotect(self.path.read_bytes()).decode("utf-8"))
        except Exception as exc:
            raise RuntimeError("La base chiffr\u00e9e des voix connues est illisible.") from exc
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
                    raise ValueError("Identit\u00e9 de profil dupliqu\u00e9e ou invalide.")
                _validated_vectors(profile.embeddings)
                seen_ids.add(profile.profile_id)
                seen_names.add(name.casefold())
        except (TypeError, KeyError, ValueError) as exc:
            raise RuntimeError("Un profil vocal est corrompu ; aucune donn\u00e9e n'a \u00e9t\u00e9 \u00e9cras\u00e9e.") from exc
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
                raise ValueError("Identifiant de mod\u00e8le vocal invalide.")
            incoming = _validated_vectors(request["embeddings"])
            existing = next((item for item in payload["profiles"]
                             if item["name"].casefold() == name.casefold()), None)
            now = _now()
            if existing is not None:
                if existing["model_set_id"] != model:
                    raise ValueError(
                        f"Le profil {name} utilise un autre mod\u00e8le vocal. "
                        "Supprimez/r\u00e9enr\u00f4lez ce profil ou choisissez un autre nom ; aucun m\u00e9lange n'est permis."
                    )
                current = _validated_vectors(existing["embeddings"])
                if len(current[0]) != len(incoming[0]):
                    raise ValueError("Dimension incompatible avec le profil vocal existant.")
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
                raise ValueError("Un profil porte d\u00e9j\u00e0 ce nom.")
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
                    # An unreadable review cannot preserve usable metadata.
                    updates[candidate] = None
                else:
                    updates[candidate] = self.codec.protect(json.dumps(
                        payload, ensure_ascii=False, allow_nan=False
                    ).encode("utf-8"))
            # Delete voice data, not the usable text-correction/group history.
            self.transaction.commit(updates)

    def match_cluster_for_model(
        self, cluster: SpeakerCluster, model_set_id: str, *,
        threshold: float = DEFAULT_AUTO_THRESHOLD,
        margin_required: float = DEFAULT_AUTO_MARGIN,
    ) -> SpeakerMatch:
        """Robust multi-embedding speaker identification.

        The previous implementation rejected a complete identity as soon as a
        single query embedding was noisy or closer to another profile.

        This implementation:
        - keeps every stored reference embedding;
        - compares every query embedding to every compatible profile;
        - tolerates one bad query embedding when at least four are available;
        - requires a majority of query embeddings to vote for the same profile;
        - computes ambiguity from aggregate profile scores, not from the single
          worst per-sample margin.
        """

        def rejected(reason: str) -> SpeakerMatch:
            return SpeakerMatch(None, None, False, 0.0, 0.0, False, reason)

        if cluster.source == "self":
            return SpeakerMatch(
                None,
                None,
                True,
                1.0,
                1.0,
                False,
                "microphone personnel",
            )

        if not math.isfinite(cluster.duration) or cluster.duration < MIN_AUTO_CLUSTER_SECONDS:
            return rejected("duree de parole insuffisante")

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
            query = np.asarray(
                _validated_vectors(cluster.embeddings),
                dtype=np.float32,
            )
        except (TypeError, ValueError):
            return rejected("empreintes de requete invalides")

        candidates = []

        for profile in self.list_profiles():
            if profile.model_set_id != model_set_id:
                continue

            if len(profile.embeddings) < MIN_PROFILE_EMBEDDINGS_FOR_AUTO:
                continue

            try:
                references = np.asarray(
                    _validated_vectors(profile.embeddings),
                    dtype=np.float32,
                )
            except (TypeError, ValueError):
                continue

            if references.ndim != 2 or references.shape[1] != query.shape[1]:
                continue

            # For each fresh voice excerpt, retain the closest confirmed
            # reference variant of this person.
            per_query = np.clip(
                query @ references.T,
                -1.0,
                1.0,
            ).max(axis=1)

            # Robust aggregate:
            # with >= 4 query embeddings, one potentially noisy excerpt may
            # not veto an otherwise coherent identification.
            ordered = np.sort(per_query)

            if len(ordered) >= 4:
                robust_score = float(np.mean(ordered[1:]))
            else:
                robust_score = float(np.mean(ordered))

            mean_score = float(np.mean(per_query))

            candidates.append(
                {
                    "profile": profile,
                    "per_query": per_query,
                    "score": robust_score,
                    "mean": mean_score,
                }
            )

        if not candidates:
            return rejected(
                "aucun profil compatible suffisamment documente"
            )

        # Matrix: query embeddings x known profiles.
        matrix = np.stack(
            [candidate["per_query"] for candidate in candidates],
            axis=1,
        )

        # Each fresh embedding votes independently for the closest profile.
        winners = np.argmax(matrix, axis=1)

        for index, candidate in enumerate(candidates):
            candidate["votes"] = int(np.sum(winners == index))

        candidates.sort(
            key=lambda item: (
                -item["score"],
                -item["votes"],
                -item["mean"],
                item["profile"].profile_id,
            )
        )

        best = candidates[0]
        profile = best["profile"]
        similarities = best["per_query"]
        score = float(best["score"])

        second_score = (
            float(candidates[1]["score"])
            if len(candidates) > 1
            else 0.0
        )

        margin = score - second_score

        sample_count = len(similarities)

        # 2/2, 2/3, 3/4, 3/5, 4/6...
        required_support = max(
            2,
            int(math.ceil(sample_count * 0.50)),
        )

        vote_count = int(best["votes"])

        # Individual samples may be slightly weaker than the final aggregate,
        # but a majority must remain credible.
        sample_floor = max(0.0, threshold - 0.08)

        quality_count = int(
            np.sum(similarities >= sample_floor)
        )

        accepted = bool(
            score >= threshold
            and margin >= margin_required
            and vote_count >= required_support
            and quality_count >= required_support
        )

        if accepted:
            reason = (
                f"identification robuste: score={score:.3f}, "
                f"marge={margin:.3f}, "
                f"votes={vote_count}/{sample_count}, "
                f"extraits_valides={quality_count}/{sample_count}"
            )
        elif score < threshold:
            reason = (
                f"similarite globale insuffisante: "
                f"score={score:.3f} < {threshold:.3f}"
            )
        elif margin < margin_required:
            reason = (
                f"identite ambigue entre plusieurs profils: "
                f"marge={margin:.3f} < {margin_required:.3f}"
            )
        elif vote_count < required_support:
            reason = (
                f"vote vocal insuffisant: "
                f"{vote_count}/{sample_count}, "
                f"minimum={required_support}"
            )
        else:
            reason = (
                f"pas assez d'extraits coherents: "
                f"{quality_count}/{sample_count}, "
                f"minimum={required_support}"
            )

        return SpeakerMatch(
            profile.profile_id,
            profile.name,
            profile.is_self,
            score,
            margin,
            accepted,
            reason,
        )
