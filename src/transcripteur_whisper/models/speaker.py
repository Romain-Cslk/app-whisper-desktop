"""Serializable models for diarization, voice profiles and review state."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class DiarizationTurn:
    start: float
    end: float
    cluster_id: str
    source: str = "mixed"

    @property
    def duration(self) -> float:
        return max(0.0, self.end - self.start)

    def to_dict(self) -> dict[str, Any]:
        return {
            "start": self.start,
            "end": self.end,
            "cluster_id": self.cluster_id,
            "source": self.source,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "DiarizationTurn":
        return cls(
            float(payload["start"]),
            float(payload["end"]),
            str(payload["cluster_id"]),
            str(payload.get("source") or "mixed"),
        )


@dataclass(frozen=True)
class SpeakerCluster:
    cluster_id: str
    source: str
    duration: float
    embeddings: tuple[tuple[float, ...], ...] = field(default_factory=tuple)

    def to_dict(self) -> dict[str, Any]:
        return {
            "cluster_id": self.cluster_id,
            "source": self.source,
            "duration": self.duration,
            "embeddings": [list(value) for value in self.embeddings],
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "SpeakerCluster":
        return cls(
            cluster_id=str(payload["cluster_id"]),
            source=str(payload.get("source") or "mixed"),
            duration=float(payload.get("duration") or 0.0),
            embeddings=tuple(tuple(float(v) for v in item) for item in payload.get("embeddings") or []),
        )


@dataclass(frozen=True)
class DiarizationResult:
    model_set_id: str
    turns: tuple[DiarizationTurn, ...]
    clusters: tuple[SpeakerCluster, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "model_set_id": self.model_set_id,
            "turns": [item.to_dict() for item in self.turns],
            "clusters": [item.to_dict() for item in self.clusters],
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "DiarizationResult":
        if int(payload.get("schema_version", 1)) != 1:
            raise ValueError("Version de résultat de diarisation non prise en charge.")
        return cls(
            str(payload["model_set_id"]),
            tuple(DiarizationTurn.from_dict(item) for item in payload.get("turns") or []),
            tuple(SpeakerCluster.from_dict(item) for item in payload.get("clusters") or []),
        )


@dataclass(frozen=True)
class SpeakerMatch:
    profile_id: str | None
    name: str | None
    is_self: bool
    score: float
    margin: float
    accepted: bool
    reason: str


@dataclass(frozen=True)
class SpeakerProfile:
    profile_id: str
    name: str
    is_self: bool
    model_set_id: str
    embeddings: tuple[tuple[float, ...], ...]
    created_at: str
    updated_at: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "profile_id": self.profile_id,
            "name": self.name,
            "is_self": self.is_self,
            "model_set_id": self.model_set_id,
            "embeddings": [list(value) for value in self.embeddings],
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "SpeakerProfile":
        return cls(
            str(payload["profile_id"]),
            str(payload["name"]),
            bool(payload.get("is_self")),
            str(payload["model_set_id"]),
            tuple(tuple(float(v) for v in item) for item in payload.get("embeddings") or []),
            str(payload["created_at"]),
            str(payload["updated_at"]),
        )
