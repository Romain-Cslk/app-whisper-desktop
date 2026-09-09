"""Pure, non-destructive editing model for speaker grouping and preview."""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, replace
from typing import Any, Iterable

from ..models.transcript import StructuredTranscript
from .speaker_profile_service import SpeakerProfileService


@dataclass
class ReviewGroup:
    group_id: str
    members: tuple[str, ...]
    name: str
    remember: bool = False


class SpeakerReviewState:
    def __init__(self, payload: dict[str, Any]) -> None:
        items = payload.get("clusters") or []
        self.clusters = {str(item["cluster_id"]): deepcopy(item) for item in items}
        if len(self.clusters) != len(items):
            raise ValueError("Identifiants de locuteurs dupliqu\u00e9s.")
        self._order = {key: index for index, key in enumerate(self.clusters)}
        self.groups = {
            key: ReviewGroup(key, (key,), str(item.get("current_name") or key))
            for key, item in self.clusters.items()
        }
        self._undo: list[dict[str, ReviewGroup]] = []
        if payload.get("groups") is not None:
            self.set_groups(payload["groups"])

    def set_groups(self, entries: list[dict[str, Any]]) -> None:
        groups: dict[str, ReviewGroup] = {}
        seen: set[str] = set()
        for entry in entries:
            key = str(entry["group_id"])
            members = tuple(str(item) for item in entry["members"])
            if not members or len(set(members)) != len(members) or key not in members or key in groups:
                raise ValueError("Groupe de locuteurs invalide.")
            if not set(members).issubset(self.clusters) or seen.intersection(members):
                raise ValueError("Un locuteur est absent ou appartient \u00e0 plusieurs groupes.")
            seen.update(members)
            groups[key] = ReviewGroup(key, members, str(entry.get("name") or key))
        if seen != set(self.clusters):
            raise ValueError("La fusion doit conserver tous les locuteurs, sans en supprimer.")
        self.groups = groups

    def ordered_groups(self) -> list[ReviewGroup]:
        return sorted(self.groups.values(), key=lambda group: min(self._order[key] for key in group.members))

    @property
    def can_undo(self) -> bool:
        return bool(self._undo)

    def merge(self, group_ids: Iterable[str], name: str) -> str:
        keys = list(dict.fromkeys(group_ids))
        if len(keys) < 2 or not set(keys).issubset(self.groups):
            raise ValueError("S\u00e9lectionnez au moins deux locuteurs existants.")
        clean = SpeakerProfileService.validate_name(name)
        before = deepcopy(self.groups)
        members = tuple(sorted(
            (member for key in keys for member in self.groups[key].members), key=self._order.__getitem__
        ))
        remember = any(self.groups[key].remember for key in keys)
        for key in keys:
            del self.groups[key]
        canonical = members[0]
        self.groups[canonical] = ReviewGroup(canonical, members, clean, remember)
        self._undo.append(before)
        return canonical

    def split(self, group_id: str) -> None:
        group = self.groups[group_id]
        if len(group.members) < 2:
            raise ValueError("Ce locuteur n'est pas issu d'une fusion.")
        self._undo.append(deepcopy(self.groups))
        del self.groups[group_id]
        for member in group.members:
            self.groups[member] = ReviewGroup(
                member, (member,), str(self.clusters[member].get("current_name") or member), False
            )

    def undo(self) -> None:
        if not self._undo:
            return
        self.groups = self._undo.pop()

    def serialized_groups(self) -> list[dict[str, Any]]:
        return [
            {"group_id": group.group_id, "members": list(group.members), "name": group.name}
            for group in self.ordered_groups()
        ]

    def set_choices(self, names: dict[str, str], remember: set[str]) -> None:
        if not set(names).issubset(self.groups) or not remember.issubset(self.groups):
            raise ValueError("Un locuteur demand\u00e9 n'existe plus ; rouvrez la revue.")
        for key, group in self.groups.items():
            group.name = SpeakerProfileService.validate_name(names.get(key, group.name))
            group.remember = key in remember
        # Homonyms may be displayed, but silently merging their biometric profiles is unsafe.
        remembered = [group.name.casefold() for group in self.groups.values() if group.remember]
        if len(remembered) != len(set(remembered)):
            raise ValueError("Fusionnez d'abord les groupes de la m\u00eame personne avant de m\u00e9moriser sa voix.")

    def preview(self, transcript: StructuredTranscript) -> StructuredTranscript:
        mapping = {member: group for group in self.groups.values() for member in group.members}
        segments = []
        for segment in transcript.segments:
            original = segment.speaker_original_key or segment.speaker_key
            group = mapping.get(original or "")
            if group is None:
                segments.append(segment)
                continue
            # A human edit is not a new machine-confidence score.
            confidence = segment.speaker_confidence
            if group.name != segment.speaker_name or len(group.members) > 1:
                confidence = None
            segments.append(replace(
                segment, speaker_key=group.group_id, speaker_name=group.name,
                speaker_original_key=original, speaker_confidence=confidence,
            ))
        return StructuredTranscript(tuple(segments), transcript.language)

    def embeddings(self, group_id: str) -> list[list[float]]:
        return [
            list(vector)
            for member in self.groups[group_id].members
            for vector in self.clusters[member].get("embeddings") or []
        ]

    def duration(self, group_id: str) -> float:
        return sum(float(self.clusters[key].get("duration") or 0.0) for key in self.groups[group_id].members)
