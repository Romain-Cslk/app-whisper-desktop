"""Bounded, local audio excerpt selection. No network and no extra audio copies."""
from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

MAX_EXCERPT_SECONDS = 6.0
EXCERPTS_PER_ORIGINAL_SPEAKER = 3


@dataclass(frozen=True)
class AudioExcerpt:
    path: Path
    start: float
    end: float
    global_start: float
    cluster_id: str
    isolated: bool
    overlap_possible: bool = False

    def __post_init__(self) -> None:
        if (not all(math.isfinite(value) for value in (self.start, self.end, self.global_start))
                or self.start < 0 or self.end <= self.start
                or self.end - self.start > MAX_EXCERPT_SECONDS + 0.001):
            raise ValueError("Bornes d'extrait audio invalides.")


def _subtract(start: float, end: float, cuts: Iterable[tuple[float, float]]) -> list[tuple[float, float]]:
    pieces = [(start, end)]
    for left, right in sorted(cuts):
        next_pieces = []
        for a, b in pieces:
            if right <= a or left >= b:
                next_pieces.append((a, b))
            else:
                if left > a:
                    next_pieces.append((a, min(left, b)))
                if right < b:
                    next_pieces.append((max(right, a), b))
        pieces = next_pieces
    return pieces


def excerpts_for(
    payload: dict[str, Any], members: Iterable[str], *, source_override: Path | None = None,
) -> list[AudioExcerpt]:
    """Select up to three non-redundant excerpts per ORIGINAL cluster.

    Each merged voice variant stays independently auditionable. Clean intervals
    are preferred; overlapping fallback excerpts are explicitly marked.
    """
    clusters = {str(item["cluster_id"]): item for item in payload.get("clusters") or []}
    turns = []
    for item in payload.get("turns") or []:
        try:
            a, b = float(item["start"]), float(item["end"])
            if math.isfinite(a) and math.isfinite(b) and b > a >= 0:
                turns.append({**item, "start": a, "end": b})
        except (ValueError, TypeError, KeyError):
            continue
    sources = payload.get("audio_sources") or {}
    output = []
    for member in dict.fromkeys(members):
        cluster = clusters.get(member)
        if cluster is None:
            raise ValueError("Locuteur introuvable pour l'\u00e9coute.")
        source = str(cluster.get("source") or "mixed")
        isolated = False
        if source_override is not None:
            path, offset = Path(source_override).resolve(), 0.0
        else:
            descriptor = sources.get(source)
            if (descriptor and source in {"self", "system"}
                    and not Path(descriptor.get("path") or "").is_file()):
                descriptor = None
            if descriptor:
                isolated = source in {"self", "system"}
            else:
                descriptor = sources.get("mixed")
            if not descriptor or not descriptor.get("path"):
                continue
            path = Path(descriptor["path"]).resolve()
            offset = float(descriptor.get("offset") or 0.0)
            if not math.isfinite(offset) or offset < 0:
                raise ValueError("D\u00e9calage de piste audio invalide.")
        own = [turn for turn in turns if turn.get("cluster_id") == member]
        candidates = []
        for turn in own:
            cuts = [
                (other["start"] - 0.03, other["end"] + 0.03)
                for other in turns
                if other.get("cluster_id") != member
                and (not isolated or str(other.get("source") or "mixed") == source)
                and min(turn["end"], other["end"]) > max(turn["start"], other["start"])
            ]
            clean = [(a, b) for a, b in _subtract(turn["start"], turn["end"], cuts) if b - a >= 0.4]
            for a, b in clean:
                candidates.append((a, b, False))
        if not candidates:
            candidates = [(turn["start"], turn["end"], True) for turn in own
                          if turn["end"] - turn["start"] >= 0.15]
        candidates.sort(key=lambda item: (-(item[1] - item[0]), item[0]))
        chosen = []
        # One excerpt from different turns first, then distinct parts of long turns.
        for part in range(EXCERPTS_PER_ORIGINAL_SPEAKER):
            for a, b, overlap in candidates:
                start = a + part * MAX_EXCERPT_SECONDS
                end = min(b, start + MAX_EXCERPT_SECONDS)
                if end - start < 0.15 or start < offset or len(chosen) >= EXCERPTS_PER_ORIGINAL_SPEAKER:
                    continue
                if any(min(end, old.end + offset) > max(start, old.start + offset) for old in chosen):
                    continue
                chosen.append(AudioExcerpt(path, start - offset, end - offset, start,
                                           member, isolated, overlap))
        output.extend(sorted(chosen, key=lambda item: item.global_start))
    return output
