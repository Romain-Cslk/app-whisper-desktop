"""Pure timestamp-preserving composition of independently transcribed native tracks."""
from __future__ import annotations

import math
from dataclasses import replace

from ..models.transcript import StructuredTranscript


def shift_source(transcript: StructuredTranscript, offset: float, source: str) -> StructuredTranscript:
    if not math.isfinite(offset) or offset < 0 or source not in {"self", "system"}:
        raise ValueError("Source ou decalage audio invalide.")
    segments = tuple(replace(
        item, start=item.start + offset, end=item.end + offset, speaker_source=source,
        words=tuple(replace(word, start=word.start + offset, end=word.end + offset) for word in item.words),
    ) for item in transcript.segments)
    return StructuredTranscript(segments, transcript.language)


def combine_sources(transcripts: list[StructuredTranscript]) -> StructuredTranscript:
    return StructuredTranscript(
        tuple(sorted((segment for transcript in transcripts for segment in transcript.segments),
                     key=lambda segment: (segment.start, segment.end, segment.speaker_source or ""))),
        next((item.language for item in transcripts if item.language), None),
    )
