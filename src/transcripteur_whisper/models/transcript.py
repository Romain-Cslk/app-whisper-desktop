"""Structured transcript models used by optional speaker diarization.

The legacy non-diarized path still renders exactly as newline-separated text.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable


def _timecode(seconds: float) -> str:
    value = max(0, int(seconds))
    hours, remainder = divmod(value, 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


@dataclass(frozen=True)
class TranscriptWord:
    start: float
    end: float
    text: str

    def to_dict(self) -> dict[str, Any]:
        return {"start": self.start, "end": self.end, "text": self.text}

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "TranscriptWord":
        return cls(float(payload["start"]), float(payload["end"]), str(payload.get("text") or ""))


@dataclass(frozen=True)
class TranscriptSegment:
    start: float
    end: float
    text: str
    words: tuple[TranscriptWord, ...] = field(default_factory=tuple)
    speaker_key: str | None = None
    speaker_name: str | None = None
    speaker_source: str | None = None
    speaker_confidence: float | None = None
    speaker_original_key: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "start": self.start,
            "end": self.end,
            "text": self.text,
            "words": [word.to_dict() for word in self.words],
            "speaker_key": self.speaker_key,
            "speaker_name": self.speaker_name,
            "speaker_source": self.speaker_source,
            "speaker_confidence": self.speaker_confidence,
            "speaker_original_key": self.speaker_original_key,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "TranscriptSegment":
        return cls(
            start=float(payload["start"]),
            end=float(payload["end"]),
            text=str(payload.get("text") or ""),
            words=tuple(TranscriptWord.from_dict(item) for item in payload.get("words") or []),
            speaker_key=payload.get("speaker_key"),
            speaker_name=payload.get("speaker_name"),
            speaker_source=payload.get("speaker_source"),
            speaker_original_key=payload.get("speaker_original_key"),
            speaker_confidence=(
                float(payload["speaker_confidence"])
                if payload.get("speaker_confidence") is not None
                else None
            ),
        )


@dataclass(frozen=True)
class StructuredTranscript:
    segments: tuple[TranscriptSegment, ...]
    language: str | None = None

    @classmethod
    def from_segments(
        cls, segments: Iterable[TranscriptSegment], language: str | None = None
    ) -> "StructuredTranscript":
        return cls(tuple(segments), language)

    def plain_text(self) -> str:
        return "\n".join(segment.text.strip() for segment in self.segments if segment.text.strip()).strip()

    def speaker_text(self) -> str:
        """Render a readable speaker transcript while merging adjacent turns.

        Each rendered turn keeps the timestamp of its first segment. Unknown
        speaker names are intentionally explicit instead of guessed.
        """
        rendered: list[tuple[float, str, list[str]]] = []
        previous_key: str | None = None
        for segment in self.segments:
            text = segment.text.strip()
            if not text:
                continue
            name = (segment.speaker_name or "Intervenant inconnu").strip() or "Intervenant inconnu"
            key = segment.speaker_key or name
            if rendered and previous_key == key and rendered[-1][1] == name:
                rendered[-1][2].append(text)
            else:
                rendered.append((segment.start, name, [text]))
            previous_key = key
        return "\n\n".join(
            f"[{_timecode(start)}] {name} : {' '.join(parts).strip()}" for start, name, parts in rendered
        ).strip()

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "language": self.language,
            "segments": [segment.to_dict() for segment in self.segments],
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "StructuredTranscript":
        if int(payload.get("schema_version", 1)) != 1:
            raise ValueError("Version de transcription structurée non prise en charge.")
        return cls(
            tuple(TranscriptSegment.from_dict(item) for item in payload.get("segments") or []),
            payload.get("language"),
        )
