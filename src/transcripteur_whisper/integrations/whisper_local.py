"""CPU/GPU faster-whisper backend with timestamp-preserving output."""
from __future__ import annotations

from pathlib import Path
from typing import Callable

from ..models.transcript import StructuredTranscript, TranscriptSegment, TranscriptWord


def load_model(model_path: Path, *, device: str = "cpu", compute_type: str | None = None):
    from faster_whisper import BatchedInferencePipeline, WhisperModel

    if device not in {"cpu", "cuda"}:
        raise ValueError("Périphérique Whisper inconnu.")

    selected_compute = compute_type or ("float16" if device == "cuda" else "int8")

    try:
        whisper_model = WhisperModel(
            str(model_path),
            device=device,
            compute_type=selected_compute,
            local_files_only=True,
        )

        # GPU = transcription batchée
        if device == "cuda":
            return BatchedInferencePipeline(model=whisper_model)

        # CPU = comportement historique
        return whisper_model

    except Exception as exc:
        if device == "cuda":
            raise RuntimeError(
                "Impossible de charger Whisper sur le GPU CUDA. "
                f"Détail : {exc}"
            ) from exc
        raise

def _run_transcription(model, path: Path, **kwargs):
    from faster_whisper import BatchedInferencePipeline

    if isinstance(model, BatchedInferencePipeline):
        kwargs["batch_size"] = 8

    return model.transcribe(str(path), **kwargs)

def transcribe_structured(
    model,
    path: Path,
    language: str | None,
    cancel_check: Callable[[], None],
    on_segment: Callable[[float, str], None],
) -> StructuredTranscript:
    segments, info = _run_transcription(
        model,
        path,
        language=language,
        beam_size=5,
        vad_filter=True,
        word_timestamps=True,
    )
    duration = info.duration or 1.0
    output: list[TranscriptSegment] = []
    for segment in segments:
        cancel_check()
        text = (segment.text or "").strip()
        words = tuple(
            TranscriptWord(float(word.start), float(word.end), str(word.word or ""))
            for word in (getattr(segment, "words", None) or [])
            if word.start is not None and word.end is not None and str(word.word or "").strip()
        )
        if text:
            output.append(
                TranscriptSegment(
                    start=max(0.0, float(segment.start or 0.0)),
                    end=max(0.0, float(segment.end or segment.start or 0.0)),
                    text=text,
                    words=words,
                )
            )
        current = StructuredTranscript(tuple(output), getattr(info, "language", language)).plain_text()
        on_segment(min(max(0.0, float(segment.end or 0.0)) / duration, 1.0), current)
    cancel_check()
    return StructuredTranscript(tuple(output), getattr(info, "language", language))



def transcribe(
    model,
    path: Path,
    language: str | None,
    cancel_check: Callable[[], None],
    on_segment: Callable[[float, str], None],
) -> str:
    """Exact legacy non-diarized path.

    Keeping this separate is intentional: when diarization is OFF we do not
    request word timestamps and do not alter the pre-1.1 transcription behavior.
    """
    segments, info = _run_transcription(
    model,
    path,
    language=language,
    beam_size=5,
    vad_filter=True,
    )
    duration = info.duration or 1.0
    lines: list[str] = []
    for segment in segments:
        cancel_check()
        text = (segment.text or "").strip()
        if text:
            lines.append(text)
        on_segment(min(max(0.0, float(segment.end or 0.0)) / duration, 1.0), "\n".join(lines).strip())
    cancel_check()
    return "\n".join(lines).strip()
