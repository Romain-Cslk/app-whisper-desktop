"""OpenAI client construction, legacy transcription and diarized transcription."""
from __future__ import annotations

from pathlib import Path
from typing import Callable, Iterable

import httpx

from ..core.config import OPENAI_MAX_RETRIES, OPENAI_TIMEOUT_SECONDS
from ..models.jobs import ValidationError
from ..models.transcript import StructuredTranscript, TranscriptSegment

DIARIZATION_API_MODEL = "gpt-4o-transcribe-diarize"
TRANSCRIPTIONS_URL = "https://api.openai.com/v1/audio/transcriptions"


def make_openai_client(api_key: str):
    if not api_key.strip():
        raise ValidationError("Aucune clé OpenAI fournie.")
    from openai import OpenAI

    return OpenAI(
        api_key=api_key.strip(),
        timeout=OPENAI_TIMEOUT_SECONDS,
        max_retries=OPENAI_MAX_RETRIES,
    )


def transcribe_chunks(
    client,
    chunks: Iterable[Path],
    model: str,
    language: str | None,
    cancel_check: Callable[[], None],
    on_chunk: Callable[[int, str], None],
) -> str:
    """Legacy path: persist each completed response before the next cancellation check."""
    texts: list[str] = []
    for index, path in enumerate(chunks, start=1):
        cancel_check()
        arguments = {"model": model}
        if language:
            arguments["language"] = language
        if texts:
            arguments["prompt"] = texts[-1][-500:]
        with path.open("rb") as stream:
            response = client.audio.transcriptions.create(file=stream, **arguments)
        text = (response if isinstance(response, str) else getattr(response, "text", "") or "").strip()
        if text:
            texts.append(text)
        on_chunk(index, "\n\n".join(texts).strip())
        cancel_check()
    return "\n\n".join(texts).strip()


def _raise_api_error(response: httpx.Response) -> None:
    try:
        payload = response.json()
        message = str((payload.get("error") or {}).get("message") or "")
    except Exception:
        message = ""
    if not message:
        message = f"HTTP {response.status_code}"
    raise RuntimeError(f"OpenAI transcription diarization: {message}")


def transcribe_diarized_chunks(
    api_key: str,
    chunks_with_offsets: Iterable[tuple[Path, float]],
    language: str | None,
    cancel_check: Callable[[], None],
    on_chunk: Callable[[int, str], None],
) -> StructuredTranscript:
    """Use OpenAI speaker-aware transcription only when diarization is enabled.

    The returned OpenAI speaker labels are deliberately not trusted as persistent
    identity. They only provide timestamped text; final identity is aligned with
    the local sherpa-onnx diarization and local encrypted voice profiles.
    """
    key = api_key.strip()
    if not key:
        raise ValidationError("Aucune clé OpenAI fournie.")
    chunks = list(chunks_with_offsets)
    output: list[TranscriptSegment] = []
    timeout = httpx.Timeout(OPENAI_TIMEOUT_SECONDS, connect=min(30.0, OPENAI_TIMEOUT_SECONDS))
    headers = {"Authorization": f"Bearer {key}"}
    with httpx.Client(timeout=timeout, headers=headers, follow_redirects=False) as client:
        for index, (path, offset) in enumerate(chunks, start=1):
            cancel_check()
            data: list[tuple[str, str]] = [
                ("model", DIARIZATION_API_MODEL),
                ("response_format", "diarized_json"),
                ("chunking_strategy", "auto"),
            ]
            if language:
                data.append(("language", language))
            with path.open("rb") as stream:
                response = client.post(
                    TRANSCRIPTIONS_URL,
                    data=data,
                    files={"file": (path.name, stream, "audio/mpeg")},
                )
            if response.status_code >= 400:
                _raise_api_error(response)
            try:
                payload = response.json()
            except ValueError as exc:
                raise RuntimeError("Réponse OpenAI de diarisation non JSON.") from exc
            segments = payload.get("segments") or []
            if not isinstance(segments, list):
                raise RuntimeError("Réponse OpenAI de diarisation sans segments valides.")
            for item in segments:
                if not isinstance(item, dict):
                    continue
                text = str(item.get("text") or "").strip()
                if not text:
                    continue
                start = max(0.0, float(item.get("start") or 0.0) + float(offset))
                end = max(start, float(item.get("end") or item.get("start") or 0.0) + float(offset))
                output.append(
                    TranscriptSegment(
                        start=start,
                        end=end,
                        text=text,
                        speaker_key=(str(item.get("speaker")) if item.get("speaker") is not None else None),
                        speaker_source="openai-temporary",
                    )
                )
            partial = StructuredTranscript(tuple(output), language).plain_text()
            on_chunk(index, partial)
            cancel_check()
    return StructuredTranscript(tuple(output), language)
