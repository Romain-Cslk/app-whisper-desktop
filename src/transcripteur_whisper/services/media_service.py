"""PyAV conversion and bounded API audio chunks, independent of UI."""
from __future__ import annotations

import shutil
from pathlib import Path
from typing import Callable, List, Optional

import av
from av.audio.resampler import AudioResampler

from ..core.config import (
    API_AUDIO_BIT_RATE,
    API_AUDIO_SAMPLE_RATE,
    API_CHUNK_CONTENT_SECONDS,
    MAX_API_CHUNK_BYTES,
    MAX_API_CHUNK_SECONDS,
    VIDEO_EXTENSIONS,
)
from .files import format_seconds_hms

try:
    av.Codec("libmp3lame", "w")
    MP3_ENCODER_ERROR = None
except Exception as exc:
    MP3_ENCODER_ERROR = str(exc)

def probe_audio_duration(path: Path) -> Optional[float]:
    try:
        with av.open(str(path)) as container:
            stream = next((s for s in container.streams if s.type == "audio"), None)
            if stream is None:
                return None
            if stream.duration and stream.time_base:
                return float(stream.duration * stream.time_base)
            if container.duration:
                return float(container.duration / av.time_base)
    except Exception:
        return None
    return None

def transcode_to_mp3_parts(
    source: Path,
    destination_for_index: Callable[[int], Path],
    max_content_seconds: Optional[int] = None,
    cancel_check: Optional[Callable[[], None]] = None,
) -> List[Path]:
    """Décode un média et écrit un ou plusieurs MP3 mono 16 kHz sans perte de trame."""
    if MP3_ENCODER_ERROR is not None:
        raise RuntimeError(
            "L'encodeur MP3 libmp3lame n'est pas disponible dans ce build PyAV : "
            f"{MP3_ENCODER_ERROR}"
        )
    created_paths: List[Path] = []
    writer = None
    output_stream = None
    current_path: Optional[Path] = None
    samples_in_part = 0
    max_samples = (
        int(max_content_seconds * API_AUDIO_SAMPLE_RATE)
        if max_content_seconds is not None
        else None
    )

    def open_writer() -> None:
        nonlocal writer, output_stream, current_path, samples_in_part
        current_path = destination_for_index(len(created_paths) + 1)
        current_path.parent.mkdir(parents=True, exist_ok=True)
        current_path.unlink(missing_ok=True)
        writer = av.open(str(current_path), "w", format="mp3")
        output_stream = writer.add_stream("libmp3lame", rate=API_AUDIO_SAMPLE_RATE)
        output_stream.layout = "mono"
        output_stream.bit_rate = API_AUDIO_BIT_RATE
        samples_in_part = 0

    def close_writer() -> None:
        nonlocal writer, output_stream, current_path, samples_in_part
        if writer is None or output_stream is None or current_path is None:
            return
        for packet in output_stream.encode(None):
            writer.mux(packet)
        writer.close()
        if not current_path.exists() or current_path.stat().st_size == 0:
            raise RuntimeError("L'encodage MP3 a produit un segment vide.")
        created_paths.append(current_path)
        writer = None
        output_stream = None
        current_path = None
        samples_in_part = 0

    def encode_frame(frame) -> None:
        nonlocal samples_in_part
        frame_samples = int(frame.samples or 0)
        if frame_samples <= 0:
            return
        if max_samples is not None and frame_samples > max_samples:
            raise RuntimeError("Trame audio anormalement longue, découpage MP3 impossible.")
        if (
            max_samples is not None
            and writer is not None
            and samples_in_part > 0
            and samples_in_part + frame_samples > max_samples
        ):
            close_writer()
        if writer is None:
            open_writer()
        frame.pts = None
        for packet in output_stream.encode(frame):
            writer.mux(packet)
        samples_in_part += frame_samples

    try:
        with av.open(str(source)) as container:
            audio_stream = next((stream for stream in container.streams if stream.type == "audio"), None)
            if audio_stream is None:
                raise RuntimeError("Ce média ne contient aucune piste audio.")

            resampler = AudioResampler(
                format="s16p",
                layout="mono",
                rate=API_AUDIO_SAMPLE_RATE,
            )
            for frame_index, decoded_frame in enumerate(container.decode(audio_stream)):
                if cancel_check is not None and frame_index % 100 == 0:
                    cancel_check()
                for resampled_frame in resampler.resample(decoded_frame):
                    encode_frame(resampled_frame)
            for resampled_frame in resampler.resample(None):
                encode_frame(resampled_frame)

        if cancel_check is not None:
            cancel_check()
        close_writer()
    except Exception:
        if writer is not None:
            try:
                writer.close()
            except Exception:
                pass
        if current_path is not None:
            current_path.unlink(missing_ok=True)
        for path in created_paths:
            path.unlink(missing_ok=True)
        raise

    if not created_paths:
        raise RuntimeError("Le média ne contient aucun échantillon audio exploitable.")
    return created_paths

def convert_media_to_mp3(
    source: Path,
    destination: Path,
    cancel_check: Optional[Callable[[], None]] = None,
) -> Path:
    parts = transcode_to_mp3_parts(source, lambda _index: destination, cancel_check=cancel_check)
    if len(parts) != 1:
        raise RuntimeError("Conversion MP3 incohérente.")
    return parts[0]

def split_media_to_mp3(
    source: Path,
    chunk_dir: Path,
    output_stem: str,
    cancel_check: Optional[Callable[[], None]] = None,
) -> List[Path]:
    return transcode_to_mp3_parts(
        source,
        lambda index: chunk_dir / f"{output_stem}_part{index:03d}.mp3",
        max_content_seconds=API_CHUNK_CONTENT_SECONDS,
        cancel_check=cancel_check,
    )


class MediaService:
    """Owns only derived media; caller source files are always preserved."""

    def __init__(self, paths):
        self.paths = paths

    @staticmethod
    def probe(path: Path) -> float | None:
        return probe_audio_duration(path)

    def prepare_api_chunks(self, job_id: str, file_index: int, source: Path,
                           output_stem: str, cancel_check=None):
        duration = self.probe(source)
        chunk_dir = self.paths.temp / job_id / f"api_chunks_{file_index:03d}"
        chunk_dir.mkdir(parents=True, exist_ok=True)
        try:
            chunks = split_media_to_mp3(source, chunk_dir, output_stem, cancel_check)
            for chunk in chunks:
                if chunk.stat().st_size > MAX_API_CHUNK_BYTES:
                    raise RuntimeError(f"Le segment {chunk.name} dépasse la limite de taille API.")
                chunk_duration = self.probe(chunk)
                if chunk_duration is None:
                    raise RuntimeError(f"Impossible de vérifier la durée du segment {chunk.name}.")
                if chunk_duration > MAX_API_CHUNK_SECONDS:
                    raise RuntimeError(f"Le segment {chunk.name} dure {format_seconds_hms(chunk_duration)}, au-delà de la limite.")
            return chunks, chunk_dir, duration
        except BaseException:
            shutil.rmtree(chunk_dir, ignore_errors=True)
            raise

    def prepare_local_media(self, job_id: str, file_index: int, source: Path,
                            output_stem: str, cancel_check=None):
        if source.suffix.lower() not in VIDEO_EXTENSIONS:
            return source, None
        converted_dir = self.paths.temp / job_id / f"converted_{file_index:03d}"
        converted_dir.mkdir(parents=True, exist_ok=True)
        converted = converted_dir / f"{output_stem}.mp3"
        try:
            convert_media_to_mp3(source, converted, cancel_check)
        except BaseException:
            shutil.rmtree(converted_dir, ignore_errors=True)
            raise
        return converted, converted_dir
