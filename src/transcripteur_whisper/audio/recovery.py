"""Recover PCM WAV journals without modifying the sole original after a crash."""

import shutil
import struct
import uuid
from pathlib import Path

import soundfile as sf


class RecordingRecoveryError(RuntimeError):
    pass


def recover_wav(source: Path, results_dir: Path) -> Path:
    results_dir.mkdir(parents=True, exist_ok=True)
    destination = results_dir / f"enregistrement_recupere_{uuid.uuid4().hex[:12]}.wav"
    try:
        shutil.copyfile(source, destination)
        # libsndfile flushes journal data regularly. A hard crash can leave a
        # valid PCM payload after a data chunk whose declared length is zero.
        with destination.open("r+b") as wav:
            size = destination.stat().st_size
            header = wav.read(12)
            if len(header) != 12 or header[:4] != b"RIFF" or header[8:] != b"WAVE":
                raise RecordingRecoveryError("Le fichier ne contient pas un journal WAV PCM récupérable.")
            block_align = None
            while wav.tell() + 8 <= size:
                chunk_header = wav.read(8)
                chunk_id, declared = struct.unpack("<4sI", chunk_header)
                data_position = wav.tell()
                if chunk_id == b"fmt ":
                    if declared < 16:
                        raise RecordingRecoveryError("L'en-tête audio est incomplet.")
                    audio_format, channels, rate, byte_rate, align, bits = struct.unpack(
                        "<HHIIHH", wav.read(16)
                    )
                    if audio_format != 1 or channels != 2 or bits != 16:
                        raise RecordingRecoveryError(
                            "Le journal n'est pas au format PCM 16 bits stéréo attendu."
                        )
                    block_align = align
                elif chunk_id == b"data":
                    if not block_align:
                        raise RecordingRecoveryError("Le journal ne contient pas de format audio valide.")
                    available = size - data_position
                    length = available - available % block_align
                    if length <= 0 or length > 0xFFFFFFFF - data_position:
                        raise RecordingRecoveryError("Le journal audio est vide ou trop volumineux.")
                    wav.seek(data_position - 4)
                    wav.write(struct.pack("<I", length))
                    wav.seek(4)
                    wav.write(struct.pack("<I", data_position + length - 8))
                    wav.truncate(data_position + length)
                    break
                wav.seek(data_position + declared + declared % 2)
            else:
                raise RecordingRecoveryError("Aucune donnée audio récupérable dans ce journal.")
        info = sf.info(destination)
        if info.frames <= 0:
            raise RecordingRecoveryError("Le journal audio est vide.")
        return destination
    except Exception:
        destination.unlink(missing_ok=True)
        raise
