from __future__ import annotations

import tempfile
import unittest
from fractions import Fraction
from pathlib import Path
from unittest.mock import patch

import av
import numpy as np

from transcripteur_whisper.core.paths import AppPaths
from transcripteur_whisper.models.jobs import ValidationError
from transcripteur_whisper.services import media_service as media
from transcripteur_whisper.services.files import output_filename, safe_output_name, safe_upload_name


def create_audio_mp4(path: Path, duration_seconds: float = 2.2) -> None:
    sample_rate = 48_000
    frame_samples = 1024
    frame_count = int(duration_seconds * sample_rate / frame_samples) + 1
    with av.open(str(path), "w", format="mp4") as container:
        stream = container.add_stream("aac", rate=sample_rate)
        stream.layout = "mono"
        pts = 0
        for _ in range(frame_count):
            samples = np.zeros((1, frame_samples), dtype=np.float32)
            frame = av.AudioFrame.from_ndarray(samples, format="fltp", layout="mono")
            frame.sample_rate = sample_rate
            frame.pts = pts
            frame.time_base = Fraction(1, sample_rate)
            pts += frame_samples
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode(None):
            container.mux(packet)


def create_video_without_audio(path: Path) -> None:
    with av.open(str(path), "w", format="mp4") as container:
        stream = container.add_stream("mpeg4", rate=1)
        stream.width = 32
        stream.height = 32
        stream.pix_fmt = "yuv420p"
        image = np.zeros((32, 32, 3), dtype=np.uint8)
        frame = av.VideoFrame.from_ndarray(image, format="rgb24")
        for packet in stream.encode(frame):
            container.mux(packet)
        for packet in stream.encode(None):
            container.mux(packet)


class MediaPipelineTests(unittest.TestCase):
    def test_api_chunk_limit_is_ten_minutes_with_encoding_margin(self) -> None:
        self.assertEqual(media.MAX_API_CHUNK_SECONDS, 600)
        self.assertGreater(media.API_CHUNK_CONTENT_SECONDS, 0)
        self.assertLess(media.API_CHUNK_CONTENT_SECONDS, media.MAX_API_CHUNK_SECONDS)

    def test_prepare_api_chunks_rejects_mp3_with_unknown_duration(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.mp3"
            source.write_bytes(b"source")
            temp_dir = root / "temp"

            def fake_split(_source, chunk_dir, _output_stem, cancel_check=None):
                chunk = chunk_dir / "source_part001.mp3"
                chunk.write_bytes(b"mp3")
                return [chunk]

            def fake_probe(path):
                return 42.0 if Path(path) == source else None

            with (
                patch.object(media, "split_media_to_mp3", side_effect=fake_split),
                patch.object(media, "probe_audio_duration", side_effect=fake_probe),
            ):
                with self.assertRaisesRegex(RuntimeError, "(?i)durée"):
                    media.MediaService(AppPaths.create(root)).prepare_api_chunks("job-id", 1, source, "source")

            self.assertFalse((temp_dir / "job-id" / "api_chunks_001").exists())

    def test_mp4_is_converted_to_mp3(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "reunion.mp4"
            destination = root / "reunion.mp3"
            create_audio_mp4(source)

            result = media.convert_media_to_mp3(source, destination)

            self.assertEqual(result, destination)
            self.assertGreater(destination.stat().st_size, 0)
            self.assertEqual(destination.suffix, ".mp3")
            duration = media.probe_audio_duration(destination)
            self.assertIsNotNone(duration)
            self.assertGreater(duration, 2.0)

    def test_mp3_parts_are_bounded_and_keep_all_audio(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "longue.mp4"
            create_audio_mp4(source, duration_seconds=2.4)

            parts = media.transcode_to_mp3_parts(
                source,
                lambda index: root / f"part_{index:03d}.mp3",
                max_content_seconds=1,
            )

            self.assertGreaterEqual(len(parts), 3)
            durations = [media.probe_audio_duration(part) or 0 for part in parts]
            self.assertTrue(all(part.suffix == ".mp3" for part in parts))
            self.assertTrue(all(duration <= 1.2 for duration in durations), durations)
            self.assertGreater(sum(durations), 2.4)

    def test_video_without_audio_fails_clearly(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "muette.mp4"
            create_video_without_audio(source)

            with self.assertRaisesRegex(RuntimeError, "aucune piste audio"):
                media.convert_media_to_mp3(source, root / "muette.mp3")

    def test_upload_name_is_sanitized_and_validated(self) -> None:
        safe = safe_upload_name(r"..\..\réunion?.MP4", 1)
        self.assertEqual(safe, "001_réunion_.mp4")
        with self.assertRaisesRegex(ValidationError, "Format non pris en charge"):
            safe_upload_name("notes.exe", 2)

    def test_output_name_is_sanitized_or_replaced_by_local_timestamp(self) -> None:
        self.assertEqual(
            safe_output_name(r"..\Comité projet?.txt"),
            "Comité projet_",
        )
        self.assertRegex(
            safe_output_name("   "),
            r"^\d{2}_\d{2}_\d{4}_\d{2}h\d{2}$",
        )

    def test_output_filename_starts_with_type_and_handles_multiple_files(self) -> None:
        single_job = {"output_name": "Réunion client", "files": [{}]}
        multiple_job = {"output_name": "Réunion client", "files": [{}, {}]}

        self.assertEqual(
            output_filename(single_job, "compte_rendu", 0),
            "compte_rendu_Réunion client.txt",
        )
        self.assertEqual(
            output_filename(multiple_job, "transcription", 1),
            "transcription_Réunion client_02.txt",
        )
        self.assertEqual(
            output_filename(multiple_job, "transcription"),
            "transcription_Réunion client.txt",
        )

