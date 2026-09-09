from __future__ import annotations

import tempfile
import threading
import unittest
from contextlib import ExitStack
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from transcripteur_whisper.audio import native_recorder as recorder_module


class _FakeSession:
    def __init__(
        self,
        recording_id: str = "recording-id",
        output_path: Path = Path("recording.wav"),
        microphone_index: int = 1,
        microphone_name: str = "Microphone",
        speaker_id: str = "speaker-id",
        speaker_name: str = "Speakers",
        *,
        recording: bool = False,
        live: bool = False,
        stop_error: Exception | None = None,
        start_entered: threading.Event | None = None,
        release_start: threading.Event | None = None,
    ) -> None:
        self.recording_id = recording_id
        self.output_path = Path(output_path)
        self.microphone_index = microphone_index
        self.microphone_name = microphone_name
        self.speaker_id = speaker_id
        self.speaker_name = speaker_name
        self.recording = recording
        self.live = live
        self.stop_error = stop_error
        self.start_entered = start_entered
        self.release_start = release_start
        self.start_calls = 0
        self.stop_calls = 0
        self.discarded = False

    def start(self) -> None:
        self.start_calls += 1
        if self.start_entered is not None:
            self.start_entered.set()
        if self.release_start is not None and not self.release_start.wait(2.0):
            raise RuntimeError("test start timeout")
        self.recording = True
        self.live = True

    def stop(self) -> recorder_module.NativeRecordingResult:
        self.stop_calls += 1
        if self.stop_error is not None:
            raise self.stop_error
        self.recording = False
        self.live = False
        return recorder_module.NativeRecordingResult(
            recording_id=self.recording_id,
            path=self.output_path,
            duration=1.0,
            system_device_name=self.speaker_name,
            microphone_device_name=self.microphone_name,
        )

    def is_recording(self) -> bool:
        return self.recording and self.live

    def has_live_workers(self) -> bool:
        return self.live

    def discard_output(self) -> None:
        self.discarded = True


class NativeRecorderCoordinatorTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        self.base_dir = Path(self.temp_dir.name)
        self.recorder = recorder_module.NativeRecorder(self.base_dir)

    def _start_patches(self, session: _FakeSession) -> ExitStack:
        def create_session(**kwargs):
            for key, value in kwargs.items():
                setattr(session, key, value)
            return session

        stack = ExitStack()
        stack.enter_context(
            mock.patch.object(
                self.recorder,
                "_resolve_microphone",
                return_value=(1, session.microphone_name),
            )
        )
        stack.enter_context(
            mock.patch.object(
                self.recorder,
                "_resolve_speaker",
                return_value=(session.speaker_id, session.speaker_name),
            )
        )
        stack.enter_context(
            mock.patch.object(recorder_module, "_RecordingSession", side_effect=create_session)
        )
        return stack

    def test_duplicate_start_recovers_the_active_recording(self) -> None:
        active = _FakeSession(recording=True, live=True)
        self.recorder._active = active

        result = self.recorder.start("another-mic", "another-speaker")

        self.assertEqual(result.recording_id, active.recording_id)
        self.assertTrue(result.resumed)
        self.assertIs(self.recorder._active, active)

    def test_dead_session_is_released_before_a_new_start(self) -> None:
        dead = _FakeSession(recording=False, live=False)
        replacement = _FakeSession(output_path=self.base_dir / "replacement.wav")
        self.recorder._active = dead

        with self._start_patches(replacement):
            result = self.recorder.start()

        self.assertFalse(result.resumed)
        self.assertIs(self.recorder._active, replacement)
        self.assertEqual(replacement.start_calls, 1)
        self.assertFalse(dead.discarded)

    def test_concurrent_starts_create_only_one_session(self) -> None:
        entered = threading.Event()
        release = threading.Event()
        session = _FakeSession(
            output_path=self.base_dir / "concurrent.wav",
            start_entered=entered,
            release_start=release,
        )
        results: list[recorder_module.NativeRecordingResult] = []
        errors: list[BaseException] = []

        def start() -> None:
            try:
                results.append(self.recorder.start())
            except BaseException as exc:  # pragma: no cover - assertion aid
                errors.append(exc)

        with self._start_patches(session):
            first = threading.Thread(target=start)
            second = threading.Thread(target=start)
            first.start()
            self.assertTrue(entered.wait(1.0))
            second.start()
            release.set()
            first.join(2.0)
            second.join(2.0)

        self.assertFalse(errors)
        self.assertEqual(len(results), 2)
        self.assertEqual({result.recording_id for result in results}, {session.recording_id})
        self.assertEqual(sum(result.resumed for result in results), 1)
        self.assertEqual(session.start_calls, 1)

    def test_failed_stop_releases_dead_session_and_preserves_recovery_files(self) -> None:
        session = _FakeSession(
            recording=False,
            live=False,
            stop_error=recorder_module.NativeRecorderError("mix failed"),
        )
        self.recorder._active = session

        with self.assertRaisesRegex(recorder_module.NativeRecorderError, "mix failed"):
            self.recorder.stop(session.recording_id)

        self.assertIsNone(self.recorder._active)
        self.assertFalse(session.discarded)

    def test_failed_stop_keeps_a_genuinely_live_session_for_retry(self) -> None:
        session = _FakeSession(
            recording=False,
            live=True,
            stop_error=recorder_module.NativeRecorderError("worker stuck"),
        )
        self.recorder._active = session

        with self.assertRaisesRegex(recorder_module.NativeRecorderError, "worker stuck"):
            self.recorder.stop(session.recording_id)

        self.assertIs(self.recorder._active, session)

    def test_stop_retry_returns_the_completed_result(self) -> None:
        session = _FakeSession(recording=True, live=True, output_path=self.base_dir / "done.wav")
        self.recorder._active = session

        first = self.recorder.stop(session.recording_id)
        second = self.recorder.stop(session.recording_id)

        self.assertIs(first, second)
        self.assertEqual(session.stop_calls, 1)
        self.assertIsNone(self.recorder._active)


class NativeRecordingSessionStateTests(unittest.TestCase):
    def _bare_session(self) -> recorder_module._RecordingSession:
        session = object.__new__(recorder_module._RecordingSession)
        session._mic_stream = None
        session._system_reader_thread = None
        session._sources = {}
        session._start_ts = 1.0
        session._stop_event = threading.Event()
        return session

    def test_active_microphone_stream_counts_as_a_live_capture_resource(self) -> None:
        session = self._bare_session()
        session._mic_stream = SimpleNamespace(active=True)

        self.assertTrue(session.has_live_workers())
        self.assertTrue(session.is_recording())

    def test_stop_request_makes_session_non_recording_even_if_worker_is_stuck(self) -> None:
        session = self._bare_session()
        session._mic_stream = SimpleNamespace(active=True)
        session._stop_event.set()

        self.assertTrue(session.has_live_workers())
        self.assertFalse(session.is_recording())


if __name__ == "__main__":
    unittest.main()
