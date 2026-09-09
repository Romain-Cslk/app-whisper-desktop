"""Recording lifecycle, persistent results and recovery, independent of Qt."""

from __future__ import annotations

import json
import logging
import queue
import shutil
import threading
from concurrent.futures import Future
from dataclasses import replace
from pathlib import Path
from typing import Any, Callable, TypeVar

from transcripteur_whisper.audio.native_recorder import NativeRecorder, NativeRecordingResult
from transcripteur_whisper.audio.recovery import recover_wav
from transcripteur_whisper.audio.windows_com import com_apartment
from transcripteur_whisper.models.audio_devices import AudioDeviceSnapshot
from transcripteur_whisper.services.audio_device_service import AudioDeviceError, AudioDeviceService

T = TypeVar("T")
logger = logging.getLogger(__name__)


class RecordingService:
    """All stream open/close operations share a dedicated COM apartment thread."""

    def __init__(self, paths: Any, devices: AudioDeviceService):
        self.paths = paths
        self.devices = devices
        self._journal_dir = Path(paths.temp) / "recordings"
        self._recorder = NativeRecorder(self._journal_dir)
        self._commands: queue.Queue = queue.Queue()
        self._worker_ready = threading.Event()
        self._startup_error: BaseException | None = None
        self._worker = threading.Thread(target=self._run, name="recording-coordinator", daemon=True)
        self._worker.start()
        self._active_result: NativeRecordingResult | None = None
        self._last_completed: NativeRecordingResult | None = None
        self._pending_final: NativeRecordingResult | None = None
        self._resources_active = False
        self._selected: dict[str, str] = {}
        self._closed = False

    def _run(self) -> None:
        try:
            with com_apartment():
                self._worker_ready.set()
                while True:
                    command = self._commands.get()
                    if command is None:
                        return
                    callback, future = command
                    try:
                        future.set_result(callback())
                    except BaseException as exc:
                        future.set_exception(exc)
        except BaseException as exc:
            self._startup_error = exc
        finally:
            self._worker_ready.set()

    def _call(self, callback: Callable[[], T]) -> T:
        if self._closed:
            raise RuntimeError("Le service d'enregistrement est fermé.")
        if not self._worker_ready.wait(10):
            raise AudioDeviceError("Le moteur audio ne répond pas à l'initialisation.")
        if self._startup_error:
            raise AudioDeviceError(f"Initialisation du moteur audio impossible : {self._startup_error}")
        future: Future = Future()
        self._commands.put((callback, future))
        return future.result()

    @property
    def is_active(self) -> bool:
        return self._active_result is not None or self._resources_active

    def start(
        self,
        microphone_key: str | None = None,
        speaker_key: str | None = None,
        *,
        microphone_enabled: bool = True,
        system_enabled: bool = True,
        preserve_sources: bool = False,
    ) -> NativeRecordingResult:
        return self._call(
            lambda: self._start(
                microphone_key,
                speaker_key,
                microphone_enabled,
                system_enabled,
                preserve_sources,
            )
        )

    def _start(
        self,
        microphone_key: str | None,
        speaker_key: str | None,
        microphone_enabled: bool,
        system_enabled: bool,
        preserve_sources: bool,
    ) -> NativeRecordingResult:
        with self.devices.gate:
            if self._active_result is not None:
                if self._recorder.has_live_capture():
                    return replace(self._active_result, resumed=True)
                self._active_result = None
                self._selected = {}
                self._resources_active = False
                self.devices.set_capture_active(False)
            if not microphone_enabled and not system_enabled:
                raise AudioDeviceError("Activez le microphone ou le son du PC.")
            snapshot = self.devices.snapshot(force_refresh=True)
            microphone = (
                self.devices.resolve(microphone_key, "input", snapshot=snapshot)
                if microphone_enabled
                else None
            )
            speaker = (
                self.devices.resolve(speaker_key, "output", snapshot=snapshot)
                if system_enabled
                else None
            )
            self.devices.set_capture_active(True)
            try:
                recorder_kwargs = {
                    "microphone_enabled": microphone_enabled,
                    "system_enabled": system_enabled,
                }
                # Preserve the exact legacy recorder call when diarization is off;
                # this keeps old fake backends and behavior unchanged.
                if preserve_sources:
                    recorder_kwargs["preserve_sources"] = True
                result = self._recorder.start(
                    f"sd:{microphone.backend_index}" if microphone else None,
                    speaker.backend_id if speaker else None,
                    **recorder_kwargs,
                )
                self._selected = {}
                if microphone:
                    self._selected["microphone"] = microphone.stable_key
                if speaker:
                    self._selected["system"] = speaker.stable_key
                self._active_result = result
                self._last_completed = None
                self._pending_final = None
                return result
            finally:
                self._resources_active = self._recorder.has_live_capture()
                self.devices.set_capture_active(self._resources_active)

    def stop(self) -> NativeRecordingResult:
        return self._call(self._stop)

    def _persist_sources(self, result: NativeRecordingResult) -> tuple[Path | None, Path | None]:
        microphone = getattr(result, "microphone_source_path", None)
        system = getattr(result, "system_source_path", None)
        if microphone is None and system is None:
            return None, None
        parent = Path(self.paths.results) / "recording_sources"
        target = parent / result.recording_id
        staging = parent / f".{result.recording_id}.partial"
        parent.mkdir(parents=True, exist_ok=True)
        # A previous successful retry can reuse the already committed source set.
        if (target / "recording.json").is_file():
            mic = target / "microphone.wav"
            sys_path = target / "system.wav"
            return (mic if mic.is_file() else None, sys_path if sys_path.is_file() else None)
        shutil.rmtree(staging, ignore_errors=True)
        staging.mkdir(parents=True, exist_ok=True)
        persisted: dict[str, Path | None] = {"microphone": None, "system": None}
        try:
            for key, source in (("microphone", microphone), ("system", system)):
                if source is None or not Path(source).is_file():
                    continue
                destination = staging / f"{key}.wav"
                shutil.copyfile(source, destination)
                persisted[key] = destination
            if persisted["microphone"] is None and persisted["system"] is None:
                return None, None
            manifest = {
                "schema_version": 1,
                "recording_id": result.recording_id,
                "microphone_file": "microphone.wav" if persisted["microphone"] else None,
                "system_file": "system.wav" if persisted["system"] else None,
                "microphone_offset": float(getattr(result, "microphone_offset", 0.0) or 0.0),
                "system_offset": float(getattr(result, "system_offset", 0.0) or 0.0),
                "sample_rate": int(getattr(result, "sample_rate", 48000) or 48000),
            }
            (staging / "recording.json").write_text(
                json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            if target.exists():
                shutil.rmtree(target)
            staging.replace(target)
            mic = target / "microphone.wav"
            sys_path = target / "system.wav"
            return (mic if mic.is_file() else None, sys_path if sys_path.is_file() else None)
        except Exception:
            # The final mixed WAV remains valid. Keep temp source journals so a
            # storage error never turns a successful recording into data loss.
            logger.exception("Unable to persist diarization source tracks for %s", result.recording_id)
            return None, None
        finally:
            shutil.rmtree(staging, ignore_errors=True)

    def _stop(self) -> NativeRecordingResult:
        with self.devices.gate:
            if not self.is_active and self._last_completed is not None:
                return self._last_completed
            try:
                result = self._pending_final or self._recorder.stop(
                    self._active_result.recording_id if self._active_result else None
                )
                self._pending_final = result
                self.paths.results.mkdir(parents=True, exist_ok=True)
                destination = self.paths.results / result.path.name
                partial = destination.with_suffix(".wav.partial")
                try:
                    shutil.copyfile(result.path, partial)
                    partial.replace(destination)
                finally:
                    partial.unlink(missing_ok=True)
                persisted_mic, persisted_system = self._persist_sources(result)
                self._last_completed = NativeRecordingResult(
                    recording_id=result.recording_id,
                    path=destination,
                    duration=result.duration,
                    system_device_name=result.system_device_name,
                    microphone_device_name=result.microphone_device_name,
                    microphone_source_path=persisted_mic,
                    system_source_path=persisted_system,
                    microphone_offset=float(getattr(result, "microphone_offset", 0.0) or 0.0),
                    system_offset=float(getattr(result, "system_offset", 0.0) or 0.0),
                    sample_rate=int(getattr(result, "sample_rate", 48000) or 48000),
                )
                self._pending_final = None
                try:
                    result.path.unlink(missing_ok=True)
                except OSError:
                    pass
                # Source journals can be deleted only after their persistent
                # copies + manifest exist. If persistence failed, keep them.
                if persisted_mic is not None or persisted_system is not None:
                    for source in (
                        getattr(result, "microphone_source_path", None),
                        getattr(result, "system_source_path", None),
                    ):
                        if source is not None:
                            try:
                                Path(source).unlink(missing_ok=True)
                            except OSError:
                                pass
                return self._last_completed
            finally:
                live = self._recorder.has_live_capture()
                self._resources_active = live
                self.devices.set_capture_active(live)
                if not live:
                    self._active_result = None
                    self._selected = {}

    def levels(self) -> dict | None:
        active = self._active_result
        return self._recorder.get_levels(active.recording_id) if active else None

    def device_snapshot_changed(self, snapshot: AudioDeviceSnapshot) -> None:
        """Never substitute a different device into an already open stream."""
        with self.devices.gate:
            if snapshot is not self.devices.current_snapshot:
                return
            keys = {item.stable_key for item in snapshot.all}
            for source, key in tuple(self._selected.items()):
                if key not in keys:
                    self._recorder.source_unavailable(source)

    def recoverable_recordings(self) -> tuple[Path, ...]:
        active_id = self._active_result.recording_id if self._active_result else None
        return tuple(
            sorted(
                path
                for path in self._journal_dir.glob("*.wav")
                if (path.name.startswith(".native_recording_") or path.name.startswith("native_recording_"))
                and (not active_id or active_id not in path.name)
            )
        )

    def recover(self, path: Path) -> Path:
        source = Path(path).resolve()
        if source.parent != self._journal_dir.resolve() or source not in self.recoverable_recordings():
            raise AudioDeviceError("Ce fichier ne fait pas partie des enregistrements récupérables.")
        return recover_wav(source, self.paths.results)

    def close(self) -> None:
        if not self._closed:
            if self.is_active:
                self.stop()
            self._closed = True
            self._commands.put(None)
        self._worker.join(timeout=15)
        if self._worker.is_alive():
            raise AudioDeviceError("Le moteur audio n'a pas terminé son arrêt.")
