"""Windows native recorder: system loopback with SoundCard + microphone with sounddevice.
Designed for the Transcripteur Whisper application.
Why two backends?
- SoundCard is convenient for Windows WASAPI loopback (PC/system audio).
- python-sounddevice/PortAudio is used for physical microphones, avoiding
  SoundCard's known Windows/WASAPI single-channel recording issue.
Capture engine used by RecordingService:
    NativeRecorder(base_dir)
    .is_available()
    .list_devices()
    .start(microphone_id=None, speaker_id=None)
    .get_levels(recording_id)
    .stop(recording_id=None)
    .get_completed(recording_id)
"""

from __future__ import annotations

import math
import os
import queue
import re
import shutil
import sys
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

from transcripteur_whisper.audio.windows_com import com_apartment

_OUTPUT_CHANNELS = 2
_SYSTEM_CAPTURE_FRAMES = 2048
_SYSTEM_BLOCKSIZE = 8192
_QUEUE_MAX_BLOCKS = 512
_QUEUE_PUT_TIMEOUT = 0.1
_STOP_TIMEOUT = 10.0
_MIX_CHUNK_FRAMES = 65536
_METER_FLOOR_DB = -60.0
_METER_STALE_SECONDS = 0.8
_WRITER_FLUSH_SECONDS = 1.0
_RECORDING_RE = re.compile(r"native_recording_([0-9a-f]{32})\.wav\Z")
_SYSTEM_GAIN = max(0.0, float(os.getenv("WHISPER_SYSTEM_AUDIO_GAIN", "0.80")))
_MIC_GAIN = max(0.0, float(os.getenv("WHISPER_MICROPHONE_GAIN", "1.00")))
try:
    import numpy as np
except Exception:  # pragma: no cover
    np = None  # type: ignore
try:
    import soundfile as sf
except Exception:  # pragma: no cover
    sf = None  # type: ignore
try:
    import soundcard as sc
except Exception:  # pragma: no cover
    sc = None  # type: ignore
try:
    import sounddevice as sd
except Exception:  # pragma: no cover
    sd = None  # type: ignore


class NativeRecorderError(RuntimeError):
    pass


class NativeRecorderUnsupported(NativeRecorderError):
    pass


class NativeRecorderBusy(NativeRecorderError):
    pass


class NativeRecorderNotRunning(NativeRecorderError):
    pass


@dataclass(frozen=True)
class NativeRecordingResult:
    recording_id: str
    path: Path
    duration: float
    system_device_name: str = ""
    microphone_device_name: str = ""
    resumed: bool = False
    microphone_source_path: Path | None = None
    system_source_path: Path | None = None
    microphone_offset: float = 0.0
    system_offset: float = 0.0
    sample_rate: int = 48000


@dataclass
class _SourceState:
    key: str
    path: Path
    gain: float
    audio_queue: "queue.Queue[Any]" = field(default_factory=lambda: queue.Queue(maxsize=_QUEUE_MAX_BLOCKS))
    reader_done: threading.Event = field(default_factory=threading.Event)
    writer_ready: threading.Event = field(default_factory=threading.Event)
    writer_done: threading.Event = field(default_factory=threading.Event)
    writer_thread: Optional[threading.Thread] = None
    frames_written: int = 0
    first_frame_monotonic: Optional[float] = None
    last_frame_monotonic: Optional[float] = None
    enabled: bool = True
    failed: bool = False
    failure_role: str = ""
    failure_message: str = ""


class _RecordingSession:
    def __init__(
        self,
        recording_id: str,
        output_path: Path,
        microphone_index: int,
        microphone_name: str,
        speaker_id: str,
        speaker_name: str,
        microphone_enabled: bool = True,
        system_enabled: bool = True,
        preserve_sources: bool = False,
    ) -> None:
        if np is None or sf is None or sc is None or sd is None:
            raise NativeRecorderUnsupported(
                "Les modules numpy, soundfile, SoundCard et sounddevice sont requis."
            )
        self.recording_id = recording_id
        self.output_path = output_path
        self.microphone_index = int(microphone_index)
        self.microphone_name = microphone_name
        self.speaker_id = speaker_id
        self.speaker_name = speaker_name
        self.preserve_sources = bool(preserve_sources)
        self._stop_event = threading.Event()
        self._closing_system = threading.Event()
        self._state_lock = threading.RLock()
        self._samplerate = 48000
        self._mic_channels = 1
        self._start_ts: Optional[float] = None
        self._stop_ts: Optional[float] = None
        self._mic_stream: Optional[Any] = None
        self._system_context: Optional[Any] = None
        self._system_recorder: Optional[Any] = None
        self._system_reader_thread: Optional[threading.Thread] = None
        self._sources: Dict[str, _SourceState] = {
            "system": _SourceState(
                key="system",
                path=output_path.with_name(f".{output_path.stem}_system.wav"),
                gain=_SYSTEM_GAIN,
                enabled=system_enabled,
            ),
            "microphone": _SourceState(
                key="microphone",
                path=output_path.with_name(f".{output_path.stem}_microphone.wav"),
                gain=_MIC_GAIN,
                enabled=microphone_enabled,
            ),
        }
        self._levels: Dict[str, Dict[str, Any]] = {
            "system": {
                "rms_db": _METER_FLOOR_DB,
                "peak_db": _METER_FLOOR_DB,
                "clipped": False,
                "updated_monotonic": 0.0,
            },
            "microphone": {
                "rms_db": _METER_FLOOR_DB,
                "peak_db": _METER_FLOOR_DB,
                "clipped": False,
                "updated_monotonic": 0.0,
            },
        }

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    def start(self) -> None:
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self.output_path.unlink(missing_ok=True)
        for source in self._sources.values():
            source.path.unlink(missing_ok=True)
        mic_enabled = self._sources["microphone"].enabled
        mic_info = (
            sd.query_devices(self.microphone_index, "input")
            if mic_enabled
            else {"default_samplerate": 48000, "max_input_channels": 1}
        )
        default_rate = int(round(float(mic_info.get("default_samplerate") or 0)))
        max_input_channels = int(mic_info.get("max_input_channels") or 0)
        if max_input_channels <= 0:
            raise NativeRecorderUnsupported(
                f"Le périphérique « {self.microphone_name} » n'a aucun canal d'entrée."
            )
        rate_candidates: list[int] = []
        for rate in (default_rate, 48000, 44100):
            if rate > 0 and rate not in rate_candidates:
                rate_candidates.append(rate)
        channel_candidates = [1]
        if max_input_channels >= 2:
            channel_candidates.append(2)
        attempts: list[str] = []
        last_exc: Optional[BaseException] = None
        # We open the microphone with sounddevice and system loopback with
        # SoundCard at the same sample rate so no resampler is needed later.
        for rate in rate_candidates:
            for mic_channels in channel_candidates:
                self._reset_runtime_state()
                try:
                    self._samplerate = rate
                    self._mic_channels = mic_channels
                    if mic_enabled:
                        self._open_microphone_stream()
                    if self._sources["system"].enabled:
                        self._open_system_loopback()
                    self._start_writer_threads()
                    self._start_ts = time.monotonic()
                    if self._sources["system"].enabled:
                        self._start_system_reader()
                    if mic_enabled:
                        self._mic_stream.start()
                    self._ensure_at_least_one_source_available()
                    return
                except Exception as exc:
                    last_exc = exc
                    attempts.append(
                        f"{rate} Hz, micro={mic_channels} canal(aux): {type(exc).__name__}: {exc}"
                    )
                    self._cleanup_failed_attempt()
        recent = " | ".join(attempts[-6:])
        detail = f" Dernière erreur : {type(last_exc).__name__}: {last_exc}." if last_exc else ""
        raise NativeRecorderUnsupported(
            f"Impossible d'ouvrir le microphone « {self.microphone_name} » avec sounddevice/PortAudio."
            f"{detail} Essais : {recent}"
        ) from last_exc

    def stop(self) -> NativeRecordingResult:
        if self._start_ts is None:
            if not self.has_live_workers():
                raise NativeRecorderNotRunning("Aucun enregistrement natif en cours.")
            self._start_ts = time.monotonic()
        self._stop_ts = time.monotonic()
        self._stop_event.set()
        deadline = time.monotonic() + _STOP_TIMEOUT
        # A device may already be gone. Closing it must never prevent salvage.
        try:
            self._close_microphone_stream()
        except Exception as exc:
            self._mark_source_failure("microphone", "fermeture microphone", exc)
        finally:
            self._sources["microphone"].reader_done.set()
        reader = self._system_reader_thread
        if reader is not None and reader.is_alive():
            reader.join(timeout=self._remaining(deadline))
        try:
            self._close_system_loopback()
        except Exception as exc:
            self._mark_source_failure("system", "fermeture son système", exc)
        self._sources["system"].reader_done.set()
        stuck: list[str] = []
        if self._mic_stream is not None:
            stuck.append("flux microphone")
        if self._system_context is not None:
            stuck.append("flux système")
        if reader is not None and reader.is_alive():
            stuck.append("capture système")
        for key, source in self._sources.items():
            writer = source.writer_thread
            if writer is not None and writer.is_alive():
                writer.join(timeout=self._remaining(deadline))
            if writer is not None and writer.is_alive():
                stuck.append(f"écriture {key}")
        if stuck:
            # Important: do not delete anything. A second stop() can retry after
            # the blocked worker releases, and the temporary WAV files remain.
            raise NativeRecorderError(
                "Arrêt incomplet : threads toujours actifs (" + ", ".join(stuck) + "). "
                "Les données temporaires ont été conservées pour récupération."
            )
        # Mix every readable source that contains samples. A disconnected source
        # is simply absent from the remainder of the mix; the surviving source
        # continues to determine the final duration.
        try:
            duration = self._mix_sources()
        except Exception as mix_exc:
            # Last-resort path: if mixing itself fails, promote the longest valid
            # source journal to the final WAV instead of losing the recording.
            try:
                duration = self._salvage_best_source()
            except Exception as salvage_exc:
                raise NativeRecorderError(
                    "La finalisation a échoué, mais les pistes temporaires ont été conservées. "
                    f"Mix: {mix_exc}; récupération: {salvage_exc}"
                ) from mix_exc
        self._verify_final_output()
        source_meta = self._source_result_metadata() if self.preserve_sources else {}
        # Legacy behavior stays byte-for-byte equivalent when diarization is off.
        # When source preservation is enabled, RecordingService persists them
        # transactionally before deleting these journals.
        if not self.preserve_sources:
            self._discard_source_files()
        self._start_ts = None
        microphone = source_meta.get("microphone")
        system = source_meta.get("system")
        return NativeRecordingResult(
            recording_id=self.recording_id,
            path=self.output_path,
            duration=duration,
            system_device_name=self.speaker_name,
            microphone_device_name=self.microphone_name,
            microphone_source_path=microphone[0] if microphone else None,
            system_source_path=system[0] if system else None,
            microphone_offset=microphone[1] if microphone else 0.0,
            system_offset=system[1] if system else 0.0,
            sample_rate=self._samplerate,
        )

    def _source_result_metadata(self) -> Dict[str, tuple[Path, float]]:
        """Return validated source paths and their offsets in the mixed timeline."""
        try:
            details = self._usable_source_details()
        except NativeRecorderError:
            return {}
        known_starts = [detail["start"] for detail in details.values() if detail["start"] is not None]
        global_start = min(known_starts) if known_starts else (self._start_ts or time.monotonic())
        result: Dict[str, tuple[Path, float]] = {}
        for key, detail in details.items():
            source_start = detail["start"] if detail["start"] is not None else global_start
            offset = max(0.0, float(source_start - global_start))
            source_path = Path(detail["source"].path)
            if source_path.is_file():
                result[key] = (source_path, offset)
        return result

    def has_live_workers(self) -> bool:
        microphone_stream = self._mic_stream
        if microphone_stream is not None:
            try:
                if bool(microphone_stream.active) or not getattr(microphone_stream, "closed", False):
                    return True
            except Exception:
                # PortAudio may not expose a reliable state while closing.
                return True
        if getattr(self, "_system_context", None) is not None:
            return True
        if self._system_reader_thread is not None and self._system_reader_thread.is_alive():
            return True
        for source in self._sources.values():
            if source.writer_thread is not None and source.writer_thread.is_alive():
                return True
        return False

    def is_recording(self) -> bool:
        return self._start_ts is not None and not self._stop_event.is_set() and self.has_live_workers()

    def discard_output(self) -> None:
        self.output_path.unlink(missing_ok=True)
        self._discard_source_files()

    # ------------------------------------------------------------------
    # Device opening
    # ------------------------------------------------------------------
    def _open_microphone_stream(self) -> None:
        try:
            sd.check_input_settings(
                device=self.microphone_index,
                channels=self._mic_channels,
                samplerate=self._samplerate,
                dtype="float32",
            )
            self._mic_stream = sd.InputStream(
                device=self.microphone_index,
                samplerate=self._samplerate,
                channels=self._mic_channels,
                dtype="float32",
                blocksize=0,
                latency="high",
                callback=self._microphone_callback,
            )
        except Exception as exc:
            raise NativeRecorderUnsupported(
                f"PortAudio refuse « {self.microphone_name} » "
                f"({self._samplerate} Hz, {self._mic_channels} canal(aux)) : {exc}"
            ) from exc

    def _open_system_loopback(self) -> None:
        try:
            # Exact endpoint identity only. SoundCard's public get_speaker()
            # permits fuzzy matching, which could silently change the source.
            speaker = next((item for item in sc.all_speakers() if str(item.id) == self.speaker_id), None)
            if speaker is None:
                raise NativeRecorderUnsupported("La sortie sélectionnée a été déconnectée.")
            loopback = sc.get_microphone(
                id=getattr(speaker, "id", self.speaker_id),
                include_loopback=True,
            )
            channels = _device_channel_count(loopback)
            if channels <= 0:
                raise NativeRecorderUnsupported(
                    f"La sortie « {self.speaker_name} » n'expose aucun canal loopback."
                )
            channel_map = list(range(min(_OUTPUT_CHANNELS, channels)))
            self._system_context = loopback.recorder(
                samplerate=self._samplerate,
                channels=channel_map,
                blocksize=_SYSTEM_BLOCKSIZE,
                exclusive_mode=False,
            )
            self._system_recorder = self._system_context.__enter__()
        except Exception as exc:
            raise NativeRecorderUnsupported(
                f"Impossible d'ouvrir le loopback du son PC « {self.speaker_name} » "
                f"à {self._samplerate} Hz : {exc}"
            ) from exc

    def _close_microphone_stream(self) -> None:
        stream = self._mic_stream
        if stream is None:
            return
        try:
            if getattr(stream, "active", False):
                stream.stop()
        finally:
            stream.close()
            self._mic_stream = None

    def _close_system_loopback(self) -> None:
        context = self._system_context
        if context is None:
            return
        reader = self._system_reader_thread
        if reader is not None and reader.is_alive():
            # SoundCard __exit__ releases raw CFFI COM pointers. Releasing them
            # concurrently with record() can crash the process. Keep ownership
            # and the recovery journals until the reader returns; stop can retry.
            raise NativeRecorderBusy("Le lecteur système utilise encore le flux audio.")
        self._closing_system.set()
        context.__exit__(None, None, None)
        self._system_context = None
        self._system_recorder = None

    # ------------------------------------------------------------------
    # Capture
    # ------------------------------------------------------------------
    def _microphone_callback(self, indata, frames, time_info, status) -> None:
        source = self._sources["microphone"]
        if self._stop_event.is_set():
            return
        if source.failed:
            raise sd.CallbackAbort
        try:
            data = np.asarray(indata, dtype="float32").copy()
            if data.ndim == 1:
                data = data[:, None]
            if data.ndim != 2 or data.shape[0] <= 0:
                return
            mono = np.mean(data, axis=1, dtype="float32")
            stereo = np.column_stack((mono, mono)).astype("float32", copy=False)
            now = time.monotonic()
            estimated_start = now - (stereo.shape[0] / self._samplerate)
            with self._state_lock:
                if source.first_frame_monotonic is None:
                    source.first_frame_monotonic = estimated_start
                source.last_frame_monotonic = now
            self._update_meter("microphone", stereo * source.gain)
            try:
                source.audio_queue.put_nowait(stereo)
            except queue.Full as exc:
                self._mark_source_failure(
                    "microphone",
                    "capture microphone",
                    NativeRecorderError("La file audio microphone est saturée."),
                )
                raise sd.CallbackAbort from exc
        except sd.CallbackAbort:
            raise
        except Exception as exc:
            self._mark_source_failure("microphone", "capture microphone", exc)
            raise sd.CallbackAbort from exc

    def _start_system_reader(self) -> None:
        self._system_reader_thread = threading.Thread(
            target=self._system_reader_loop,
            daemon=True,
            name="native-system-loopback-reader",
        )
        self._system_reader_thread.start()

    def _system_reader_loop(self) -> None:
        with com_apartment():
            self._system_reader_loop_com()

    def _system_reader_loop_com(self) -> None:
        source = self._sources["system"]
        try:
            while not self._stop_event.is_set() and not source.failed:
                recorder = self._system_recorder
                if recorder is None:
                    break
                chunk = recorder.record(numframes=_SYSTEM_CAPTURE_FRAMES)
                now = time.monotonic()
                if chunk is None or getattr(chunk, "size", 0) == 0:
                    continue
                data = self._normalise_system_stereo(chunk)
                estimated_start = now - (data.shape[0] / self._samplerate)
                with self._state_lock:
                    if source.first_frame_monotonic is None:
                        source.first_frame_monotonic = estimated_start
                    source.last_frame_monotonic = now
                self._update_meter("system", data * source.gain)
                while not self._stop_event.is_set() and not source.failed:
                    try:
                        source.audio_queue.put(data, timeout=_QUEUE_PUT_TIMEOUT)
                        break
                    except queue.Full:
                        continue
        except Exception as exc:
            if not self._closing_system.is_set() and not self._stop_event.is_set():
                self._mark_source_failure("system", "capture système", exc)
        finally:
            source.reader_done.set()

    def _normalise_system_stereo(self, chunk: Any) -> Any:
        data = np.asarray(chunk, dtype="float32")
        if data.ndim == 1:
            data = data[:, None]
        if data.ndim != 2 or data.shape[0] <= 0 or data.shape[1] <= 0:
            raise NativeRecorderError(f"Bloc loopback invalide : {data.shape}.")
        if data.shape[1] == 1:
            return np.repeat(data, 2, axis=1)
        return data[:, :2]

    # ------------------------------------------------------------------
    # Writers
    # ------------------------------------------------------------------
    def _start_writer_threads(self) -> None:
        for key, source in self._sources.items():
            if not source.enabled:
                source.reader_done.set()
                source.writer_done.set()
                continue
            source.writer_thread = threading.Thread(
                target=self._writer_loop,
                args=(key,),
                daemon=True,
                name=f"native-audio-writer-{key}",
            )
            source.writer_thread.start()
        for source in self._sources.values():
            if not source.enabled:
                continue
            if not source.writer_ready.wait(timeout=_STOP_TIMEOUT):
                self._mark_source_failure(
                    source.key,
                    f"écriture {source.key}",
                    NativeRecorderError(f"Le writer audio « {source.key} » ne s'est pas initialisé."),
                )
        self._ensure_at_least_one_source_available()

    def _writer_loop(self, key: str) -> None:
        source = self._sources[key]
        try:
            with sf.SoundFile(
                source.path,
                mode="w",
                samplerate=self._samplerate,
                channels=_OUTPUT_CHANNELS,
                subtype="PCM_16",
            ) as output_file:
                source.writer_ready.set()
                last_flush = time.monotonic()
                while True:
                    try:
                        data = source.audio_queue.get(timeout=0.2)
                    except queue.Empty:
                        if source.reader_done.is_set():
                            break
                        continue
                    try:
                        array = np.asarray(data, dtype="float32")
                        if array.ndim != 2 or array.shape[1] != 2:
                            raise NativeRecorderError(f"Bloc {key} inattendu : {array.shape}.")
                        output_file.write(np.clip(array, -1.0, 1.0))
                        with self._state_lock:
                            source.frames_written += int(array.shape[0])
                        now = time.monotonic()
                        if now - last_flush >= _WRITER_FLUSH_SECONDS:
                            output_file.flush()
                            last_flush = now
                    finally:
                        source.audio_queue.task_done()
                output_file.flush()
        except Exception as exc:
            self._mark_source_failure(key, f"écriture {key}", exc)
            # Stop only this source. The other source remains recordable.
            source.reader_done.set()
        finally:
            source.writer_ready.set()
            source.writer_done.set()

    def _update_meter(self, key: str, data: Any) -> None:
        absolute = np.abs(np.asarray(data, dtype="float32"))
        if absolute.size == 0:
            return
        peak = float(np.max(absolute))
        rms = float(np.sqrt(np.mean(np.square(absolute, dtype="float64"))))
        peak_db = self._to_db(peak)
        rms_db = self._to_db(rms)
        with self._state_lock:
            previous = self._levels[key]
            if previous["updated_monotonic"]:
                peak_db = max(peak_db, float(previous["peak_db"]) - 3.0)
                rms_db = 0.35 * rms_db + 0.65 * float(previous["rms_db"])
            self._levels[key] = {
                "rms_db": max(_METER_FLOOR_DB, rms_db),
                "peak_db": max(_METER_FLOOR_DB, peak_db),
                "clipped": peak >= 0.999,
                "updated_monotonic": time.monotonic(),
            }

    @staticmethod
    def _to_db(value: float) -> float:
        if value <= 0:
            return _METER_FLOOR_DB
        return max(_METER_FLOOR_DB, 20.0 * math.log10(value))

    def levels_snapshot(self) -> Dict[str, Any]:
        now = time.monotonic()
        with self._state_lock:
            sources: Dict[str, Any] = {}
            for key, source in self._sources.items():
                meter = dict(self._levels[key])
                updated = float(meter["updated_monotonic"] or 0.0)
                age = (now - updated) if updated else None
                stale = age is None or age > _METER_STALE_SECONDS
                peak = _METER_FLOOR_DB if stale else float(meter["peak_db"])
                rms = _METER_FLOOR_DB if stale else float(meter["rms_db"])
                sources[key] = {
                    "name": self.speaker_name if key == "system" else self.microphone_name,
                    "gain": source.gain,
                    "rms_db": round(rms, 1),
                    "peak_db": round(peak, 1),
                    "clipped": False if stale else bool(meter["clipped"]),
                    "signal": peak > -50.0,
                    "age_ms": None if age is None else int(max(0.0, age) * 1000),
                    "frames_captured": int(source.frames_written),
                    "enabled": source.enabled,
                    "available": source.enabled and not source.failed,
                    "failed": source.enabled and bool(source.failed),
                    "error": source.failure_message or None,
                }
            elapsed = 0.0
            if self._start_ts is not None:
                elapsed = max(0.0, (self._stop_ts or now) - self._start_ts)
            available_sources = sum(
                1 for source in self._sources.values() if source.enabled and not source.failed
            )
            enabled_sources = sum(source.enabled for source in self._sources.values())
            return {
                "recording_id": self.recording_id,
                "active": not self._stop_event.is_set(),
                "degraded": available_sources < enabled_sources,
                "available_sources": available_sources,
                "sample_rate": self._samplerate,
                "elapsed": elapsed,
                "sources": sources,
            }

    def _usable_source_details(self) -> Dict[str, Dict[str, Any]]:
        details: Dict[str, Dict[str, Any]] = {}
        for key, source in self._sources.items():
            try:
                info = sf.info(str(source.path))
            except Exception:
                continue
            frames = int(info.frames or 0)
            if frames <= 0:
                continue
            if int(info.samplerate or 0) != self._samplerate or int(info.channels or 0) != 2:
                continue
            source_start = source.first_frame_monotonic
            if source_start is None:
                source_start = self._start_ts
            details[key] = {
                "source": source,
                "frames": frames,
                "start": source_start,
            }
        if not details:
            errors = [
                f"{key}: {source.failure_message}"
                for key, source in self._sources.items()
                if source.failure_message
            ]
            suffix = " | ".join(errors)
            raise NativeRecorderError(
                "Aucun échantillon audio récupérable n'a été enregistré."
                + (f" Détails : {suffix}" if suffix else "")
            )
        return details

    def _mix_sources(self) -> float:
        details = self._usable_source_details()
        known_starts = [detail["start"] for detail in details.values() if detail["start"] is not None]
        global_start = min(known_starts) if known_starts else (self._start_ts or time.monotonic())
        total_frames = 0
        for detail in details.values():
            source_start = detail["start"] if detail["start"] is not None else global_start
            offset = max(0, int(round((source_start - global_start) * self._samplerate)))
            detail["offset"] = offset
            total_frames = max(total_frames, offset + int(detail["frames"]))
        opened: Dict[str, Any] = {}
        self.output_path.unlink(missing_ok=True)
        try:
            for key, detail in details.items():
                opened[key] = sf.SoundFile(detail["source"].path, mode="r")
            with sf.SoundFile(
                self.output_path,
                mode="w",
                samplerate=self._samplerate,
                channels=2,
                subtype="PCM_16",
            ) as output:
                for output_start in range(0, total_frames, _MIX_CHUNK_FRAMES):
                    count = min(_MIX_CHUNK_FRAMES, total_frames - output_start)
                    mixed = np.zeros((count, 2), dtype="float32")
                    for key, detail in details.items():
                        source_start = int(detail["offset"])
                        source_end = source_start + int(detail["frames"])
                        overlap_start = max(output_start, source_start)
                        overlap_end = min(output_start + count, source_end)
                        if overlap_end <= overlap_start:
                            continue
                        handle = opened[key]
                        handle.seek(overlap_start - source_start)
                        data = handle.read(
                            overlap_end - overlap_start,
                            dtype="float32",
                            always_2d=True,
                        )
                        destination = overlap_start - output_start
                        mixed[destination : destination + data.shape[0], :] += (
                            data[:, :2] * detail["source"].gain
                        )
                    peak = float(np.max(np.abs(mixed))) if mixed.size else 0.0
                    if peak > 0.999:
                        mixed *= 0.999 / peak
                    output.write(np.clip(mixed, -1.0, 1.0))
                output.flush()
        except Exception:
            # Never touch source files here. They are the recovery journal.
            self.output_path.unlink(missing_ok=True)
            raise
        finally:
            for handle in opened.values():
                try:
                    handle.close()
                except Exception:
                    pass
        info = sf.info(str(self.output_path))
        return int(info.frames) / max(self._samplerate, 1)

    def _salvage_best_source(self) -> float:
        candidates: list[tuple[int, Path]] = []
        for source in self._sources.values():
            try:
                info = sf.info(str(source.path))
            except Exception:
                continue
            frames = int(info.frames or 0)
            if frames > 0 and int(info.samplerate or 0) == self._samplerate and int(info.channels or 0) == 2:
                candidates.append((frames, source.path))
        if not candidates:
            raise NativeRecorderError("Aucune piste temporaire exploitable à récupérer.")
        frames, source_path = max(candidates, key=lambda item: item[0])
        self.output_path.unlink(missing_ok=True)
        shutil.copyfile(source_path, self.output_path)
        return frames / max(self._samplerate, 1)

    def _verify_final_output(self) -> None:
        try:
            info = sf.info(str(self.output_path))
        except Exception as exc:
            raise NativeRecorderError(
                "Le WAV final est illisible. Les pistes temporaires ont été conservées."
            ) from exc
        if int(info.frames or 0) <= 0:
            raise NativeRecorderError("Le WAV final est vide. Les pistes temporaires ont été conservées.")
        if int(info.samplerate or 0) != self._samplerate or int(info.channels or 0) != 2:
            raise NativeRecorderError(
                "Le WAV final a un format incohérent. Les pistes temporaires ont été conservées."
            )

    def _reset_runtime_state(self) -> None:
        self._stop_event.clear()
        self._closing_system.clear()
        self._system_reader_thread = None
        self._mic_stream = None
        self._system_context = None
        self._system_recorder = None
        for source in self._sources.values():
            source.audio_queue = queue.Queue(maxsize=_QUEUE_MAX_BLOCKS)
            source.reader_done.clear()
            source.writer_ready.clear()
            source.writer_done.clear()
            source.writer_thread = None
            source.frames_written = 0
            source.first_frame_monotonic = None
            source.last_frame_monotonic = None
            source.failed = not source.enabled
            source.failure_role = ""
            source.failure_message = ""
            source.path.unlink(missing_ok=True)

    def _cleanup_failed_attempt(self) -> None:
        self._stop_event.set()
        try:
            self._close_microphone_stream()
        except Exception:
            pass
        self._sources["microphone"].reader_done.set()
        self._sources["system"].reader_done.set()
        deadline = time.monotonic() + 2.0
        if self._system_reader_thread is not None and self._system_reader_thread.is_alive():
            self._system_reader_thread.join(timeout=self._remaining(deadline))
        try:
            self._close_system_loopback()
        except Exception:
            pass
        for source in self._sources.values():
            if source.writer_thread is not None and source.writer_thread.is_alive():
                source.writer_thread.join(timeout=self._remaining(deadline))
        if self.has_live_workers():
            raise NativeRecorderBusy(
                "Une source reste active après un échec de démarrage. "
                "Arrêtez la capture ; ses pistes de récupération sont conservées."
            )
        if any(source.frames_written > 0 for source in self._sources.values()):
            raise NativeRecorderError(
                "Le démarrage a échoué après capture audio. Les pistes récupérables sont conservées."
            )
        self._discard_source_files()
        self._start_ts = None

    def _discard_source_files(self) -> None:
        for source in self._sources.values():
            try:
                source.path.unlink(missing_ok=True)
            except OSError:
                pass

    def _mark_source_failure(self, key: str, role: str, exc: BaseException) -> None:
        source = self._sources[key]
        with self._state_lock:
            if not source.failed:
                source.failed = True
                source.failure_role = role
                source.failure_message = f"{type(exc).__name__}: {exc}"
        # Close/finalize only this source's journal as soon as the capture dies.
        # Its writer drains anything already queued, then exits; the other source
        # is deliberately left untouched.
        source.reader_done.set()

    def _ensure_at_least_one_source_available(self) -> None:
        if any(source.enabled and not source.failed for source in self._sources.values()):
            return
        details = " | ".join(
            f"{key}: {source.failure_message or 'indisponible'}" for key, source in self._sources.items()
        )
        raise NativeRecorderError("Aucune source audio n'est encore disponible. " + details)

    @staticmethod
    def _remaining(deadline: float) -> float:
        return max(0.0, deadline - time.monotonic())


class NativeRecorder:
    def __init__(self, base_dir: Path):
        self._base_dir = Path(base_dir)
        self._base_dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._active: Optional[_RecordingSession] = None
        self._completed: Dict[str, NativeRecordingResult] = {}
        self._purge_old_files_locked()

    def is_available(self) -> bool:
        if sys.platform != "win32" or np is None or sf is None or sc is None or sd is None:
            return False
        try:
            return bool(self._wasapi_microphones()) and sc.default_speaker() is not None
        except Exception:
            return False

    def list_devices(self) -> Dict[str, Any]:
        self._require_backend()
        microphones = self._wasapi_microphones()
        if not microphones:
            raise NativeRecorderUnsupported(
                "Aucun microphone Windows WASAPI n'a été détecté par sounddevice."
            )
        try:
            speakers = list(sc.all_speakers())
            default_speaker = sc.default_speaker()
        except Exception as exc:
            raise NativeRecorderUnsupported(
                "Impossible de lister les sorties audio Windows avec SoundCard."
            ) from exc
        default_speaker_id = str(getattr(default_speaker, "id", ""))
        speaker_payload = []
        for speaker in speakers:
            speaker_payload.append(
                {
                    "id": str(getattr(speaker, "id", "")),
                    "name": str(getattr(speaker, "name", "Sortie audio")),
                    "channels": _device_channel_count(speaker),
                    "default": str(getattr(speaker, "id", "")) == default_speaker_id,
                }
            )
        return {"microphones": microphones, "speakers": speaker_payload}

    def start(
        self,
        microphone_id: Optional[str] = None,
        speaker_id: Optional[str] = None,
        *,
        microphone_enabled: bool = True,
        system_enabled: bool = True,
        preserve_sources: bool = False,
    ) -> NativeRecordingResult:
        # Opening WASAPI can take a moment. Serializing the full start avoids a
        # second request mistaking a half-created session for an abandoned one.
        with self._lock:
            active = self._active
            if active is not None:
                if active.is_recording():
                    # Idempotent retry: recover the id after a lost response,
                    # a double click or a page reload during capture.
                    return NativeRecordingResult(
                        recording_id=active.recording_id,
                        path=active.output_path,
                        duration=0.0,
                        system_device_name=active.speaker_name,
                        microphone_device_name=active.microphone_name,
                        resumed=True,
                    )
                if active.has_live_workers():
                    raise NativeRecorderBusy(
                        "La finalisation de l'enregistrement précédent est encore en cours."
                    )
                # A failed finalization used to leave this dead object active
                # forever. Keep its temporary WAVs, but release the coordinator.
                self._active = None
            self._purge_old_files_locked()
            if not microphone_enabled and not system_enabled:
                raise NativeRecorderUnsupported("Activez au moins une source audio.")
            mic_index, mic_name = (
                self._resolve_microphone(microphone_id) if microphone_enabled else (-1, "Désactivé")
            )
            speaker_id_resolved, speaker_name = (
                self._resolve_speaker(speaker_id) if system_enabled else ("", "Désactivé")
            )
            recording_id = uuid.uuid4().hex
            output_path = self._base_dir / f"native_recording_{recording_id}.wav"
            session = _RecordingSession(
                recording_id=recording_id,
                output_path=output_path,
                microphone_index=mic_index,
                microphone_name=mic_name,
                speaker_id=speaker_id_resolved,
                speaker_name=speaker_name,
                microphone_enabled=microphone_enabled,
                system_enabled=system_enabled,
                preserve_sources=preserve_sources,
            )
            self._active = session
            try:
                session.start()
            except Exception:
                if self._active is session and not session.has_live_workers():
                    self._active = None
                # Failed-attempt cleanup already removes empty attempts. Any
                # remaining journals contain samples or belong to live workers.
                raise
            return NativeRecordingResult(
                recording_id=recording_id,
                path=output_path,
                duration=0.0,
                system_device_name=speaker_name,
                microphone_device_name=mic_name,
            )

    def stop(self, recording_id: Optional[str] = None) -> NativeRecordingResult:
        # A retried stop must not mix/write the same output concurrently.
        with self._lock:
            session = self._active
            if session is None:
                completed = self._completed.get(recording_id or "")
                if completed is not None:
                    return completed
                raise NativeRecorderNotRunning("Aucun enregistrement natif en cours.")
            if recording_id and recording_id != session.recording_id:
                completed = self._completed.get(recording_id)
                if completed is not None:
                    return completed
                raise NativeRecorderNotRunning("Identifiant d'enregistrement invalide.")
            try:
                result = session.stop()
            except Exception:
                # Keep a genuinely live worker attached so stop() can be
                # retried. Once capture resources are dead, release the slot
                # while preserving the temporary recovery WAVs.
                if self._active is session and not session.has_live_workers():
                    self._active = None
                raise
            if self._active is session:
                self._active = None
            self._completed[result.recording_id] = result
            return result

    def get_levels(self, recording_id: str) -> Optional[Dict[str, Any]]:
        # Atomic reference read: meters must never wait behind slow stop/mix.
        session = self._active
        if session is None or session.recording_id != recording_id:
            return None
        return session.levels_snapshot()

    def get_completed(self, recording_id: str) -> Optional[NativeRecordingResult]:
        with self._lock:
            self._purge_old_files_locked()
            return self._completed.get(recording_id)

    # ------------------------------------------------------------------
    # Device discovery
    # ------------------------------------------------------------------
    def _require_backend(self) -> None:
        if sys.platform != "win32":
            raise NativeRecorderUnsupported("L'enregistrement natif est disponible sous Windows.")
        missing = []
        if np is None:
            missing.append("numpy")
        if sf is None:
            missing.append("soundfile")
        if sc is None:
            missing.append("SoundCard")
        if sd is None:
            missing.append("sounddevice")
        if missing:
            raise NativeRecorderUnsupported("Dépendances audio manquantes : " + ", ".join(missing) + ".")

    def _wasapi_microphones(self) -> list[Dict[str, Any]]:
        self._require_backend()
        devices = sd.query_devices()
        hostapis = sd.query_hostapis()
        default_name = ""
        try:
            default_input = sd.query_devices(kind="input")
            default_name = str(default_input.get("name") or "")
        except Exception:
            pass
        payload: list[Dict[str, Any]] = []
        for index, device in enumerate(devices):
            max_inputs = int(device.get("max_input_channels") or 0)
            if max_inputs <= 0:
                continue
            hostapi_index = int(device.get("hostapi") or 0)
            try:
                hostapi_name = str(hostapis[hostapi_index].get("name") or "")
            except Exception:
                hostapi_name = ""
            if "WASAPI" not in hostapi_name.upper():
                continue
            name = str(device.get("name") or f"Microphone {index}")
            payload.append(
                {
                    "id": f"sd:{index}",
                    "name": name,
                    "channels": max_inputs,
                    "sample_rate": int(round(float(device.get("default_samplerate") or 0))),
                    "host_api": hostapi_name,
                    "default": bool(default_name and _same_device_name(name, default_name)),
                }
            )
        if payload and not any(item["default"] for item in payload):
            payload[0]["default"] = True
        return payload

    def _resolve_microphone(self, microphone_id: Optional[str]) -> tuple[int, str]:
        microphones = self._wasapi_microphones()
        if not microphones:
            raise NativeRecorderUnsupported("Aucun microphone WASAPI disponible.")
        if microphone_id:
            token = str(microphone_id).strip()
            if token.startswith("sd:"):
                token = token[3:]
            try:
                wanted = int(token)
            except ValueError:
                wanted = -1
            for mic in microphones:
                if int(str(mic["id"]).split(":", 1)[1]) == wanted:
                    return wanted, str(mic["name"])
            raise NativeRecorderUnsupported(
                "Le microphone sélectionné n'est plus disponible. Sélectionnez un périphérique disponible."
            )
        selected = next((mic for mic in microphones if mic.get("default")), microphones[0])
        index = int(str(selected["id"]).split(":", 1)[1])
        return index, str(selected["name"])

    def _resolve_speaker(self, speaker_id: Optional[str]) -> tuple[str, str]:
        try:
            if speaker_id:
                speakers = list(sc.all_speakers())
                speaker = next(
                    (item for item in speakers if str(getattr(item, "id", "")) == str(speaker_id)),
                    None,
                )
                if speaker is None:
                    raise NativeRecorderUnsupported(
                        "La sortie audio sélectionnée n'est plus disponible. Sélectionnez un périphérique disponible."
                    )
            else:
                speaker = sc.default_speaker()
            if speaker is None:
                raise NativeRecorderUnsupported("Aucune sortie audio Windows disponible.")
            return str(getattr(speaker, "id", "")), str(getattr(speaker, "name", "Son du PC"))
        except NativeRecorderUnsupported:
            raise
        except Exception as exc:
            raise NativeRecorderUnsupported("Impossible de sélectionner la sortie audio Windows.") from exc

    def _purge_old_files_locked(self, max_age_seconds: float = 24 * 3600.0) -> None:
        # Source journals and completed audio may be the only copy after a
        # crash. Age alone is never evidence that deletion is safe.
        return

    def source_unavailable(self, source_key: str) -> None:
        session = self._active
        if session is None:
            return
        source = session._sources[source_key]
        if source.enabled and not source.failed:
            session._mark_source_failure(
                source_key,
                "périphérique déconnecté",
                NativeRecorderError("La source audio a été déconnectée ou désactivée."),
            )

    def has_live_capture(self) -> bool:
        session = self._active
        return session is not None and session.has_live_workers()


def _device_channel_count(device: Any) -> int:
    raw = getattr(device, "channels", 0)
    if isinstance(raw, int):
        return raw
    try:
        return len(raw)
    except Exception:
        return 0


def _normalise_name(value: str) -> str:
    text = str(value or "").casefold()
    for token in ("windows wasapi", "windows directsound", "mme", "wdm-ks"):
        text = text.replace(token, "")
    return " ".join(text.replace("®", "").split())


def _same_device_name(left: str, right: str) -> bool:
    a = _normalise_name(left)
    b = _normalise_name(right)
    return bool(a and b and (a == b or a in b or b in a))
