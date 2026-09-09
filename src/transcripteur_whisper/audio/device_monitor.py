"""Qt device events trigger backend enumeration without blocking the GUI."""

from __future__ import annotations

from typing import Any

from PySide6.QtCore import QObject, QRunnable, QThreadPool, QTimer, Signal, Slot
from PySide6.QtMultimedia import QMediaDevices

from transcripteur_whisper.services.audio_device_service import AudioDeviceService


class _ScanSignals(QObject):
    result = Signal(object)
    error = Signal(str)
    finished = Signal()


class _Scan(QRunnable):
    def __init__(self, devices: AudioDeviceService, recording: Any):
        super().__init__()
        self.signals = _ScanSignals()
        self.devices = devices
        self.recording = recording

    @Slot()
    def run(self) -> None:
        try:
            snapshot = self.devices.snapshot()
            if self.recording is not None:
                self.recording.device_snapshot_changed(snapshot)
            self.signals.result.emit(snapshot)
        except Exception as exc:
            self.signals.error.emit(str(exc))
        finally:
            self.signals.finished.emit()


class AudioDeviceMonitor(QObject):
    devices_changed = Signal(object)
    error = Signal(str)
    finished = Signal()

    def __init__(
        self,
        devices: AudioDeviceService,
        recording: Any = None,
        parent: QObject | None = None,
        *,
        media_devices: QObject | None = None,
        debounce_ms: int = 400,
        poll_ms: int = 2500,
    ):
        super().__init__(parent)
        self.devices = devices
        self.recording = recording
        # This instance must remain alive for Windows device notifications.
        self._media_devices = media_devices if media_devices is not None else QMediaDevices(self)
        self._media_devices.audioInputsChanged.connect(self._schedule)
        self._media_devices.audioOutputsChanged.connect(self._schedule)
        self._debounce = QTimer(self)
        self._debounce.setSingleShot(True)
        self._debounce.setInterval(debounce_ms)
        self._debounce.timeout.connect(self._request_scan)
        self._poll = QTimer(self)
        self._poll.setInterval(poll_ms)
        self._poll.timeout.connect(self._request_scan)
        self._pool = QThreadPool(self)
        self._pool.setMaxThreadCount(1)
        self._running = False
        self._task: _Scan | None = None
        self._pending = False
        self._last_snapshot = None
        self._last_error = ""

    @Slot()
    def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._poll.start()
        self._request_scan()

    @Slot()
    def stop(self) -> None:
        self._running = False
        self._debounce.stop()
        self._poll.stop()
        self._pending = False
        if self._task is None:
            self.finished.emit()

    @Slot()
    def _schedule(self) -> None:
        if self._running:
            self._debounce.start()

    @Slot()
    def _request_scan(self) -> None:
        if not self._running:
            return
        if self._task is not None:
            self._pending = True
            return
        self._task = _Scan(self.devices, self.recording)
        self._task.signals.result.connect(self._scan_result)
        self._task.signals.error.connect(self._scan_error)
        self._task.signals.finished.connect(self._scan_finished)
        self._pool.start(self._task)

    @Slot(object)
    def _scan_result(self, snapshot) -> None:
        self._last_error = ""
        if self._running and snapshot != self._last_snapshot:
            self._last_snapshot = snapshot
            self.devices_changed.emit(snapshot)

    @Slot(str)
    def _scan_error(self, message: str) -> None:
        if self._running and message != self._last_error:
            self._last_error = message
            self.error.emit(message)

    @Slot()
    def _scan_finished(self) -> None:
        self._task = None
        if not self._running:
            self.finished.emit()
        elif self._pending:
            self._pending = False
            self._request_scan()


DeviceMonitor = AudioDeviceMonitor
