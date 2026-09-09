from __future__ import annotations

import threading

from PySide6.QtCore import QObject, QTimer, Signal

from tests.test_audio_devices import FakeSoundCard, FakeSoundDevice, endpoint, pa_device
from transcripteur_whisper.audio.device_monitor import AudioDeviceMonitor
from transcripteur_whisper.models.audio_devices import AudioDeviceSnapshot
from transcripteur_whisper.services.audio_device_service import AudioDeviceService


class FakeMediaDevices(QObject):
    audioInputsChanged = Signal()
    audioOutputsChanged = Signal()


def test_qt_event_burst_is_debounced_and_only_changes_notify(qtbot):
    sd, sc = FakeSoundDevice(), FakeSoundCard()
    devices = AudioDeviceService(sounddevice=sd, soundcard=sc)
    events = FakeMediaDevices()
    monitor = AudioDeviceMonitor(devices, media_devices=events, debounce_ms=40, poll_ms=10000)
    notifications = []
    monitor.devices_changed.connect(notifications.append)
    monitor.start()
    qtbot.waitUntil(lambda: len(notifications) == 1 and monitor._task is None)
    initial_scans = len(sd.threads)
    sc.microphones.append(endpoint("usb-new", "New USB"))
    sd.current.append(pa_device("New USB"))
    for _ in range(8):
        events.audioInputsChanged.emit()
        events.audioOutputsChanged.emit()
    qtbot.waitUntil(lambda: len(notifications) == 2 and monitor._task is None)
    assert len(sd.threads) == initial_scans + 1
    events.audioInputsChanged.emit()
    qtbot.wait(100)
    assert len(notifications) == 2
    assert all(thread_id != threading.get_ident() for thread_id in sd.threads)
    with qtbot.waitSignal(monitor.finished):
        monitor.stop()


def test_fallback_poll_detects_unplug_without_os_events(qtbot):
    sd, sc = FakeSoundDevice(), FakeSoundCard()
    devices = AudioDeviceService(sounddevice=sd, soundcard=sc)
    monitor = AudioDeviceMonitor(devices, media_devices=FakeMediaDevices(), poll_ms=50)
    snapshots = []
    monitor.devices_changed.connect(snapshots.append)
    monitor.start()
    qtbot.waitUntil(lambda: len(snapshots) == 1)
    sc.microphones.clear()
    sc.default_input = None
    sd.current.clear()
    qtbot.waitUntil(lambda: len(snapshots) == 2)
    assert snapshots[-1].inputs == ()
    with qtbot.waitSignal(monitor.finished):
        monitor.stop()


def test_slow_scan_keeps_gui_responsive_and_stop_waits_asynchronously(qtbot):
    entered, release = threading.Event(), threading.Event()

    class SlowDevices:
        def snapshot(self):
            entered.set()
            release.wait(3)
            return AudioDeviceSnapshot()

    monitor = AudioDeviceMonitor(SlowDevices(), media_devices=FakeMediaDevices(), poll_ms=10000)
    ticks = []
    heartbeat = QTimer()
    heartbeat.setInterval(5)
    heartbeat.timeout.connect(lambda: ticks.append(True))
    heartbeat.start()
    monitor.start()
    qtbot.waitUntil(entered.is_set)
    monitor.stop()
    qtbot.wait(50)
    assert len(ticks) > 2
    with qtbot.waitSignal(monitor.finished):
        release.set()
    heartbeat.stop()
