"""Native Qt tests use bounded fake backends, never audio hardware or network."""

from __future__ import annotations

import threading
from types import SimpleNamespace

import pytest
from PySide6.QtCore import QThread, QTimer
from PySide6.QtWidgets import QApplication, QMessageBox

from transcripteur_whisper.core.paths import AppPaths
from transcripteur_whisper.core.settings import SettingsStore
from transcripteur_whisper.models.audio_devices import AudioDevice, AudioDeviceSnapshot
from transcripteur_whisper.ui.main_window import MainWindow
from transcripteur_whisper.ui.workers import TaskRunner


class FakeMedia:
    def __init__(self):
        self.threads = []
        self.release = None

    def probe(self, path):
        self.threads.append(QThread.currentThread())
        if self.release:
            assert self.release.wait(5)
        return 65.0


class FakeModels:
    def __init__(self, available=True):
        self.cached = available
        self.threads = []

    def available(self, name):
        self.threads.append(QThread.currentThread())
        return self.cached


class FakeJobs:
    def __init__(self):
        self.busy = False
        self.submitted = []
        self.cancelled = []
        self.threads = []
        self.closed = False
        self.release = None
        self.started = threading.Event()
        self.state = {'id': 'job-one', 'status': 'done', 'progress': 1.0, 'files': [], 'logs': []}

    def submit(self, files, options, api_key=''):
        self.threads.append(QThread.currentThread())
        self.started.set()
        if self.release:
            assert self.release.wait(5)
        self.submitted.append((files, options, api_key))
        self.busy = self.state['status'] not in {'done', 'cancelled'}
        return 'job-one'

    def history(self):
        return [dict(self.state)] if self.submitted else []

    def snapshot(self, identifier):
        return dict(self.state)

    def cancel(self, identifier):
        self.threads.append(QThread.currentThread())
        self.cancelled.append(identifier)
        self.state['status'] = 'cancelled'
        self.busy = False

    def shutdown(self, wait=False):
        self.threads.append(QThread.currentThread())
        self.closed = True


class FakeRecording:
    def __init__(self, paths):
        self.paths = paths
        self.is_active = False
        self.started = []
        self.threads = []
        self.closed = False
        self.status = {'elapsed': 65, 'degraded': False, 'sources': {}}

    def recoverable_recordings(self):
        return ()

    def start(self, **kwargs):
        self.threads.append(QThread.currentThread())
        self.started.append(kwargs)
        self.is_active = True
        return SimpleNamespace(path=self.paths.results / 'captured.wav')

    def stop(self):
        self.threads.append(QThread.currentThread())
        self.is_active = False
        path = self.paths.results / 'captured.wav'
        path.write_bytes(b'captured audio')
        return SimpleNamespace(path=path, duration=65)

    def levels(self):
        return self.status

    def close(self):
        self.threads.append(QThread.currentThread())
        self.closed = True


@pytest.fixture
def desktop(qtbot, tmp_path):
    paths = AppPaths.create(tmp_path / 'user')
    jobs, media, models = FakeJobs(), FakeMedia(), FakeModels()
    recording = FakeRecording(paths)
    window = MainWindow(paths, jobs=jobs, media=media, models=models,
                        recording=recording, devices=object(), monitor=False)
    def finish_shutdown(widget):
        if media.release:
            media.release.set()
        if jobs.release:
            jobs.release.set()
        widget.close()
        qtbot.waitUntil(lambda: widget._close_ready, timeout=6000)

    qtbot.addWidget(window, before_close_func=finish_shutdown)
    window.show()
    qtbot.waitUntil(lambda: window.runner.active_count == 0)
    yield window


def audio_device(key, index, *, default=False, kind='input'):
    return AudioDevice(key, key, index, key, kind, 'WASAPI', 2, 48000, default)


def test_worker_delivers_result_on_gui_thread_while_gui_remains_responsive(qtbot):
    runner = TaskRunner()
    gate = threading.Event()
    worker_threads, callbacks, ticks = [], [], []

    def work():
        worker_threads.append(QThread.currentThread())
        assert gate.wait(5)
        return 'complete'

    timer = QTimer()
    timer.setInterval(10)
    timer.timeout.connect(lambda: ticks.append(True))
    timer.start()
    runner.submit(work, lambda value: callbacks.append((value, QThread.currentThread())))
    try:
        qtbot.waitUntil(lambda: len(ticks) >= 3)
        assert not callbacks
    finally:
        gate.set()
    qtbot.waitUntil(lambda: runner.active_count == 0)
    timer.stop()
    assert worker_threads[0] != QApplication.instance().thread()
    assert callbacks == [('complete', QApplication.instance().thread())]


def test_media_inspection_is_async_and_file_can_be_removed_before_completion(desktop, qtbot, tmp_path):
    path = tmp_path / 'meeting.wav'
    path.write_bytes(b'media')
    desktop.media.release = threading.Event()
    desktop.files.add_paths([path, path, tmp_path / 'unsupported.exe'])
    qtbot.waitUntil(lambda: len(desktop.media.threads) == 1)
    assert desktop.files.paths == [path]
    desktop.files.table.selectRow(0)
    desktop.files.remove_selected()
    assert desktop.files.paths == []
    desktop.media.release.set()
    qtbot.waitUntil(lambda: desktop.runner.active_count == 0)
    assert desktop.files.table.rowCount() == 0
    assert desktop.media.threads[0] != QApplication.instance().thread()


def test_local_launch_model_preflight_and_submission_run_off_gui(desktop, qtbot, tmp_path):
    path = tmp_path / 'meeting.wav'
    path.write_bytes(b'media')
    desktop.files.add_paths([path])
    desktop.start_transcription()
    qtbot.waitUntil(lambda: len(desktop.jobs.submitted) == 1 and not desktop._active)
    files, options, key = desktop.jobs.submitted[0]
    assert files == [path]
    assert options.mode == 'local'
    assert options.language == 'fr'
    assert key == ''
    assert all(thread != QApplication.instance().thread() for thread in desktop.models.threads + desktop.jobs.threads)
    assert desktop.progress.value() == 1000


def test_missing_model_requires_consent_and_decline_stops_launch(desktop, qtbot, tmp_path, monkeypatch):
    path = tmp_path / 'meeting.wav'
    path.write_bytes(b'media')
    desktop.models.cached = False
    questions = []

    def decline(*args):
        questions.append(args[2])
        return QMessageBox.StandardButton.No

    monkeypatch.setattr(QMessageBox, 'question', decline)
    desktop.files.add_paths([path])
    desktop.start_transcription()
    qtbot.waitUntil(lambda: not desktop._active)
    assert questions and 'Go' in questions[0]
    assert desktop.jobs.submitted == []


def test_api_key_is_password_memory_only_and_gui_preferences_survive(desktop, qtbot, tmp_path):
    path = tmp_path / 'meeting.wav'
    path.write_bytes(b'media')
    desktop.mode.setCurrentIndex(desktop.mode.findData('api'))
    desktop.api_key.setText('temporary-test-credential')
    desktop.files.add_paths([path])
    desktop.start_transcription()
    qtbot.waitUntil(lambda: len(desktop.jobs.submitted) == 1 and not desktop._active)
    assert desktop.jobs.submitted[0][2] == 'temporary-test-credential'
    assert desktop.api_key.echoMode() == desktop.api_key.EchoMode.Password
    for config_file in desktop.paths.config.glob('*'):
        assert 'temporary-test-credential' not in config_file.read_text(encoding='utf-8')
    assert SettingsStore(desktop.paths).get('mode') == 'api'


def test_close_during_pending_submission_cancels_then_joins_services(desktop, qtbot, tmp_path):
    path = tmp_path / 'meeting.wav'
    path.write_bytes(b'media')
    desktop.jobs.release = threading.Event()
    desktop.jobs.state['status'] = 'running'
    desktop.files.add_paths([path])
    desktop.start_transcription()
    qtbot.waitUntil(desktop.jobs.started.is_set)
    desktop.close()
    assert desktop.isVisible()
    assert not desktop._close_ready
    desktop.jobs.release.set()
    qtbot.waitUntil(lambda: desktop._close_ready, timeout=6000)
    assert desktop.jobs.cancelled == ['job-one']
    assert desktop.jobs.closed and desktop.recording.closed
    assert desktop.runner.active_count == 0
    assert all(thread != QApplication.instance().thread() for thread in desktop.jobs.threads)


def test_hotplug_keeps_identity_then_uses_new_default_without_switching_active_stream(desktop):
    panel = desktop.recording_panel
    selected = audio_device('USB microphone', 2, default=True)
    other = audio_device('Built in', 3)
    panel.update_devices(AudioDeviceSnapshot((selected, other), ()))
    assert panel.microphone.currentData() == selected.stable_key
    panel.update_devices(AudioDeviceSnapshot((audio_device('Built in', 1, default=True),
                                             audio_device('USB microphone', 5), audio_device('New USB', 6)), ()))
    assert panel.microphone.currentData() == selected.stable_key
    desktop.recording.is_active = True
    panel.update_devices(AudioDeviceSnapshot((audio_device('Built in', 0, default=True),), ()))
    assert panel.microphone.currentData() == 'Built in'
    assert desktop.recording.started == []
    assert 'flux actif ne change pas' in desktop.notice.text()
    desktop.recording.is_active = False


def test_recording_start_stop_flags_meters_and_saved_file(desktop, qtbot):
    panel = desktop.recording_panel
    panel.update_devices(AudioDeviceSnapshot((audio_device('USB mic', 2, default=True),), ()))
    panel.speaker_enabled.setChecked(False)
    panel.start_recording()
    qtbot.waitUntil(lambda: desktop.recording.is_active and not panel._transition)
    assert desktop.recording.started == [{'microphone_key': 'USB mic', 'speaker_key': None,
                                          'microphone_enabled': True, 'system_enabled': False}]
    desktop.recording.status = {'elapsed': 65, 'degraded': True,
                                'sources': {'microphone': {'rms_db': 0, 'failed': False},
                                            'system': {'rms_db': -60, 'failed': True}}}
    qtbot.waitUntil(lambda: panel.mic_meter.value() == 600)
    assert panel.timer_label.text() == '00:01:05'
    assert 'dégradé' in panel.status.text()
    panel.stop_recording()
    qtbot.waitUntil(lambda: not panel.busy and bool(desktop.files.paths))
    assert desktop.files.paths == [desktop.paths.results / 'captured.wav']
    assert all(thread != QApplication.instance().thread() for thread in desktop.recording.threads)


def test_close_preserves_active_recording_before_services_close(desktop, qtbot):
    desktop.recording.is_active = True
    desktop.close()
    qtbot.waitUntil(lambda: desktop._close_ready, timeout=6000)
    assert (desktop.paths.results / 'captured.wav').read_bytes() == b'captured audio'
    assert desktop.recording.closed


def test_result_copy_runs_file_read_in_worker_and_clipboard_on_gui(desktop, qtbot):
    result = desktop.paths.results / 'transcription.txt'
    result.write_text('Une transcription disponible.', encoding='utf-8')
    desktop.results.load('job-one', {'files': [{'transcription_path': str(result), 'out_path': str(result)}]})
    desktop.results.copy_selected()
    qtbot.waitUntil(lambda: QApplication.clipboard().text() == 'Une transcription disponible.')
    assert desktop.results.entries == [result]
    assert desktop.results.zip_button.isEnabled()


def test_native_window_starts_and_closes_without_binding_any_socket(qtbot, tmp_path, monkeypatch):
    import socket

    def forbidden_bind(*args, **kwargs):
        pytest.fail('A native desktop window must not bind a network listener.')

    monkeypatch.setattr(socket.socket, 'bind', forbidden_bind)
    paths = AppPaths.create(tmp_path / 'native')
    window = MainWindow(paths, monitor=False)

    def shutdown(widget):
        widget.close()
        qtbot.waitUntil(lambda: widget._close_ready, timeout=6000)

    qtbot.addWidget(window, before_close_func=shutdown)
    window.show()
    qtbot.waitUntil(lambda: not window._initializing and window.runner.active_count == 0)
    assert window.isVisible()
    shutdown(window)
