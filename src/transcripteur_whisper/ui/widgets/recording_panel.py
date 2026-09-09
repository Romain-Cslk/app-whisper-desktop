"""Recording controls; audio backend operations never execute in Qt slots."""

from __future__ import annotations

import shutil
import uuid
from pathlib import Path
from typing import Any

from PySide6.QtCore import QStandardPaths, QTimer, Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFileDialog,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QProgressBar,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from ..workers import TaskRunner
from .file_list import duration_label


class RecordingPanel(QGroupBox):
    recording_saved = Signal(object)
    notice = Signal(str)
    state_changed = Signal()
    shutdown_failed = Signal(str)

    def __init__(self, runner: TaskRunner, service: Any, settings: Any = None, parent: QWidget | None = None) -> None:
        super().__init__("Enregistrer le microphone et le son du PC", parent)
        self.runner = runner
        self.service = service
        self.settings = settings
        self._transition = False
        self._reading_levels = False
        self._closing = False
        self._snapshot = None
        self._stopping = False
        self._last_recording_path: Path | None = None
        self.microphone = QComboBox()
        self.microphone.setAccessibleName("Microphone pour le prochain enregistrement")
        self.speaker = QComboBox()
        self.speaker.setAccessibleName("Sortie système pour le prochain enregistrement")
        self.microphone_enabled = QCheckBox("Microphone")
        self.microphone_enabled.setChecked(True)
        self.speaker_enabled = QCheckBox("Son du PC")
        self.speaker_enabled.setChecked(True)
        self.mic_meter = self._make_meter("Niveau du microphone")
        self.system_meter = self._make_meter("Niveau du son du PC")
        self.record_button = QPushButton("Enregistrer")
        self.record_button.setObjectName("recordButton")
        self.record_button.clicked.connect(self.toggle_recording)
        self.export_button = QPushButton("Enregistrer le WAV sous…")
        self.export_button.setObjectName("exportRecordingButton")
        self.export_button.setToolTip("Copier le dernier enregistrement WAV vers le dossier de votre choix")
        self.export_button.clicked.connect(self.export_last_recording)
        self.export_button.hide()
        self.timer_label = QLabel("00:00:00")
        self.timer_label.setObjectName("recordingTimer")
        self.status = QLabel("Détection automatique des périphériques audio…")
        self.status.setWordWrap(True)
        self.recovery_choices = QComboBox()
        self.recover_button = QPushButton("Récupérer le WAV")
        self.recover_button.clicked.connect(self.recover_selected)
        self.recovery_choices.hide()
        self.recover_button.hide()
        grid = QGridLayout()
        grid.addWidget(self.microphone_enabled, 0, 0)
        grid.addWidget(self.speaker_enabled, 0, 1)
        grid.addWidget(self.microphone, 1, 0)
        grid.addWidget(self.speaker, 1, 1)
        grid.addWidget(self.mic_meter, 2, 0)
        grid.addWidget(self.system_meter, 2, 1)
        row = QHBoxLayout()
        row.addWidget(self.record_button)
        row.addWidget(self.export_button)
        row.addWidget(self.timer_label)
        row.addStretch()
        recovery = QHBoxLayout()
        recovery.addWidget(self.recovery_choices, 1)
        recovery.addWidget(self.recover_button)
        layout = QVBoxLayout(self)
        layout.addLayout(grid)
        layout.addLayout(row)
        layout.addWidget(self.status)
        layout.addLayout(recovery)
        self.level_timer = QTimer(self)
        self.level_timer.setInterval(100)
        self.level_timer.timeout.connect(self._poll_levels)
        self.microphone.currentIndexChanged.connect(self._save_selection)
        self.speaker.currentIndexChanged.connect(self._save_selection)

    @staticmethod
    def _make_meter(label: str) -> QProgressBar:
        meter = QProgressBar()
        meter.setRange(0, 600)
        meter.setValue(0)
        meter.setFormat("Silence")
        meter.setAccessibleName(label)
        meter.setMaximumHeight(20)
        return meter

    @property
    def busy(self) -> bool:
        return self._transition or self.service.is_active

    def _save_selection(self) -> None:
        if self.settings and self._snapshot is not None:
            try:
                self.settings.set("microphone_key", self.microphone.currentData())
                self.settings.set("speaker_key", self.speaker.currentData())
            except OSError:
                self.notice.emit("La sélection audio ne peut pas être mémorisée dans votre dossier utilisateur.")

    def update_devices(self, snapshot: Any) -> None:
        previous = self._snapshot
        self._snapshot = snapshot
        for combo, devices, setting, source in (
            (self.microphone, snapshot.inputs, "microphone_key", "microphone"),
            (self.speaker, snapshot.outputs, "speaker_key", "sortie système"),
        ):
            old_key = combo.currentData()
            if previous is None and self.settings:
                old_key = self.settings.get(setting)
            combo.blockSignals(True)
            combo.clear()
            for device in devices:
                suffix = " (par défaut Windows)" if device.is_default else ""
                combo.addItem(device.name + suffix, device.stable_key)
            index = combo.findData(old_key) if old_key is not None else -1
            missing = old_key is not None and index < 0
            if index < 0:
                index = next((i for i, device in enumerate(devices) if device.is_default), 0 if devices else -1)
            combo.setCurrentIndex(index)
            combo.blockSignals(False)
            if missing:
                suffix = " Le flux actif ne change pas." if self.service.is_active else ""
                self.notice.emit(
                    f"Le {source} sélectionné a disparu. Sélection adaptée pour le prochain enregistrement.{suffix}"
                )
        if previous is not None:
            previous_keys = {device.stable_key for device in (*previous.inputs, *previous.outputs)}
            added = [
                device.name
                for device in (*snapshot.inputs, *snapshot.outputs)
                if device.stable_key not in previous_keys
            ]
            if added:
                self.notice.emit("Nouveau périphérique audio détecté : " + ", ".join(added))
        if not self.service.is_active:
            self.status.setText("Périphériques actualisés automatiquement. Choisissez une ou deux sources.")
        self._save_selection()

    def scan_recovery(self) -> None:
        if hasattr(self.service, "recoverable_recordings"):
            self.runner.submit(self.service.recoverable_recordings, self._show_recovery, self.notice.emit)

    def _show_recovery(self, paths: tuple[Path, ...]) -> None:
        self.recovery_choices.clear()
        for path in paths:
            self.recovery_choices.addItem(Path(path).name, Path(path))
        self.recovery_choices.setVisible(bool(paths))
        self.recover_button.setVisible(bool(paths))
        if paths:
            self.notice.emit(f"{len(paths)} enregistrement(s) interrompu(s) peuvent être récupérés.")

    def recover_selected(self) -> None:
        path = self.recovery_choices.currentData()
        if path is None or self.busy:
            return
        self.recover_button.setEnabled(False)
        self.runner.submit(
            lambda: self.service.recover(path),
            self._recovered,
            self.notice.emit,
            lambda: self.recover_button.setEnabled(True),
        )

    def _recovered(self, path: Path) -> None:
        recovered = Path(path)
        self._set_exportable_recording(recovered)
        self.recording_saved.emit(recovered)
        self.notice.emit(f"Enregistrement récupéré : {recovered.name}")
        self.scan_recovery()

    def _set_exportable_recording(self, path: Path) -> None:
        self._last_recording_path = Path(path)
        available = self._last_recording_path.is_file()
        self.export_button.setVisible(available)
        self.export_button.setEnabled(available)

    def export_last_recording(self) -> None:
        source = self._last_recording_path
        if source is None or not source.is_file():
            self.export_button.hide()
            self.notice.emit("Le dernier WAV n'est plus disponible pour l'export.")
            return

        downloads = QStandardPaths.writableLocation(QStandardPaths.StandardLocation.DownloadLocation)
        initial_dir = Path(downloads) if downloads else Path.home()
        selected, _ = QFileDialog.getSaveFileName(
            self,
            "Enregistrer l'enregistrement WAV sous…",
            str(initial_dir / source.name),
            "Audio WAV (*.wav)",
        )
        if not selected:
            return
        destination = Path(selected)
        if destination.suffix.lower() != ".wav":
            destination = destination.with_suffix(".wav")
        try:
            if source.resolve() == destination.resolve():
                self.notice.emit("Le WAV est déjà conservé à cet emplacement.")
                return
        except OSError:
            pass

        self.export_button.setEnabled(False)
        self.status.setText(f"Export du WAV vers {destination.name}…")
        self.runner.submit(
            lambda: self._copy_recording(source, destination),
            self._exported,
            self._export_failed,
            self._export_finished,
        )

    @staticmethod
    def _copy_recording(source: Path, destination: Path) -> Path:
        source = Path(source)
        destination = Path(destination)
        if not source.is_file():
            raise FileNotFoundError("Le WAV source n'existe plus.")
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_name(f".{destination.name}.{uuid.uuid4().hex}.partial")
        try:
            shutil.copy2(source, temporary)
            temporary.replace(destination)
        finally:
            temporary.unlink(missing_ok=True)
        return destination

    def _exported(self, destination: Path) -> None:
        path = Path(destination)
        self.status.setText(f"WAV exporté : {path}")
        self.notice.emit(f"Enregistrement exporté : {path}")

    def _export_failed(self, message: str) -> None:
        self.status.setText(f"Échec de l'export du WAV : {message}")
        self.notice.emit(f"Échec de l'export du WAV : {message}")

    def _export_finished(self) -> None:
        self.export_button.setEnabled(bool(self._last_recording_path and self._last_recording_path.is_file()))

    def toggle_recording(self) -> None:
        if self._transition:
            return
        if self.service.is_active:
            self.stop_recording()
        else:
            self.start_recording()

    def start_recording(self) -> None:
        if self._closing:
            return
        microphone = self.microphone.currentData() if self.microphone_enabled.isChecked() else None
        speaker = self.speaker.currentData() if self.speaker_enabled.isChecked() else None
        microphone_enabled = self.microphone_enabled.isChecked()
        system_enabled = self.speaker_enabled.isChecked()
        if not microphone and not speaker:
            self.notice.emit("Choisissez au moins une source audio disponible.")
            return
        kwargs = {
            "microphone_key": microphone,
            "speaker_key": speaker,
            "microphone_enabled": microphone_enabled,
            "system_enabled": system_enabled,
        }
        # When diarization is enabled before recording, keep the two source
        # journals so the personal microphone can be labelled without guessing.
        if self.settings and bool(self.settings.get("diarization_enabled", False)):
            kwargs["preserve_sources"] = True
        self._set_transition(True)
        self.status.setText("Ouverture des sources audio…")
        self.runner.submit(
            lambda: self.service.start(**kwargs),
            self._started,
            self._failed,
            self._transition_finished,
        )

    def _started(self, result: Any) -> None:
        self.timer_label.setText("00:00:00")
        self.record_button.setText("Arrêter et conserver le WAV")
        detail = (
            " Les pistes micro/système seront conservées pour la diarisation."
            if self.settings and bool(self.settings.get("diarization_enabled", False))
            else ""
        )
        self.status.setText(
            "Enregistrement en cours. Les choix de périphériques concernent le prochain enregistrement." + detail
        )
        self.level_timer.start()

    def _transition_finished(self) -> None:
        self._set_transition(False)
        if self._closing and self.service.is_active:
            self.stop_recording()

    def _set_transition(self, busy: bool) -> None:
        self._transition = busy
        self.record_button.setEnabled(not busy and not self._closing)
        self.state_changed.emit()

    def stop_recording(self) -> None:
        if self._transition:
            return
        self.level_timer.stop()
        self._stopping = True
        self._set_transition(True)
        self.status.setText("Finalisation et conservation du WAV…")
        self.runner.submit(self.service.stop, self._stopped, self._failed, self._transition_finished)

    def _stopped(self, result: Any) -> None:
        self._stopping = False
        self.record_button.setText("Enregistrer")
        self.mic_meter.setValue(0)
        self.system_meter.setValue(0)
        path = Path(result.path)
        self._set_exportable_recording(path)
        self.status.setText(
            f"WAV conservé : {path.name}. Vous pouvez l'exporter ou poursuivre vers la transcription."
        )
        self.recording_saved.emit(path)

    def _failed(self, message: str) -> None:
        self.status.setText(message)
        self.notice.emit(message)
        if self._closing and self._stopping:
            self._closing = False
            self.shutdown_failed.emit(message)
        self._stopping = False
        if not self.service.is_active:
            self.level_timer.stop()
            self.record_button.setText("Enregistrer")

    def _poll_levels(self) -> None:
        if self._reading_levels or self._transition:
            return
        self._reading_levels = True
        self.runner.submit(self.service.levels, self._render_levels, self._failed, self._levels_finished)

    def _levels_finished(self) -> None:
        self._reading_levels = False

    def _render_levels(self, levels: dict | None) -> None:
        if not levels:
            return
        self.timer_label.setText(duration_label(levels.get("elapsed", 0)))
        for key, meter in (("microphone", self.mic_meter), ("system", self.system_meter)):
            source = levels.get("sources", {}).get(key, {})
            value = source.get("rms_db", -60)
            db = max(-60.0, min(0.0, float(value if value is not None else -60)))
            meter.setValue(round((db + 60) * 10))
            meter.setFormat("Indisponible" if source.get("failed") else f"{db:.1f} dB")
        if levels.get("degraded"):
            self.status.setText(
                "Enregistrement dégradé : une source est indisponible. Les données capturées sont conservées ; "
                "les autres sources continuent."
            )

    def shutdown(self) -> None:
        self._closing = True
        self.level_timer.stop()
        self.record_button.setEnabled(False)
        if self.service.is_active and not self._transition:
            self.stop_recording()

    def resume_after_shutdown_error(self) -> None:
        self._closing = False
        self.record_button.setEnabled(not self._transition)
