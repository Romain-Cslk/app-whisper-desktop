"""Persistent results, export actions and optional speaker review."""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

from PySide6.QtCore import QUrl, Signal
from PySide6.QtGui import QDesktopServices
from PySide6.QtWidgets import (
    QApplication,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from ..workers import TaskRunner


class ResultsPanel(QWidget):
    notice = Signal(str)
    speaker_review_requested = Signal(str)

    def __init__(self, runner: TaskRunner, jobs: Any, results_dir: Path, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.runner = runner
        self.jobs = jobs
        self.results_dir = results_dir
        self.job_id: str | None = None
        self.entries: list[Path] = []
        self._review_ids: list[str | None] = []
        self.files = QListWidget()
        self.files.setAccessibleName("Résultats enregistrés")
        self.files.currentRowChanged.connect(self._selection_changed)
        self.files.itemDoubleClicked.connect(self.open_selected)
        self.open_button = QPushButton("Ouvrir")
        self.folder_button = QPushButton("Ouvrir le dossier")
        self.save_button = QPushButton("Enregistrer sous…")
        self.copy_button = QPushButton("Copier le texte")
        self.speaker_button = QPushButton("Identifier les interlocuteurs…")
        self.zip_button = QPushButton("Exporter le lot en ZIP…")
        self.txt_button = QPushButton("Exporter les transcriptions TXT…")
        self.open_button.clicked.connect(self.open_selected)
        self.folder_button.clicked.connect(self.open_folder)
        self.save_button.clicked.connect(self.save_selected)
        self.copy_button.clicked.connect(self.copy_selected)
        self.speaker_button.clicked.connect(self.review_speakers)
        self.zip_button.clicked.connect(lambda: self.export_batch("zip"))
        self.txt_button.clicked.connect(lambda: self.export_batch("txt"))
        row = QHBoxLayout()
        for button in (self.open_button, self.folder_button, self.save_button, self.copy_button):
            row.addWidget(button)
        speakers = QHBoxLayout()
        speakers.addWidget(self.speaker_button)
        speakers.addStretch()
        batch = QHBoxLayout()
        batch.addWidget(self.txt_button)
        batch.addWidget(self.zip_button)
        batch.addStretch()
        layout = QVBoxLayout(self)
        layout.addWidget(
            QLabel(
                "Les résultats sont conservés dans votre dossier utilisateur, y compris les résultats partiels."
            )
        )
        layout.addWidget(self.files)
        layout.addLayout(row)
        layout.addLayout(speakers)
        layout.addLayout(batch)
        self._selection_changed()

    @property
    def selected_path(self) -> Path | None:
        row = self.files.currentRow()
        return self.entries[row] if 0 <= row < len(self.entries) else None

    @property
    def selected_review_id(self) -> str | None:
        row = self.files.currentRow()
        return self._review_ids[row] if 0 <= row < len(self._review_ids) else None

    def load(self, job_id: str | None, snapshot: dict) -> None:
        self.job_id = job_id
        paths: list[Path] = []
        reviews: list[str | None] = []
        for result in snapshot.get("files", []):
            review_id = result.get("speaker_review_id") if result.get("speaker_review_available") else None
            for key in ("transcription_path", "out_path"):
                raw = result.get(key)
                if not raw:
                    continue
                path = Path(raw)
                # A speaker review rewrites the transcript, not a separately
                # generated summary/report. Attach the action only to the
                # transcription entry (or its duplicate out_path).
                entry_review = str(review_id) if review_id and key == "transcription_path" else None
                if path in paths:
                    position = paths.index(path)
                    if reviews[position] is None and entry_review:
                        reviews[position] = entry_review
                    continue
                paths.append(path)
                reviews.append(entry_review)
        if paths == self.entries and reviews == self._review_ids:
            self._selection_changed()
            return
        selected = self.selected_path
        self.entries = paths
        self._review_ids = reviews
        self.files.clear()
        for path, review_id in zip(paths, reviews):
            suffix = " · locuteurs à vérifier" if review_id else ""
            item = QListWidgetItem(path.name + suffix)
            item.setToolTip(str(path))
            self.files.addItem(item)
        if paths:
            self.files.setCurrentRow(paths.index(selected) if selected in paths else 0)
        self._selection_changed()

    def _selection_changed(self, *_: Any) -> None:
        available = self.selected_path is not None
        for button in (self.open_button, self.save_button, self.copy_button):
            button.setEnabled(available)
        self.speaker_button.setEnabled(bool(self.selected_review_id))
        self.zip_button.setEnabled(bool(self.entries and self.job_id))
        self.txt_button.setEnabled(bool(self.entries and self.job_id))

    def review_speakers(self) -> None:
        review_id = self.selected_review_id
        if review_id:
            self.speaker_review_requested.emit(review_id)

    def open_selected(self, *_: Any) -> None:
        if self.selected_path:
            if not QDesktopServices.openUrl(QUrl.fromLocalFile(str(self.selected_path))):
                self.notice.emit("Windows ne peut pas ouvrir ce fichier. Utilisez « Enregistrer sous… ».")

    def open_folder(self) -> None:
        folder = self.selected_path.parent if self.selected_path else self.results_dir
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(folder)))

    def save_selected(self) -> None:
        path = self.selected_path
        if path is None:
            return
        destination, _ = QFileDialog.getSaveFileName(self, "Enregistrer le résultat", path.name, "Texte (*.txt)")
        if destination:
            self.runner.submit(
                lambda: shutil.copyfile(path, destination),
                lambda _: self.notice.emit(f"Résultat enregistré : {destination}"),
                self.notice.emit,
            )

    def copy_selected(self) -> None:
        path = self.selected_path
        if path:
            self.runner.submit(lambda: path.read_text(encoding="utf-8"), self._copy_text, self.notice.emit)

    def _copy_text(self, text: str) -> None:
        QApplication.clipboard().setText(text)
        self.notice.emit("Texte copié dans le presse-papiers.")

    def export_batch(self, extension: str) -> None:
        job_id = self.job_id
        if job_id is None:
            return
        destination, _ = QFileDialog.getSaveFileName(
            self,
            "Exporter le lot",
            f"transcriptions.{extension}",
            "Archive ZIP (*.zip)" if extension == "zip" else "Texte (*.txt)",
        )
        if destination:
            path = Path(destination)
            if not path.suffix:
                path = path.with_suffix("." + extension)
            kind = "result" if extension == "zip" else "transcription"
            self.runner.submit(
                lambda: self.jobs.export(job_id, path, kind=kind),
                lambda result: self.notice.emit(f"Export terminé : {result}"),
                self.notice.emit,
            )
