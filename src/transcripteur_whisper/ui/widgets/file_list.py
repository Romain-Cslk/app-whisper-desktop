"""Multi-file selection with asynchronous media inspection."""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QDragEnterEvent, QDropEvent, QKeyEvent
from PySide6.QtWidgets import (
    QAbstractItemView,
    QFileDialog,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from ..workers import TaskRunner

EXTENSIONS = {'.aac', '.flac', '.m4a', '.mp3', '.mp4', '.mpga', '.ogg', '.wav', '.webm', '.mkv', '.mov'}
MEDIA_FILTER = 'Audio / vidéo (' + ' '.join('*' + ext for ext in sorted(EXTENSIONS)) + ')'
STAGES = {
    'queued': 'En attente', 'running': 'En cours', 'conversion': 'Conversion audio',
    'transcription_api': 'Transcription OpenAI', 'transcription_local': 'Transcription locale',
    'recomposition': 'Assemblage du texte', 'document': 'Création du document',
    'done': 'Terminé', 'partial': 'Résultat partiel', 'cancelled': 'Annulé', 'error': 'Échec',
}


def duration_label(seconds: float | None) -> str:
    if seconds is None:
        return 'Durée inconnue'
    hours, rest = divmod(int(seconds), 3600)
    minutes, secs = divmod(rest, 60)
    return f'{hours:02d}:{minutes:02d}:{secs:02d}'


class DropTable(QTableWidget):
    paths_dropped = Signal(list)
    remove_requested = Signal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(0, 4, parent)
        self.setAcceptDrops(True)
        self.setDragDropMode(QAbstractItemView.DragDropMode.DropOnly)
        self.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.setHorizontalHeaderLabels(['Fichier', 'Taille', 'Durée', 'État'])
        self.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        self.horizontalHeader().setSectionResizeMode(3, QHeaderView.ResizeMode.Stretch)
        self.verticalHeader().hide()
        self.setMinimumHeight(165)
        self.setAccessibleName('Fichiers audio ou vidéo sélectionnés')

    def dragEnterEvent(self, event: QDragEnterEvent) -> None:
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dragMoveEvent(self, event: QDragEnterEvent) -> None:
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event: QDropEvent) -> None:
        paths = [Path(url.toLocalFile()) for url in event.mimeData().urls() if url.isLocalFile()]
        self.paths_dropped.emit(paths)
        event.acceptProposedAction()

    def keyPressEvent(self, event: QKeyEvent) -> None:
        if event.key() == Qt.Key.Key_Delete:
            self.remove_requested.emit()
        else:
            super().keyPressEvent(event)


class FileList(QWidget):
    changed = Signal()
    notice = Signal(str)

    def __init__(self, runner: TaskRunner, media: object, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.runner = runner
        self.media = media
        self.table = DropTable()
        self.table.paths_dropped.connect(self.add_paths)
        self.table.remove_requested.connect(self.remove_selected)
        self.add_button = QPushButton('Ajouter des fichiers…')
        self.remove_button = QPushButton('Retirer la sélection')
        self.add_button.clicked.connect(self.choose_files)
        self.remove_button.clicked.connect(self.remove_selected)
        self.summary = QLabel('Glissez vos fichiers audio ou vidéo dans cette liste.')
        self.summary.setWordWrap(True)
        row = QHBoxLayout()
        row.addWidget(self.add_button)
        row.addWidget(self.remove_button)
        row.addStretch()
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addLayout(row)
        layout.addWidget(self.table)
        layout.addWidget(self.summary)

    @property
    def paths(self) -> list[Path]:
        return [self.table.item(row, 0).data(Qt.ItemDataRole.UserRole) for row in range(self.table.rowCount())]

    def choose_files(self) -> None:
        selected, _ = QFileDialog.getOpenFileNames(self, 'Ajouter des fichiers audio / vidéo', '', MEDIA_FILTER)
        self.add_paths([Path(path) for path in selected])

    def add_paths(self, paths: list[Path]) -> None:
        existing = set(self.paths)
        for path in paths:
            path = Path(path).absolute()
            if path in existing:
                continue
            if path.suffix.lower() not in EXTENSIONS:
                self.notice.emit(f'Format non pris en charge : {path.name}')
                continue
            existing.add(path)
            row = self.table.rowCount()
            self.table.insertRow(row)
            item = QTableWidgetItem(path.name)
            item.setData(Qt.ItemDataRole.UserRole, path)
            item.setToolTip(str(path))
            self.table.setItem(row, 0, item)
            for column in (1, 2, 3):
                self.table.setItem(row, column, QTableWidgetItem('Analyse…' if column == 3 else '—'))
            self.runner.submit(
                lambda p=path: (p.stat().st_size, self.media.probe(p)),
                lambda result, p=path: self._metadata(p, result),
                lambda error, p=path: self._metadata_error(p, error),
            )
        self._summary()
        self.changed.emit()

    def _find(self, path: Path) -> int | None:
        for row, candidate in enumerate(self.paths):
            if candidate == path:
                return row
        return None

    def _metadata(self, path: Path, result: tuple[int, float | None]) -> None:
        row = self._find(path)
        if row is not None:
            size, duration = result
            self.table.item(row, 1).setText(f'{size / 1024 / 1024:.1f} Mo')
            self.table.item(row, 2).setText(duration_label(duration))
            if self.table.item(row, 3).text() == 'Analyse…':
                self.table.item(row, 3).setText('Prêt')

    def _metadata_error(self, path: Path, error: str) -> None:
        row = self._find(path)
        if row is not None:
            self.table.item(row, 3).setText('Fichier inaccessible')
            self.table.item(row, 3).setToolTip(error)
        self.notice.emit(f'{path.name} : {error}')

    def remove_selected(self) -> None:
        if not self.isEnabled():
            return
        for row in sorted({item.row() for item in self.table.selectedItems()}, reverse=True):
            self.table.removeRow(row)
        self._summary()
        self.changed.emit()

    def _summary(self) -> None:
        self.summary.setText(f'{self.table.rowCount()} fichier(s) · Audio et vidéo · Déposez des fichiers pour les ajouter.')

    def update_progress(self, files: list[dict]) -> None:
        for row, item in enumerate(files):
            if row >= self.table.rowCount():
                break
            text = item.get('error') or item.get('stage') or item.get('status', '')
            text = STAGES.get(text, text)
            if item.get('segment_count'):
                text += f' · segment {item.get("segment_index", 0)}/{item["segment_count"]}'
            progress = int(float(item.get('progress') or 0) * 100)
            self.table.item(row, 3).setText(f'{text} · {progress} %')
            self.table.item(row, 3).setToolTip(text)
