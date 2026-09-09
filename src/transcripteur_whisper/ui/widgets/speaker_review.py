"""Explicit speaker correction, local audition, merge/undo and consent."""
from __future__ import annotations

from pathlib import Path
from typing import Any

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from ...services.speaker_review_state import SpeakerReviewState


def _clock(seconds: float) -> str:
    total = max(0, int(seconds))
    return f"{total // 3600:02d}:{total // 60 % 60:02d}:{total % 60:02d}"


class SpeakerReviewDialog(QDialog):
    def __init__(self, service: Any, review_id: str, self_name: str, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.service, self.review_id, self.self_name = service, review_id, self_name
        self.payload = service.load_review(review_id)
        self.state = SpeakerReviewState(self.payload)
        self.original = service.transcript(review_id)
        self.source_override: Path | None = None
        self._applying = False
        self._player = None
        self._playback_problem = ""
        self.setWindowTitle("Identifier les interlocuteurs")
        self.resize(1160, 780)
        intro = QLabel(
            "\u00c9coutez plusieurs extraits, cochez les groupes d'une m\u00eame personne puis fusionnez-les. "
            "L'aper\u00e7u change imm\u00e9diatement ; seul Appliquer modifie les fichiers. Annuler ne conserve rien."
        )
        intro.setWordWrap(True)
        privacy = QLabel(
            "M\u00e9moriser reste d\u00e9coch\u00e9 par d\u00e9faut. Apr\u00e8s confirmation, les variantes vocales disponibles "
            "alimentent un profil local chiffr\u00e9, jamais une moyenne unique. Les empreintes temporaires "
            "sont effac\u00e9es apr\u00e8s validation, ou au prochain acc\u00e8s pass\u00e9 7 jours. "
            "Une voix au microphone n'est vous que si personne d'autre n'utilise ce microphone."
        )
        privacy.setWordWrap(True)
        self.table = QTableWidget(0, 6)
        self.table.setHorizontalHeaderLabels([
            "S\u00e9lection / groupe", "Dur\u00e9e", "Nom affich\u00e9", "Extraits audio", "Voix disponibles", "M\u00e9moriser",
        ])
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.verticalHeader().setVisible(False)
        self._name_edits: dict[str, QLineEdit] = {}
        self._remember: dict[str, QCheckBox] = {}
        self._selected: dict[str, QCheckBox] = {}
        self._excerpts: dict[str, list] = {}
        self._excerpt_choices: dict[str, QComboBox] = {}
        self.merge_button = QPushButton("Fusionner la s\u00e9lection")
        self.split_button = QPushButton("D\u00e9fusionner la s\u00e9lection")
        self.undo_button = QPushButton("Annuler la derni\u00e8re fusion / d\u00e9fusion")
        self.stop_button = QPushButton("Arr\u00eater l'\u00e9coute")
        self.source_button = QPushButton("Associer l'audio d'origine\u2026")
        self.merge_button.clicked.connect(self._merge)
        self.split_button.clicked.connect(self._split)
        self.undo_button.clicked.connect(self._undo)
        self.stop_button.clicked.connect(self._stop)
        self.source_button.clicked.connect(self._choose_source)
        actions = QHBoxLayout()
        for button in (self.merge_button, self.split_button, self.undo_button):
            actions.addWidget(button)
        audio_actions = QHBoxLayout()
        audio_actions.addWidget(self.source_button)
        audio_actions.addWidget(self.stop_button)
        audio_actions.addStretch()
        self.audio_status = QLabel()
        self.audio_status.setWordWrap(True)
        self.preview = QPlainTextEdit()
        self.preview.setReadOnly(True)
        self.preview.setPlaceholderText("Aper\u00e7u de la transcription corrig\u00e9e")
        self.buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Save | QDialogButtonBox.StandardButton.Cancel)
        self.buttons.button(QDialogButtonBox.StandardButton.Save).setText("Appliquer")
        self.buttons.accepted.connect(self._apply)
        self.buttons.rejected.connect(self.reject)
        layout = QVBoxLayout(self)
        layout.addWidget(intro)
        layout.addWidget(privacy)
        layout.addWidget(self.table, 3)
        layout.addLayout(actions)
        layout.addLayout(audio_actions)
        layout.addWidget(self.audio_status)
        if self.payload.get("_has_generated_document"):
            warning = QLabel(
                "Attention : les documents d\u00e9j\u00e0 g\u00e9n\u00e9r\u00e9s (CR, r\u00e9sum\u00e9\u2026) ne sont pas r\u00e9\u00e9crits "
                "par cette correction. V\u00e9rifiez leurs noms et actions avant diffusion."
            )
            warning.setWordWrap(True)
            layout.addWidget(warning)
        layout.addWidget(QLabel("Aper\u00e7u (non enregistr\u00e9 avant Appliquer)"))
        layout.addWidget(self.preview, 2)
        layout.addWidget(self.buttons)
        try:
            from .speaker_player import SpeakerExcerptPlayer
            self._player = SpeakerExcerptPlayer(self)
            self._player.status.connect(self.audio_status.setText)
            self._player.failed.connect(self.audio_status.setText)
        except (ImportError, RuntimeError, OSError) as exc:
            self._playback_problem = f"Lecture Qt indisponible : {exc}. La correction des noms reste utilisable."
        self._render()

    def _render(self) -> None:
        self._stop()
        self.table.setRowCount(0)
        self._name_edits.clear()
        self._remember.clear()
        self._selected.clear()
        self._excerpts.clear()
        self._excerpt_choices.clear()
        groups = self.state.ordered_groups()
        self.table.setRowCount(len(groups))
        for row, group in enumerate(groups):
            key = group.group_id
            selected = QCheckBox(key if len(group.members) == 1 else f"{len(group.members)} groupes fusionn\u00e9s")
            selected.setToolTip(", ".join(group.members))
            selected.toggled.connect(self._update_actions)
            self._selected[key] = selected
            self.table.setCellWidget(row, 0, selected)
            self.table.setItem(row, 1, QTableWidgetItem(_clock(self.state.duration(key))))
            edit = QLineEdit(group.name)
            edit.setMaxLength(80)
            details = []
            for member in group.members:
                item = self.state.clusters[member]
                if item.get("source") == "self":
                    details.append(f"{member}: piste microphone personnelle (hypothese a verifier).")
                else:
                    status = "accepte" if item.get("auto_accepted") else "non attribue"
                    details.append(f"{member}: {status}; {item.get('auto_reason') or 'aucun profil confirme'}")
            edit.setToolTip("\n".join(details))
            edit.textChanged.connect(lambda text, group_id=key: self._rename(group_id, text))
            self._name_edits[key] = edit
            self.table.setCellWidget(row, 2, edit)
            excerpts = self.service.excerpts(self.payload, group.members, source_override=self.source_override)
            self._excerpts[key] = excerpts
            choices = QComboBox()
            for excerpt in excerpts:
                mode = "piste isol\u00e9e" if excerpt.isolated else "audio mix\u00e9"
                warning = " / chevauchement" if excerpt.overlap_possible else ""
                choices.addItem(f"{excerpt.cluster_id} \u00b7 {_clock(excerpt.global_start)} \u00b7 {mode}{warning}")
            choices.setMinimumWidth(255)
            choices.setEnabled(bool(excerpts))
            self._excerpt_choices[key] = choices
            listen = QPushButton("\u00c9couter")
            listen.setEnabled(bool(excerpts) and self._player is not None)
            listen.clicked.connect(lambda _checked=False, group_id=key: self._listen(group_id))
            cell = QWidget()
            line = QHBoxLayout(cell)
            line.setContentsMargins(0, 0, 0, 0)
            line.addWidget(choices, 1)
            line.addWidget(listen)
            self.table.setCellWidget(row, 3, cell)
            count = len(self.state.embeddings(key))
            available_members = sum(bool(self.state.clusters[member].get("embeddings")) for member in group.members)
            detail = QTableWidgetItem(f"{count} empreinte(s) / {available_members} sur {len(group.members)} variante(s)")
            detail.setToolTip("Seuls les extraits propres disponibles peuvent enrichir le profil. "
                              "Un score de similarit\u00e9 n'est pas un pourcentage de certitude.")
            self.table.setItem(row, 4, detail)
            remember = QCheckBox("M\u00e9moriser")
            remember.setEnabled(count > 0)
            remember.setChecked(group.remember and count > 0)
            remember.toggled.connect(lambda checked, group_id=key: self._remember_changed(group_id, checked))
            remember.setToolTip("Consentement explicite requis. Aucun apprentissage automatique silencieux."
                                if count else "Empreintes absentes ou effac\u00e9es. Relancez une transcription pour en obtenir.")
            self._remember[key] = remember
            self.table.setCellWidget(row, 5, remember)
            self.table.setRowHeight(row, 44)
        self.table.resizeColumnsToContents()
        self.table.horizontalHeader().setStretchLastSection(True)
        if self._playback_problem:
            self.audio_status.setText(self._playback_problem)
        elif not any(self._excerpts.values()):
            self.audio_status.setText("Aucun audio associ\u00e9. Choisissez l'audio original, sans coupe ni modification.")
        else:
            self.audio_status.setText("Choisissez un extrait puis \u00c9couter. Un seul extrait est lu \u00e0 la fois.")
        self._refresh_preview()
        self._update_actions()

    def _rename(self, key: str, text: str) -> None:
        self.state.groups[key].name = text.strip()
        self._refresh_preview()

    def _remember_changed(self, key: str, checked: bool) -> None:
        self.state.groups[key].remember = checked

    def _refresh_preview(self) -> None:
        self.preview.setPlainText(self.state.preview(self.original).speaker_text())

    def _selection(self) -> list[str]:
        return [key for key, box in self._selected.items() if box.isChecked()]

    def _update_actions(self, *_args) -> None:
        keys = self._selection()
        self.merge_button.setEnabled(len(keys) >= 2)
        self.split_button.setEnabled(len(keys) == 1 and len(self.state.groups[keys[0]].members) > 1)
        self.undo_button.setEnabled(self.state.can_undo)
        self.stop_button.setEnabled(self._player is not None)

    def _merge(self) -> None:
        keys = self._selection()
        if len(keys) < 2:
            return
        name, accepted = QInputDialog.getText(
            self, "Fusionner les interlocuteurs", "Nom du groupe fusionn\u00e9 :",
            QLineEdit.EchoMode.Normal, self.state.groups[keys[0]].name,
        )
        if not accepted:
            return
        try:
            self.state.merge(keys, name)
            self._render()
        except ValueError as exc:
            QMessageBox.warning(self, "Fusion impossible", str(exc))

    def _split(self) -> None:
        keys = self._selection()
        if len(keys) == 1:
            self.state.split(keys[0])
            self._render()

    def _undo(self) -> None:
        self.state.undo()
        self._render()

    def _listen(self, key: str) -> None:
        index = self._excerpt_choices[key].currentIndex()
        if self._player is not None and 0 <= index < len(self._excerpts[key]):
            self._player.play_excerpt(self._excerpts[key][index])

    def _stop(self) -> None:
        if self._player is not None:
            self._player.stop()

    def _choose_source(self) -> None:
        name, _filter = QFileDialog.getOpenFileName(
            self, "Choisir exactement l'audio original (m\u00eames horodatages)", "",
            "M\u00e9dias (*.wav *.mp3 *.m4a *.flac *.ogg *.aac *.mp4 *.mkv *.webm *.mov);;Tous (*)",
        )
        if name:
            self.source_override = Path(name).resolve()
            self._render()

    def _apply(self) -> None:
        if self._applying:
            return
        names = {key: edit.text() for key, edit in self._name_edits.items()}
        remember = {key for key, checkbox in self._remember.items() if checkbox.isChecked()}
        try:
            self.state.set_choices(names, remember)
        except ValueError as exc:
            QMessageBox.warning(self, "V\u00e9rifiez les noms", str(exc))
            return
        if remember and QMessageBox.question(
            self, "Confirmer la m\u00e9morisation vocale",
            "Enregistrer les variantes vocales disponibles des personnes coch\u00e9es dans des profils locaux chiffr\u00e9s ? "
            "Un profil portant d\u00e9j\u00e0 ce nom sera enrichi. Confirmez qu'il s'agit des m\u00eames personnes "
            "et que vous \u00eates autoris\u00e9 \u00e0 m\u00e9moriser ces voix.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        ) != QMessageBox.StandardButton.Yes:
            return
        self._applying = True
        self.buttons.setEnabled(False)
        self._stop()
        try:
            self.service.apply(
                self.review_id, names, remember, self_speaker_name=self.self_name,
                groups=self.state.serialized_groups(), expected_revision=self.payload.get("_revision"),
                expected_transcript_revision=self.payload.get("_transcript_revision"),
                source_override=self.source_override,
            )
        except Exception as exc:
            self._applying = False
            self.buttons.setEnabled(True)
            QMessageBox.critical(self, "Modification impossible", str(exc))
            return
        self.accept()

    def done(self, result: int) -> None:
        if self._player is not None:
            self._player.close()
        # Avoid retaining review embeddings in an invisible dialog after closing.
        for cluster in self.payload.get("clusters") or []:
            cluster["embeddings"] = []
        for cluster in self.state.clusters.values():
            cluster["embeddings"] = []
        self.state._undo.clear()
        super().done(result)


class SpeakerProfilesDialog(QDialog):
    def __init__(self, service: Any, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.service = service
        self.setWindowTitle("Voix connues")
        self.resize(560, 390)
        info = QLabel(
            "Ces profils contiennent des empreintes vocales locales chiffrées. "
            "Vous pouvez les renommer ou les supprimer à tout moment."
        )
        info.setWordWrap(True)
        self.list = QListWidget()
        self.rename_edit = QLineEdit()
        self.rename_edit.setMaxLength(80)
        self.rename_button = QPushButton("Renommer")
        self.delete_button = QPushButton("Supprimer")
        self.delete_all_button = QPushButton("Supprimer toutes les empreintes")
        self.close_button = QPushButton("Fermer")
        self.rename_button.clicked.connect(self._rename)
        self.delete_button.clicked.connect(self._delete)
        self.delete_all_button.clicked.connect(self._delete_all)
        self.close_button.clicked.connect(self.accept)
        self.list.currentItemChanged.connect(self._selection)
        row = QHBoxLayout()
        row.addWidget(self.rename_edit, 1)
        row.addWidget(self.rename_button)
        row.addWidget(self.delete_button)
        bottom = QHBoxLayout()
        bottom.addWidget(self.delete_all_button)
        bottom.addStretch()
        bottom.addWidget(self.close_button)
        layout = QVBoxLayout(self)
        layout.addWidget(info)
        layout.addWidget(self.list, 1)
        layout.addLayout(row)
        layout.addLayout(bottom)
        self._reload()

    def _reload(self) -> None:
        self.list.clear()
        for profile in self.service.list_profiles():
            suffix = " · profil personnel" if profile.is_self else ""
            item = QListWidgetItem(f"{profile.name} · {len(profile.embeddings)} empreinte(s){suffix}")
            item.setData(Qt.ItemDataRole.UserRole, profile.profile_id)
            item.setData(Qt.ItemDataRole.UserRole + 1, profile.name)
            self.list.addItem(item)
        self._selection()

    def _selection(self, *_: Any) -> None:
        item = self.list.currentItem()
        enabled = item is not None
        self.rename_button.setEnabled(enabled)
        self.delete_button.setEnabled(enabled)
        self.rename_edit.setEnabled(enabled)
        self.rename_edit.setText(str(item.data(Qt.ItemDataRole.UserRole + 1)) if item else "")

    def _rename(self) -> None:
        item = self.list.currentItem()
        if item is None:
            return
        try:
            self.service.rename(str(item.data(Qt.ItemDataRole.UserRole)), self.rename_edit.text())
            self._reload()
        except Exception as exc:
            QMessageBox.critical(self, "Renommage impossible", str(exc))

    def _delete(self) -> None:
        item = self.list.currentItem()
        if item is None:
            return
        if QMessageBox.question(
            self,
            "Supprimer l'empreinte",
            "Supprimer définitivement ce profil vocal local ?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        ) != QMessageBox.StandardButton.Yes:
            return
        try:
            self.service.delete(str(item.data(Qt.ItemDataRole.UserRole)))
            self._reload()
        except Exception as exc:
            QMessageBox.critical(self, "Suppression impossible", str(exc))

    def _delete_all(self) -> None:
        if QMessageBox.question(
            self,
            "Supprimer toutes les empreintes",
            "Supprimer définitivement toutes les empreintes vocales enregistrées sur ce compte Windows ?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        ) != QMessageBox.StandardButton.Yes:
            return
        try:
            self.service.delete_all()
            self._reload()
        except Exception as exc:
            QMessageBox.critical(self, "Suppression impossible", str(exc))
