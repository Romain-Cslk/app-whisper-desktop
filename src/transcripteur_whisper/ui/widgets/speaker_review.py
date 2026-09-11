"""Speaker correction, per-sample verification and local voice-profile management."""
from __future__ import annotations

from dataclasses import dataclass
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

from ...services.speaker_evidence import EvidenceRecordRequest
from ...services.speaker_review_state import SpeakerReviewState
from ...services.speaker_sample_service import SpeakerSampleService


def _clock(seconds: float) -> str:
    total = max(0, int(seconds))
    return f"{total // 3600:02d}:{total // 60 % 60:02d}:{total % 60:02d}"


def _excerpt_key(excerpt: Any) -> str:
    return "|".join((
        str(Path(excerpt.path).resolve()).casefold(),
        f"{float(excerpt.start):.3f}",
        f"{float(excerpt.end):.3f}",
        str(excerpt.cluster_id),
    ))


@dataclass
class SampleChoice:
    excerpt: Any
    decision: str = ""
    actual_name: str = ""


class SpeakerSampleReviewDialog(QDialog):
    """Review each audible sample independently so one bad voice cannot contaminate a profile."""

    def __init__(
        self,
        speaker_name: str,
        excerpts: list[Any],
        profile_names: list[str],
        choices: dict[str, SampleChoice],
        player: Any | None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.speaker_name = speaker_name
        self.excerpts = excerpts
        self.profile_names = profile_names
        self.choices = choices
        self.player = player
        self.setWindowTitle(f"Vérifier les voix - {speaker_name}")
        self.resize(980, 520)

        intro = QLabel(
            "Classez chaque extrait séparément. Si 2 extraits sont corrects et le 3e appartient à quelqu'un d'autre, "
            "mettez les deux premiers en Bonne voix et le troisième en Mauvaise voix. Vous pouvez alors choisir la "
            "vraie personne, ou laisser le champ vide si vous ne la connaissez pas."
        )
        intro.setWordWrap(True)

        self.table = QTableWidget(len(excerpts), 5)
        self.table.setHorizontalHeaderLabels(["Extrait", "Écouter", "Décision", "Vraie personne si mauvaise", "Info"])
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.verticalHeader().setVisible(False)
        self._decision_boxes: list[QComboBox] = []
        self._actual_boxes: list[QComboBox] = []

        for row, excerpt in enumerate(excerpts):
            self.table.setItem(row, 0, QTableWidgetItem(f"{excerpt.cluster_id} · {_clock(excerpt.global_start)}"))
            listen = QPushButton("Écouter")
            listen.setEnabled(player is not None)
            listen.clicked.connect(lambda _checked=False, index=row: self._listen(index))
            self.table.setCellWidget(row, 1, listen)

            decision = QComboBox()
            decision.addItem("À vérifier", "")
            decision.addItem("Bonne voix", "GOOD")
            decision.addItem("Mauvaise voix", "BAD")
            decision.addItem("Incertain", "UNSURE")
            existing = choices.get(_excerpt_key(excerpt))
            if existing:
                index = decision.findData(existing.decision)
                if index >= 0:
                    decision.setCurrentIndex(index)
            self._decision_boxes.append(decision)
            self.table.setCellWidget(row, 2, decision)

            actual = QComboBox()
            actual.setEditable(True)
            actual.addItem("")
            actual.addItems(profile_names)
            if existing and existing.actual_name:
                actual.setCurrentText(existing.actual_name)
            actual.setEnabled(decision.currentData() == "BAD")
            decision.currentIndexChanged.connect(
                lambda _value, box=decision, target=actual: target.setEnabled(box.currentData() == "BAD")
            )
            self._actual_boxes.append(actual)
            self.table.setCellWidget(row, 3, actual)

            mode = "piste isolée" if excerpt.isolated else "audio mixé"
            if excerpt.overlap_possible:
                mode += " · chevauchement possible"
            self.table.setItem(row, 4, QTableWidgetItem(mode))
            self.table.setRowHeight(row, 42)

        self.table.resizeColumnsToContents()
        self.table.horizontalHeader().setStretchLastSection(True)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Save | QDialogButtonBox.StandardButton.Cancel)
        buttons.button(QDialogButtonBox.StandardButton.Save).setText("Enregistrer les choix")
        buttons.accepted.connect(self._save)
        buttons.rejected.connect(self.reject)

        layout = QVBoxLayout(self)
        layout.addWidget(intro)
        layout.addWidget(self.table, 1)
        layout.addWidget(buttons)

    def _listen(self, index: int) -> None:
        if self.player is not None and 0 <= index < len(self.excerpts):
            self.player.play_excerpt(self.excerpts[index])

    def _save(self) -> None:
        for row, excerpt in enumerate(self.excerpts):
            decision = str(self._decision_boxes[row].currentData() or "")
            actual_name = self._actual_boxes[row].currentText().strip() if decision == "BAD" else ""
            self.choices[_excerpt_key(excerpt)] = SampleChoice(excerpt, decision, actual_name)
        self.accept()


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
        self._sample_choices: dict[str, SampleChoice] = {}
        self._profile_names = [profile.name for profile in self.service.profiles.list_profiles()]

        self.setWindowTitle("Identifier les interlocuteurs")
        self.resize(1260, 800)
        intro = QLabel(
            "Choisissez une personne déjà connue dans la liste, ou saisissez un nouveau nom. Avant de mémoriser, "
            "vérifiez les extraits un par un : un extrait incorrect peut être exclu ou réattribué sans contaminer le profil."
        )
        intro.setWordWrap(True)
        privacy = QLabel(
            "La mémorisation reste volontaire. Seuls les extraits explicitement classés Bonne voix sont ajoutés comme "
            "références vérifiables. Une Mauvaise voix attribuée à une autre personne devient une référence de cette personne."
        )
        privacy.setWordWrap(True)

        self.table = QTableWidget(0, 7)
        self.table.setHorizontalHeaderLabels([
            "Sélection / groupe", "Durée", "Interlocuteur", "Extraits audio", "Vérification", "Voix", "Mémoriser",
        ])
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.verticalHeader().setVisible(False)
        self._name_boxes: dict[str, QComboBox] = {}
        self._remember: dict[str, QCheckBox] = {}
        self._selected: dict[str, QCheckBox] = {}
        self._excerpts: dict[str, list[Any]] = {}
        self._excerpt_choices: dict[str, QComboBox] = {}

        self.merge_button = QPushButton("Fusionner la sélection")
        self.split_button = QPushButton("Défusionner la sélection")
        self.undo_button = QPushButton("Annuler la dernière fusion / défusion")
        self.stop_button = QPushButton("Arrêter l'écoute")
        self.source_button = QPushButton("Associer l'audio d'origine…")
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
        self.preview.setPlaceholderText("Aperçu de la transcription corrigée")
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
            warning = QLabel("Attention : les documents déjà générés (CR, résumé…) ne sont pas réécrits par cette correction.")
            warning.setWordWrap(True)
            layout.addWidget(warning)
        layout.addWidget(QLabel("Aperçu (non enregistré avant Appliquer)"))
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

    def _identity_box(self, initial: str) -> QComboBox:
        box = QComboBox()
        box.setEditable(True)
        box.setInsertPolicy(QComboBox.InsertPolicy.NoInsert)
        box.addItems(self._profile_names)
        if initial and box.findText(initial, Qt.MatchFlag.MatchFixedString) < 0:
            box.addItem(initial)
        box.setCurrentText(initial)
        if box.lineEdit() is not None:
            box.lineEdit().setMaxLength(80)
            box.lineEdit().setPlaceholderText("Choisir ou saisir un nouveau nom")
        return box

    def _render(self) -> None:
        self._stop()
        previous_names = {key: box.currentText() for key, box in self._name_boxes.items()}
        previous_remember = {key: box.isChecked() for key, box in self._remember.items()}
        self.table.setRowCount(0)
        self._name_boxes.clear()
        self._remember.clear()
        self._selected.clear()
        self._excerpts.clear()
        self._excerpt_choices.clear()
        groups = self.state.ordered_groups()
        self.table.setRowCount(len(groups))

        for row, group in enumerate(groups):
            key = group.group_id
            selected = QCheckBox(key if len(group.members) == 1 else f"{len(group.members)} groupes fusionnés")
            selected.setToolTip(", ".join(group.members))
            selected.toggled.connect(self._update_actions)
            self._selected[key] = selected
            self.table.setCellWidget(row, 0, selected)
            self.table.setItem(row, 1, QTableWidgetItem(_clock(self.state.duration(key))))

            identity = self._identity_box(previous_names.get(key, group.name))
            identity.currentTextChanged.connect(lambda text, group_id=key: self._rename(group_id, text))
            self._name_boxes[key] = identity
            self.table.setCellWidget(row, 2, identity)

            excerpts = self.service.excerpts(self.payload, group.members, source_override=self.source_override)
            self._excerpts[key] = excerpts
            choices = QComboBox()
            for excerpt in excerpts:
                mode = "isolée" if excerpt.isolated else "mixée"
                warning = " / chevauchement" if excerpt.overlap_possible else ""
                choices.addItem(f"{excerpt.cluster_id} · {_clock(excerpt.global_start)} · {mode}{warning}")
            choices.setMinimumWidth(250)
            choices.setEnabled(bool(excerpts))
            self._excerpt_choices[key] = choices
            listen = QPushButton("Écouter")
            listen.setEnabled(bool(excerpts) and self._player is not None)
            listen.clicked.connect(lambda _checked=False, group_id=key: self._listen(group_id))
            audio_cell = QWidget()
            audio_line = QHBoxLayout(audio_cell)
            audio_line.setContentsMargins(0, 0, 0, 0)
            audio_line.addWidget(choices, 1)
            audio_line.addWidget(listen)
            self.table.setCellWidget(row, 3, audio_cell)

            verified = QPushButton("Vérifier les voix…")
            verified.setEnabled(bool(excerpts))
            verified.clicked.connect(lambda _checked=False, group_id=key: self._review_samples(group_id))
            self.table.setCellWidget(row, 4, verified)

            reviewed = sum(1 for excerpt in excerpts if self._sample_choices.get(_excerpt_key(excerpt), SampleChoice(excerpt)).decision)
            good = sum(1 for excerpt in excerpts if self._sample_choices.get(_excerpt_key(excerpt), SampleChoice(excerpt)).decision == "GOOD")
            detail = QTableWidgetItem(f"{len(excerpts)} extrait(s) · {reviewed} vérifié(s) · {good} bon(s)")
            detail.setToolTip("La mémorisation utilise les extraits audibles vérifiés, pas toutes les empreintes du cluster en bloc.")
            self.table.setItem(row, 5, detail)

            remember = QCheckBox("Mémoriser")
            remember.setEnabled(bool(excerpts))
            remember.setChecked(previous_remember.get(key, group.remember and bool(excerpts)))
            remember.toggled.connect(lambda checked, group_id=key: self._remember_changed(group_id, checked))
            remember.setToolTip("Seules les voix classées Bonne voix seront mémorisées pour cet interlocuteur.")
            self._remember[key] = remember
            self.table.setCellWidget(row, 6, remember)
            self.table.setRowHeight(row, 44)

        self.table.resizeColumnsToContents()
        self.table.horizontalHeader().setStretchLastSection(True)
        if self._playback_problem:
            self.audio_status.setText(self._playback_problem)
        elif not any(self._excerpts.values()):
            self.audio_status.setText("Aucun audio associé. Choisissez l'audio original, sans coupe ni modification.")
        else:
            self.audio_status.setText("Pour mémoriser une personne, ouvrez Vérifier les voix et classez chaque extrait utile séparément.")
        self._refresh_preview()
        self._update_actions()

    def _rename(self, key: str, text: str) -> None:
        self.state.groups[key].name = text.strip()
        self._refresh_preview()

    def _remember_changed(self, key: str, checked: bool) -> None:
        self.state.groups[key].remember = checked

    def _review_samples(self, key: str) -> None:
        name = self._name_boxes[key].currentText().strip()
        if not name:
            QMessageBox.warning(self, "Nom requis", "Choisissez ou saisissez d'abord le nom de l'interlocuteur.")
            return
        dialog = SpeakerSampleReviewDialog(name, self._excerpts.get(key, []), self._profile_names, self._sample_choices, self._player, self)
        dialog.exec()
        dialog.deleteLater()
        self._render()

    def _refresh_preview(self) -> None:
        self.preview.setPlainText(self.state.preview(self.original).speaker_text())

    def _selection(self) -> list[str]:
        return [key for key, box in self._selected.items() if box.isChecked()]

    def _update_actions(self, *_args: Any) -> None:
        keys = self._selection()
        self.merge_button.setEnabled(len(keys) >= 2)
        self.split_button.setEnabled(len(keys) == 1 and len(self.state.groups[keys[0]].members) > 1)
        self.undo_button.setEnabled(self.state.can_undo)
        self.stop_button.setEnabled(self._player is not None)

    def _merge(self) -> None:
        keys = self._selection()
        if len(keys) < 2:
            return
        suggested = self._name_boxes[keys[0]].currentText() if keys[0] in self._name_boxes else self.state.groups[keys[0]].name
        name, accepted = QInputDialog.getText(self, "Fusionner les interlocuteurs", "Nom du groupe fusionné :", QLineEdit.EchoMode.Normal, suggested)
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
        name, _filter = QFileDialog.getOpenFileName(self, "Choisir exactement l'audio original (mêmes horodatages)", "", "Médias (*.wav *.mp3 *.m4a *.flac *.ogg *.aac *.mp4 *.mkv *.webm *.mov);;Tous (*)")
        if name:
            self.source_override = Path(name).resolve()
            self._sample_choices.clear()
            self._render()

    def _selected_sample_actions(self, remember: set[str]) -> list[tuple[str, SampleChoice]]:
        actions: list[tuple[str, SampleChoice]] = []
        for key in remember:
            owner = self._name_boxes[key].currentText().strip()
            for excerpt in self._excerpts.get(key, []):
                choice = self._sample_choices.get(_excerpt_key(excerpt))
                if not choice:
                    continue
                if choice.decision == "GOOD":
                    actions.append((owner, choice))
                elif choice.decision == "BAD" and choice.actual_name.strip():
                    actions.append((choice.actual_name.strip(), choice))
        return actions

    def _apply(self) -> None:
        if self._applying:
            return
        names = {key: box.currentText() for key, box in self._name_boxes.items()}
        remember = {key for key, checkbox in self._remember.items() if checkbox.isChecked()}
        try:
            self.state.set_choices(names, remember)
        except ValueError as exc:
            QMessageBox.warning(self, "Vérifiez les noms", str(exc))
            return

        for key in remember:
            excerpts = self._excerpts.get(key, [])
            decisions = [self._sample_choices.get(_excerpt_key(item)) for item in excerpts]
            if not any(item and item.decision == "GOOD" for item in decisions):
                QMessageBox.warning(self, "Voix non vérifiées", f"{self._name_boxes[key].currentText().strip()} est coché Mémoriser mais aucun extrait n'est classé Bonne voix. Ouvrez « Vérifier les voix… ».")
                return

        actions = self._selected_sample_actions(remember)
        if remember and QMessageBox.question(self, "Confirmer la mémorisation vocale", f"{len(actions)} extrait(s) vérifié(s) seront mémorisés ou réattribués. Continuer ?", QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No, QMessageBox.StandardButton.No) != QMessageBox.StandardButton.Yes:
            return

        self._applying = True
        self.buttons.setEnabled(False)
        self._stop()
        try:
            vector_by_key: dict[str, tuple[float, ...]] = {}
            if actions:
                unique: list[Any] = []
                seen: set[str] = set()
                for _target, choice in actions:
                    key = _excerpt_key(choice.excerpt)
                    if key not in seen:
                        seen.add(key)
                        unique.append(choice.excerpt)
                sampler = SpeakerSampleService(self.service.paths)
                try:
                    vectors = sampler.extract(unique, compute_device="cuda")
                except Exception:
                    vectors = sampler.extract(unique, compute_device="cpu")
                for excerpt, vector in zip(unique, vectors):
                    if vector is not None:
                        vector_by_key[_excerpt_key(excerpt)] = vector

            self.service.apply(
                self.review_id,
                names,
                set(),
                self_speaker_name=self.self_name,
                groups=self.state.serialized_groups(),
                expected_revision=self.payload.get("_revision"),
                expected_transcript_revision=self.payload.get("_transcript_revision"),
                source_override=self.source_override,
            )

            if actions:
                profiles = list(self.service.profiles.list_profiles())
                by_name = {item.name.casefold(): item for item in profiles}
                grouped_vectors: dict[str, list[tuple[float, ...]]] = {}
                for target, choice in actions:
                    vector = vector_by_key.get(_excerpt_key(choice.excerpt))
                    if vector is not None:
                        grouped_vectors.setdefault(target, []).append(vector)

                for target, vectors in grouped_vectors.items():
                    if target.casefold() in by_name or not vectors:
                        continue
                    profile = self.service.profiles.enroll(target, str(self.payload["model_set_id"]), vectors, is_self=target.casefold() == self.self_name.strip().casefold())
                    by_name[target.casefold()] = profile

                profiles = list(self.service.profiles.list_profiles())
                self.service.profiles.evidence.save_quarantine(profiles)
                records: list[EvidenceRecordRequest] = []
                for target, choice in actions:
                    vector = vector_by_key.get(_excerpt_key(choice.excerpt))
                    profile = by_name.get(target.casefold())
                    if vector is None or profile is None:
                        continue
                    records.append(EvidenceRecordRequest(
                        profile_id=profile.profile_id,
                        embedding=vector,
                        status="GOOD",
                        excerpt=choice.excerpt,
                        metadata={
                            "profile_name": profile.name,
                            "model_set_id": profile.model_set_id,
                            "review_id": self.review_id,
                            "source": "manual_in_app",
                            "source_name": Path(choice.excerpt.path).name,
                            "actual_name": profile.name,
                        },
                    ))
                if records:
                    self.service.profiles.evidence.save_records(records)
        except Exception as exc:
            self._applying = False
            self.buttons.setEnabled(True)
            QMessageBox.critical(self, "Modification impossible", str(exc))
            return
        self.accept()

    def done(self, result: int) -> None:
        if self._player is not None:
            self._player.close()
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
        self.resize(650, 430)
        info = QLabel("Les profils vocaux sont locaux et chiffrés. En mode strict, seules les voix GOOD vérifiées avec un audio rejouable servent à l'identification automatique.")
        info.setWordWrap(True)
        self.list = QListWidget()
        self.list.currentItemChanged.connect(self._selection)
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
        rename_line = QHBoxLayout()
        rename_line.addWidget(self.rename_edit, 1)
        rename_line.addWidget(self.rename_button)
        buttons = QHBoxLayout()
        buttons.addWidget(self.delete_button)
        buttons.addWidget(self.delete_all_button)
        buttons.addStretch()
        buttons.addWidget(self.close_button)
        layout = QVBoxLayout(self)
        layout.addWidget(info)
        layout.addWidget(self.list, 1)
        layout.addLayout(rename_line)
        layout.addLayout(buttons)
        self._reload()

    def _reload(self) -> None:
        self.list.clear()
        for profile in self.service.list_profiles():
            counts = self.service.evidence.status_counts(profile.profile_id)
            suffix = " · profil personnel" if profile.is_self else ""
            item = QListWidgetItem(f"{profile.name} · {counts['GOOD']} voix vérifiées GOOD · {len(profile.embeddings)} legacy{suffix}")
            item.setData(Qt.ItemDataRole.UserRole, profile.profile_id)
            item.setData(Qt.ItemDataRole.UserRole + 1, profile.name)
            self.list.addItem(item)
        if self.list.count():
            self.list.setCurrentRow(0)
        self._selection()

    def _selection(self, *_args: Any) -> None:
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
        if QMessageBox.question(self, "Supprimer le profil", "Supprimer définitivement ce profil vocal local ?", QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No, QMessageBox.StandardButton.No) != QMessageBox.StandardButton.Yes:
            return
        try:
            self.service.delete(str(item.data(Qt.ItemDataRole.UserRole)))
            self._reload()
        except Exception as exc:
            QMessageBox.critical(self, "Suppression impossible", str(exc))

    def _delete_all(self) -> None:
        if QMessageBox.question(self, "Supprimer toutes les empreintes", "Supprimer définitivement toutes les empreintes vocales enregistrées sur ce compte Windows ?", QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No, QMessageBox.StandardButton.No) != QMessageBox.StandardButton.Yes:
            return
        try:
            self.service.delete_all()
            self._reload()
        except Exception as exc:
            QMessageBox.critical(self, "Suppression impossible", str(exc))
