"""Manual, local review of stored speaker-profile embeddings against recoverable audio.

This tool is intentionally non-destructive. It reconstructs the audio windows that
produced historical speaker embeddings when the original audio/review metadata still
exists, links them back to the selected stored profile by cosine similarity, and lets
a human classify every stored embedding as GOOD / BAD / UNSURE.

No biometric vector or audio leaves the machine. Decisions are stored locally under
TranscripteurWhisper/speakers/manual_profile_audio_review.v1.json.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from transcripteur_whisper.core.paths import AppPaths
from transcripteur_whisper.models.speaker import DiarizationTurn
from transcripteur_whisper.services.diarization_model_service import DiarizationModelService
from transcripteur_whisper.services.diarization_worker import (
    _decode_audio,
    _embedding,
    _embedding_windows,
    _make_extractor,
)
from transcripteur_whisper.services.dpapi import WindowsDpapiCodec
from transcripteur_whisper.services.speaker_audio import AudioExcerpt
from transcripteur_whisper.services.speaker_profile_service import SpeakerProfileService


DEFAULT_LINK_SIMILARITY = 0.97
DECISIONS_FILE = "manual_profile_audio_review.v1.json"
AUDIT_FILE = "audit_profils_global.csv"


@dataclass(frozen=True)
class CandidateAudio:
    embedding: np.ndarray
    excerpt: AudioExcerpt
    review_id: str
    review_name: str
    historical_name: str
    source: str


@dataclass
class ReviewRow:
    embedding_index: int
    audit_status: str
    audit_other: str
    audit_margin: float | None
    candidate: CandidateAudio | None
    link_similarity: float | None
    decision: str = ""
    actual_name: str = ""
    note: str = ""


def normalize(values: Any) -> np.ndarray:
    value = np.asarray(values, dtype=np.float32)
    if value.ndim != 1 or value.size < 16 or not np.isfinite(value).all():
        raise ValueError("Empreinte vocale invalide")
    norm = float(np.linalg.norm(value))
    if not math.isfinite(norm) or norm <= 1e-8:
        raise ValueError("Empreinte vocale vide")
    return value / norm


def load_audit(repo: Path, profile_name: str) -> dict[int, dict[str, str]]:
    path = repo / AUDIT_FILE
    if not path.is_file():
        return {}
    rows: dict[int, dict[str, str]] = {}
    try:
        with path.open("r", newline="", encoding="utf-8-sig") as handle:
            for item in csv.DictReader(handle):
                if str(item.get("profile") or "").casefold() != profile_name.casefold():
                    continue
                try:
                    index = int(item["embedding_index"])
                except (ValueError, TypeError, KeyError):
                    continue
                rows[index] = dict(item)
    except OSError:
        return {}
    return rows


def read_review(path: Path, codec: WindowsDpapiCodec) -> dict[str, Any] | None:
    try:
        payload = json.loads(codec.unprotect(path.read_bytes()).decode("utf-8"))
    except Exception:
        return None
    if not isinstance(payload, dict) or payload.get("schema_version") not in {1, 2}:
        return None
    return payload


def parse_turns(payload: dict[str, Any]) -> list[DiarizationTurn]:
    output: list[DiarizationTurn] = []
    for item in payload.get("turns") or []:
        try:
            turn = DiarizationTurn.from_dict(item)
        except (ValueError, TypeError, KeyError):
            continue
        if math.isfinite(turn.start) and math.isfinite(turn.end) and turn.end > turn.start >= 0:
            output.append(turn)
    return output


def source_for_cluster(
    payload: dict[str, Any], source: str
) -> tuple[Path, float, bool, bool] | None:
    sources = payload.get("audio_sources") or {}
    descriptor = sources.get(source)
    isolated = False
    fallback_mixed = False

    if descriptor and descriptor.get("path") and Path(str(descriptor["path"])).is_file():
        isolated = source in {"self", "system"}
    else:
        descriptor = sources.get("mixed")
        fallback_mixed = True

    if not descriptor or not descriptor.get("path"):
        return None

    path = Path(str(descriptor["path"])).expanduser().resolve()
    if not path.is_file():
        return None

    try:
        offset = float(descriptor.get("offset") or 0.0)
    except (ValueError, TypeError):
        return None
    if not math.isfinite(offset) or offset < 0:
        return None
    return path, offset, isolated, fallback_mixed


def has_overlap(start: float, end: float, cluster_id: str, turns: list[DiarizationTurn]) -> bool:
    return any(
        item.cluster_id != cluster_id
        and min(end, item.end) - max(start, item.start) > 0.08
        for item in turns
    )


def reconstruct_candidates(
    paths: AppPaths,
    *,
    provider: str,
    profile_model_set_id: str,
    progress=print,
) -> list[CandidateAudio]:
    codec = WindowsDpapiCodec()
    pending = Path(paths.speakers) / "pending"
    review_files = sorted(pending.glob("*.dpapi"))
    if not review_files:
        return []

    model_service = DiarizationModelService(paths)
    _segmentation, embedding_model = model_service.ensure()
    extractor = _make_extractor(embedding_model, provider)
    audio_cache: dict[Path, tuple[np.ndarray, int]] = {}
    seen: set[tuple[str, int, int, str]] = set()
    candidates: list[CandidateAudio] = []

    for review_number, review_path in enumerate(review_files, 1):
        payload = read_review(review_path, codec)
        if payload is None:
            continue
        if str(payload.get("model_set_id") or "") != profile_model_set_id:
            continue

        turns = parse_turns(payload)
        if not turns:
            continue
        clusters = {str(item.get("cluster_id")): item for item in payload.get("clusters") or []}
        if not clusters:
            continue

        if review_number == 1 or review_number % 10 == 0:
            progress(f"Indexation audio : revue {review_number}/{len(review_files)}...")

        for cluster_id, cluster in clusters.items():
            source = str(cluster.get("source") or "mixed")
            resolved = source_for_cluster(payload, source)
            if resolved is None:
                continue
            audio_path, offset, isolated, fallback_mixed = resolved

            relevant_global = turns if fallback_mixed else [item for item in turns if item.source == source]
            local_turns = [
                DiarizationTurn(
                    max(0.0, item.start - offset),
                    max(0.0, item.end - offset),
                    item.cluster_id,
                    item.source,
                )
                for item in relevant_global
                if item.end > offset
            ]
            windows = _embedding_windows(local_turns, cluster_id)
            if not windows:
                continue

            saved_embeddings = cluster.get("embeddings") or []
            direct_vectors: list[np.ndarray] | None = None
            if len(saved_embeddings) == len(windows):
                try:
                    direct_vectors = [normalize(item) for item in saved_embeddings]
                except ValueError:
                    direct_vectors = None

            if direct_vectors is None:
                if audio_path not in audio_cache:
                    try:
                        audio_cache[audio_path] = _decode_audio(audio_path)
                    except Exception as exc:
                        progress(f"Audio ignore ({audio_path.name}) : {exc}")
                        continue
                samples, sample_rate = audio_cache[audio_path]
            else:
                samples = np.empty(0, dtype=np.float32)
                sample_rate = 16_000

            for position, (start, end) in enumerate(windows):
                key = (
                    str(audio_path).casefold(),
                    round(start * 1000),
                    round(end * 1000),
                    cluster_id,
                )
                if key in seen:
                    continue

                if direct_vectors is not None:
                    vector = direct_vectors[position]
                else:
                    left = max(0, round(start * sample_rate))
                    right = min(samples.size, round(end * sample_rate))
                    if right <= left:
                        continue
                    value = _embedding(extractor, samples[left:right], sample_rate)
                    if value is None:
                        continue
                    vector = normalize(value)

                seen.add(key)
                overlap = has_overlap(start, end, cluster_id, local_turns)
                excerpt = AudioExcerpt(
                    audio_path,
                    start,
                    end,
                    start + offset,
                    cluster_id,
                    isolated,
                    overlap,
                )
                candidates.append(
                    CandidateAudio(
                        embedding=vector,
                        excerpt=excerpt,
                        review_id=review_path.stem,
                        review_name=audio_path.name,
                        historical_name=str(cluster.get("current_name") or ""),
                        source=source,
                    )
                )

    return candidates


def link_profile_embeddings(
    embeddings: tuple[tuple[float, ...], ...],
    candidates: list[CandidateAudio],
    minimum_similarity: float,
) -> list[tuple[CandidateAudio | None, float | None]]:
    targets = np.asarray([normalize(item) for item in embeddings], dtype=np.float32)
    if not candidates:
        return [(None, None) for _ in embeddings]
    candidate_matrix = np.asarray([item.embedding for item in candidates], dtype=np.float32)
    similarities = np.clip(targets @ candidate_matrix.T, -1.0, 1.0)
    output: list[tuple[CandidateAudio | None, float | None]] = []
    for row in similarities:
        index = int(np.argmax(row))
        score = float(row[index])
        if score >= minimum_similarity:
            output.append((candidates[index], score))
        else:
            output.append((None, score))
    return output


def load_decisions(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {"schema_version": 1, "profiles": {}}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {"schema_version": 1, "profiles": {}}
    if not isinstance(payload, dict) or payload.get("schema_version") != 1:
        return {"schema_version": 1, "profiles": {}}
    payload.setdefault("profiles", {})
    return payload


def save_decisions(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def audit_label(status: str) -> str:
    return {
        "FOREIGN_STRONG": "CRITIQUE",
        "FOREIGN_LIKELY": "SUSPECTE",
        "AMBIGUOUS": "AMBIGUE",
        "TRUSTED": "OK",
    }.get(status, status or "NON AUDITE")


def build_rows(
    profile: Any,
    links: list[tuple[CandidateAudio | None, float | None]],
    audit: dict[int, dict[str, str]],
    decisions: dict[str, Any],
) -> list[ReviewRow]:
    stored = decisions.get("profiles", {}).get(profile.profile_id, {}).get("embeddings", {})
    rows: list[ReviewRow] = []
    for index, (candidate, similarity) in enumerate(links, 1):
        item = audit.get(index, {})
        try:
            margin = float(item["margin"]) if item.get("margin") not in {None, ""} else None
        except ValueError:
            margin = None
        decision = stored.get(str(index), {}) if isinstance(stored, dict) else {}
        rows.append(
            ReviewRow(
                embedding_index=index,
                audit_status=str(item.get("status") or ""),
                audit_other=str(item.get("best_other_profile") or ""),
                audit_margin=margin,
                candidate=candidate,
                link_similarity=similarity,
                decision=str(decision.get("decision") or ""),
                actual_name=str(decision.get("actual_name") or ""),
                note=str(decision.get("note") or ""),
            )
        )
    return rows


def run_ui(paths: AppPaths, profile: Any, rows: list[ReviewRow], all_profile_names: list[str]) -> int:
    from PySide6.QtCore import Qt
    from PySide6.QtGui import QKeySequence, QShortcut
    from PySide6.QtWidgets import (
        QApplication,
        QComboBox,
        QHBoxLayout,
        QLabel,
        QLineEdit,
        QMainWindow,
        QMessageBox,
        QPushButton,
        QTableWidget,
        QTableWidgetItem,
        QVBoxLayout,
        QWidget,
    )
    from transcripteur_whisper.ui.widgets.speaker_player import SpeakerExcerptPlayer

    app = QApplication.instance() or QApplication(sys.argv)
    decisions_path = Path(paths.speakers) / DECISIONS_FILE
    payload = load_decisions(decisions_path)

    class Window(QMainWindow):
        def __init__(self) -> None:
            super().__init__()
            self.setWindowTitle(f"Audit manuel des voix - {profile.name}")
            self.resize(1320, 760)
            self.player = SpeakerExcerptPlayer(self)
            self.player.status.connect(self._set_status)
            self.player.failed.connect(self._set_status)
            self.visible_rows: list[ReviewRow] = []

            root = QWidget()
            layout = QVBoxLayout(root)

            intro = QLabel(
                f"Profil : {profile.name} - {len(rows)} empreinte(s). "
                "Ecoutez puis classez chaque empreinte vous-meme. "
                "Aucune empreinte n'est supprimee par cet outil."
            )
            intro.setWordWrap(True)
            layout.addWidget(intro)

            controls = QHBoxLayout()
            controls.addWidget(QLabel("Filtre :"))
            self.filter = QComboBox()
            self.filter.addItems([
                "Tous",
                "Problematiques d'abord",
                "Non verifies",
                "Critiques",
                "Suspectes / ambigues",
                "OK auto",
                "Sans audio retrouve",
            ])
            self.filter.currentIndexChanged.connect(self._render)
            controls.addWidget(self.filter)
            controls.addStretch()
            layout.addLayout(controls)

            self.table = QTableWidget(0, 10)
            self.table.setHorizontalHeaderLabels([
                "#",
                "Audit auto",
                "Decision humaine",
                "Audio",
                "Lien audio",
                "Nom historique",
                "Autre profil auto",
                "Marge",
                "Source",
                "Fichier",
            ])
            self.table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
            self.table.setSelectionMode(QTableWidget.SelectionMode.SingleSelection)
            self.table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
            self.table.itemSelectionChanged.connect(self._selection_changed)
            self.table.doubleClicked.connect(lambda _index: self._play())
            layout.addWidget(self.table, 1)

            actions = QHBoxLayout()
            self.play = QPushButton("Ecouter [Espace]")
            self.good = QPushButton("Bonne voix [B]")
            self.bad = QPushButton("Mauvaise voix [M]")
            self.unsure = QPushButton("Incertain [I]")
            self.play.clicked.connect(self._play)
            self.good.clicked.connect(lambda: self._decide("GOOD"))
            self.bad.clicked.connect(lambda: self._decide("BAD"))
            self.unsure.clicked.connect(lambda: self._decide("UNSURE"))
            for button in (self.play, self.good, self.bad, self.unsure):
                actions.addWidget(button)
            actions.addStretch()
            layout.addLayout(actions)

            correction = QHBoxLayout()
            correction.addWidget(QLabel("Si mauvaise voix, vraie personne :"))
            self.actual = QComboBox()
            self.actual.setEditable(True)
            self.actual.addItem("")
            self.actual.addItems(all_profile_names)
            correction.addWidget(self.actual, 1)
            correction.addWidget(QLabel("Note :"))
            self.note = QLineEdit()
            correction.addWidget(self.note, 2)
            layout.addLayout(correction)

            self.status = QLabel()
            self.status.setWordWrap(True)
            layout.addWidget(self.status)
            self.setCentralWidget(root)

            QShortcut(QKeySequence(Qt.Key.Key_Space), self, activated=self._play)
            QShortcut(QKeySequence("B"), self, activated=lambda: self._decide("GOOD"))
            QShortcut(QKeySequence("M"), self, activated=lambda: self._decide("BAD"))
            QShortcut(QKeySequence("I"), self, activated=lambda: self._decide("UNSURE"))

            self._render()

        def _priority(self, row: ReviewRow) -> tuple[int, int]:
            status = row.audit_status
            rank = {
                "FOREIGN_STRONG": 0,
                "FOREIGN_LIKELY": 1,
                "AMBIGUOUS": 2,
                "TRUSTED": 3,
                "": 4,
            }.get(status, 4)
            return rank, row.embedding_index

        def _filtered(self) -> list[ReviewRow]:
            mode = self.filter.currentText()
            candidates = list(rows)
            if mode == "Problematiques d'abord":
                return sorted(candidates, key=self._priority)
            if mode == "Non verifies":
                return [item for item in candidates if not item.decision]
            if mode == "Critiques":
                return [item for item in candidates if item.audit_status == "FOREIGN_STRONG"]
            if mode == "Suspectes / ambigues":
                return [item for item in candidates if item.audit_status in {"FOREIGN_LIKELY", "AMBIGUOUS"}]
            if mode == "OK auto":
                return [item for item in candidates if item.audit_status == "TRUSTED"]
            if mode == "Sans audio retrouve":
                return [item for item in candidates if item.candidate is None]
            return candidates

        def _render(self) -> None:
            selected_index = self.current_row().embedding_index if self.current_row() else None
            self.visible_rows = self._filtered()
            self.table.setRowCount(len(self.visible_rows))
            select_row = 0
            for table_row, item in enumerate(self.visible_rows):
                if item.embedding_index == selected_index:
                    select_row = table_row
                candidate = item.candidate
                values = [
                    str(item.embedding_index),
                    audit_label(item.audit_status),
                    {"GOOD": "BONNE", "BAD": "MAUVAISE", "UNSURE": "INCERTAIN"}.get(item.decision, "A FAIRE"),
                    "Disponible" if candidate else "Introuvable",
                    f"{item.link_similarity:.3f}" if item.link_similarity is not None else "-",
                    candidate.historical_name if candidate else "-",
                    item.audit_other or "-",
                    f"{item.audit_margin:.3f}" if item.audit_margin is not None else "-",
                    candidate.source if candidate else "-",
                    candidate.review_name if candidate else "-",
                ]
                for column, value in enumerate(values):
                    cell = QTableWidgetItem(value)
                    if column == 0:
                        cell.setData(Qt.ItemDataRole.UserRole, item.embedding_index)
                    self.table.setItem(table_row, column, cell)
            self.table.resizeColumnsToContents()
            self.table.horizontalHeader().setStretchLastSection(True)
            if self.visible_rows:
                self.table.selectRow(min(select_row, len(self.visible_rows) - 1))
            self._summary()

        def current_row(self) -> ReviewRow | None:
            selected = self.table.selectionModel().selectedRows() if self.table.rowCount() else []
            if not selected:
                return None
            row = selected[0].row()
            if not 0 <= row < len(self.visible_rows):
                return None
            return self.visible_rows[row]

        def _selection_changed(self) -> None:
            item = self.current_row()
            if item is None:
                return
            self.actual.setCurrentText(item.actual_name)
            self.note.setText(item.note)
            self.play.setEnabled(item.candidate is not None)

        def _play(self) -> None:
            item = self.current_row()
            if item is None or item.candidate is None:
                self._set_status("Aucun extrait audio retrouve pour cette empreinte.")
                return
            self.player.play_excerpt(item.candidate.excerpt)

        def _decide(self, decision: str) -> None:
            item = self.current_row()
            if item is None:
                return
            actual_name = self.actual.currentText().strip()
            note = self.note.text().strip()
            item.decision = decision
            item.actual_name = actual_name
            item.note = note

            profile_payload = payload.setdefault("profiles", {}).setdefault(profile.profile_id, {})
            profile_payload["name"] = profile.name
            profile_payload["model_set_id"] = profile.model_set_id
            profile_payload["updated_at"] = datetime.now(timezone.utc).isoformat()
            stored = profile_payload.setdefault("embeddings", {})
            stored[str(item.embedding_index)] = {
                "decision": decision,
                "actual_name": actual_name,
                "note": note,
                "updated_at": datetime.now(timezone.utc).isoformat(),
                "audit_status": item.audit_status,
                "linked_audio": bool(item.candidate),
                "link_similarity": item.link_similarity,
            }
            save_decisions(decisions_path, payload)

            current = self.table.currentRow()
            self._render()
            if self.table.rowCount():
                self.table.selectRow(min(current + 1, self.table.rowCount() - 1))

        def _summary(self) -> None:
            counts = {"GOOD": 0, "BAD": 0, "UNSURE": 0, "": 0}
            playable = 0
            for item in rows:
                counts[item.decision if item.decision in counts else ""] += 1
                playable += int(item.candidate is not None)
            self._set_status(
                f"Audio retrouve : {playable}/{len(rows)} | "
                f"Bonnes : {counts['GOOD']} | Mauvaises : {counts['BAD']} | "
                f"Incertaines : {counts['UNSURE']} | A faire : {counts['']} | "
                f"Decisions : {decisions_path}"
            )

        def _set_status(self, message: str) -> None:
            self.status.setText(message)

        def closeEvent(self, event) -> None:
            self.player.close()
            super().closeEvent(event)

    window = Window()
    window.show()
    return app.exec()


def choose_profile(profiles: list[Any], requested: str | None) -> Any:
    if requested:
        for profile in profiles:
            if profile.name.casefold() == requested.casefold():
                return profile
        raise SystemExit(f"Profil introuvable : {requested}")

    print("\n=== PROFILS VOCAUX ===")
    for index, profile in enumerate(profiles, 1):
        print(f"{index:2}. {profile.name:<25} {len(profile.embeddings):3} empreinte(s)")
    print()
    value = input("Profil a controler (numero ou nom) : ").strip()
    if value.isdigit() and 1 <= int(value) <= len(profiles):
        return profiles[int(value) - 1]
    for profile in profiles:
        if profile.name.casefold() == value.casefold():
            return profile
    raise SystemExit("Profil introuvable.")


def main() -> int:
    parser = argparse.ArgumentParser(description="Ecouter et classer manuellement les empreintes d'un profil vocal.")
    parser.add_argument("--profile", help="Nom exact du profil a controler")
    parser.add_argument("--provider", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--min-link-similarity", type=float, default=DEFAULT_LINK_SIMILARITY)
    args = parser.parse_args()

    if not 0.5 <= args.min_link_similarity <= 1.0:
        raise SystemExit("--min-link-similarity doit etre compris entre 0.5 et 1.0")

    paths = AppPaths.create()
    profile_service = SpeakerProfileService(paths)
    profiles = list(profile_service.list_profiles())
    if not profiles:
        raise SystemExit("Aucun profil vocal disponible.")

    profile = choose_profile(profiles, args.profile)
    print(f"\nProfil selectionne : {profile.name} ({len(profile.embeddings)} empreintes)")
    print("Recherche des extraits audio historiques. Cela peut prendre un moment...")

    candidates = reconstruct_candidates(
        paths,
        provider=args.provider,
        profile_model_set_id=profile.model_set_id,
    )
    print(f"Extraits audio reconstruits : {len(candidates)}")

    links = link_profile_embeddings(profile.embeddings, candidates, args.min_link_similarity)
    linked = sum(candidate is not None for candidate, _score in links)
    print(f"Empreintes avec audio retrouve : {linked}/{len(profile.embeddings)}")

    repo = Path.cwd()
    audit = load_audit(repo, profile.name)
    decisions = load_decisions(Path(paths.speakers) / DECISIONS_FILE)
    rows = build_rows(profile, links, audit, decisions)

    return run_ui(paths, profile, rows, [item.name for item in profiles])


if __name__ == "__main__":
    raise SystemExit(main())
