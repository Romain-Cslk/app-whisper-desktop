# ruff: noqa: E402
"""Real Qt widget tests (run with the project dev environment, not mocked Qt)."""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pytest

pytest.importorskip('PySide6')
from PySide6.QtWidgets import QDialog, QInputDialog, QMessageBox

from transcripteur_whisper.models.transcript import StructuredTranscript, TranscriptSegment
from transcripteur_whisper.services.speaker_audio import AudioExcerpt
from transcripteur_whisper.ui.widgets.speaker_review import SpeakerReviewDialog


class Service:
    def __init__(self):
        self.applied = []
        self.payload = {'clusters': [
            {'cluster_id': f's{i}', 'source': 'mixed', 'current_name': f'Intervenant {i + 1}',
             'duration': 12, 'embeddings': [[float(j == i) for j in range(32)]]}
            for i in range(3)
        ], '_revision': 'revision', '_transcript_revision': 'text-revision'}

    def load_review(self, _id):
        return deepcopy(self.payload)

    def transcript(self, _id):
        return StructuredTranscript(tuple(TranscriptSegment(i * 12, i * 12 + 10, f'Phrase {i}',
                                    speaker_key=f's{i}', speaker_name=f'Intervenant {i + 1}') for i in range(3)))

    def excerpts(self, payload, members, *, source_override=None):
        return [AudioExcerpt(source_override or Path('sample.wav'), 0, 3, 0, key, False) for key in members]

    def apply(self, *args, **kwargs):
        self.applied.append((args, kwargs))


@pytest.fixture
def dialog(qtbot):
    widget = SpeakerReviewDialog(Service(), 'review', 'Romain')
    qtbot.addWidget(widget)
    return widget


def merge(dialog, monkeypatch):
    monkeypatch.setattr(QInputDialog, 'getText', lambda *args, **kwargs: ('Paul', True))
    dialog._selected['s0'].setChecked(True)
    dialog._selected['s2'].setChecked(True)
    assert dialog.merge_button.isEnabled()
    dialog.merge_button.click()


def test_widget_merge_undo_split_and_immediate_preview(dialog, monkeypatch):
    assert not any(box.isChecked() for box in dialog._remember.values())
    merge(dialog, monkeypatch)
    assert dialog.table.rowCount() == 2
    assert dialog.preview.toPlainText().count('Paul') == 2
    assert len(dialog._excerpts['s0']) == 2
    dialog.undo_button.click()
    assert dialog.table.rowCount() == 3 and 'Paul' not in dialog.preview.toPlainText()
    merge(dialog, monkeypatch)
    dialog._selected['s0'].setChecked(True)
    dialog.split_button.click()
    assert dialog.table.rowCount() == 3
    dialog._name_edits['s1'].setText('Julie')
    assert 'Julie' in dialog.preview.toPlainText()
    assert not dialog.service.applied


def test_widget_cancel_never_calls_apply_and_clears_memory(dialog):
    dialog._name_edits['s0'].setText('Paul')
    dialog.reject()
    assert dialog.result() == QDialog.DialogCode.Rejected
    assert not dialog.service.applied
    assert all(not item['embeddings'] for item in dialog.state.clusters.values())


def test_widget_apply_passes_partition_and_revision_guards(dialog, monkeypatch):
    merge(dialog, monkeypatch)
    dialog._apply()
    args, kwargs = dialog.service.applied[0]
    assert args[2] == set()
    assert kwargs['groups'][0]['members'] == ['s0', 's2']
    assert kwargs['expected_revision'] == 'revision'
    assert kwargs['expected_transcript_revision'] == 'text-revision'
    assert dialog.result() == QDialog.DialogCode.Accepted


def test_widget_biometric_consent_declined_does_not_apply(dialog, monkeypatch):
    dialog._remember['s0'].setChecked(True)
    monkeypatch.setattr(QMessageBox, 'question', lambda *args: QMessageBox.StandardButton.No)
    dialog._apply()
    assert not dialog.service.applied
    monkeypatch.setattr(QMessageBox, 'question', lambda *args: QMessageBox.StandardButton.Yes)
    dialog._apply()
    assert dialog.service.applied[0][0][2] == {'s0'}


def test_widget_apply_failure_stays_open_and_can_retry(dialog, monkeypatch):
    errors = []
    monkeypatch.setattr(QMessageBox, 'critical', lambda *args: errors.append(args[2]))
    original = dialog.service.apply
    def failure(*args, **kwargs):
        raise OSError('disk full')
    dialog.service.apply = failure
    dialog._apply()
    assert errors == ['disk full'] and dialog.buttons.isEnabled() and not dialog._applying
    dialog.service.apply = original
    dialog._apply()
    assert len(dialog.service.applied) == 1


def test_widget_switching_excerpts_and_closing_stops_player(dialog):
    class FakePlayer:
        def __init__(self):
            self.played = []
            self.closed = False
        def play_excerpt(self, excerpt):
            self.played.append(excerpt)
        def stop(self):
            pass
        def close(self):
            self.closed = True
    if dialog._player is not None:
        dialog._player.close()
    player = FakePlayer()
    dialog._player = player
    dialog._listen('s0')
    dialog._listen('s1')
    assert [item.cluster_id for item in player.played] == ['s0', 's1']
    dialog.reject()
    assert player.closed
