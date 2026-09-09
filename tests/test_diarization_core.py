from __future__ import annotations

import json

import numpy as np

from transcripteur_whisper.core.paths import AppPaths
from transcripteur_whisper.core.settings import SettingsStore
from transcripteur_whisper.models.speaker import (
    DiarizationResult,
    DiarizationTurn,
    SpeakerCluster,
    SpeakerMatch,
)
from transcripteur_whisper.models.transcript import (
    StructuredTranscript,
    TranscriptSegment,
    TranscriptWord,
)
from transcripteur_whisper.models.transcription import TranscriptionOptions
from transcripteur_whisper.services.speaker_profile_service import SpeakerProfileService
from transcripteur_whisper.services.speaker_review_service import SpeakerReviewService, SpeakerReviewStore
from transcripteur_whisper.services.transcript_alignment import align_transcript, build_labels


class PlainCodec:
    def protect(self, data: bytes) -> bytes:
        return b"TEST:" + data

    def unprotect(self, data: bytes) -> bytes:
        assert data.startswith(b"TEST:")
        return data[5:]


def vector(index: int, dimension: int = 192):
    value = np.zeros(dimension, dtype=np.float32)
    value[index] = 1.0
    return tuple(float(item) for item in value)


def test_options_and_settings_default_off(tmp_path):
    paths = AppPaths.create(tmp_path)
    settings = SettingsStore(paths)
    assert settings.get("diarization_enabled", False) is False
    settings.set("diarization_enabled", True)
    settings.set("self_speaker_name", "Alice")
    restored = SettingsStore(paths)
    assert restored.get("diarization_enabled") is True
    assert restored.get("self_speaker_name") == "Alice"
    options = TranscriptionOptions(diarization_enabled=True, self_speaker_name="Alice")
    options.validate("")


def test_voice_profile_is_only_written_by_explicit_enroll_and_model_space_is_strict(tmp_path):
    paths = AppPaths.create(tmp_path)
    profiles = SpeakerProfileService(paths, codec=PlainCodec())
    cluster = SpeakerCluster("remote:00", "system", 12.0, (vector(0), vector(0)))
    # Matching alone never creates a profile.
    assert not profiles.match_cluster_for_model(cluster, "model-v1").accepted
    assert profiles.list_profiles() == ()
    profiles.enroll("Julie", "model-v1", [vector(0), vector(0)])
    matched = profiles.match_cluster_for_model(cluster, "model-v1")
    assert matched.accepted and matched.name == "Julie"
    incompatible = profiles.match_cluster_for_model(cluster, "model-v2")
    assert not incompatible.accepted and incompatible.profile_id is None
    # Similarity ambiguity is rejected because the second-best margin is too low.
    profiles.enroll("Juliette", "model-v1", [vector(0), vector(0)])
    ambiguous = profiles.match_cluster_for_model(cluster, "model-v1")
    assert not ambiguous.accepted


def test_self_track_wins_and_remote_profile_is_named(tmp_path):
    paths = AppPaths.create(tmp_path)
    profiles = SpeakerProfileService(paths, codec=PlainCodec())
    profiles.enroll("Julie", "model-v1", [vector(1), vector(1)])
    diarization = DiarizationResult(
        "model-v1",
        (
            DiarizationTurn(0.0, 2.0, "self", "self"),
            DiarizationTurn(2.0, 5.0, "remote:00", "system"),
        ),
        (
            SpeakerCluster("self", "self", 2.0, (vector(0), vector(0))),
            SpeakerCluster("remote:00", "system", 8.0, (vector(1), vector(1))),
        ),
    )
    labels, matches = build_labels(diarization, profiles, "Romain")
    assert labels == {"self": "Romain", "remote:00": "Julie"}
    transcript = StructuredTranscript(
        (
            TranscriptSegment(
                0.2,
                4.8,
                " Bonjour oui.",
                (
                    TranscriptWord(0.2, 1.0, " Bonjour"),
                    TranscriptWord(2.5, 3.0, " oui"),
                    TranscriptWord(3.0, 3.1, "."),
                ),
            ),
        ),
        "fr",
    )
    aligned = align_transcript(transcript, diarization, labels, matches)
    assert [segment.speaker_name for segment in aligned.segments] == ["Romain", "Julie"]
    text = aligned.speaker_text()
    assert "Romain" in text and "Julie" in text


def test_review_relabels_and_enrolls_only_checked_cluster(tmp_path):
    paths = AppPaths.create(tmp_path)
    profiles = SpeakerProfileService(paths, codec=PlainCodec())
    store = SpeakerReviewStore(paths, codec=PlainCodec())
    service = SpeakerReviewService(paths, profiles=profiles, store=store)
    job_id = "a" * 32
    job_dir = paths.results / job_id
    job_dir.mkdir()
    structured = StructuredTranscript(
        (TranscriptSegment(0, 2, "Bonjour", speaker_key="speaker:00", speaker_name="Intervenant 1"),),
        "fr",
    )
    structured_path = job_dir / "meeting.structured.json"
    transcript_path = job_dir / "meeting.txt"
    structured_path.write_text(json.dumps(structured.to_dict()), encoding="utf-8")
    transcript_path.write_text(structured.speaker_text(), encoding="utf-8")
    diarization = DiarizationResult(
        "model-v1",
        (DiarizationTurn(0, 2, "speaker:00"),),
        (SpeakerCluster("speaker:00", "mixed", 8.0, (vector(2), vector(2))),),
    )
    review_id = f"{job_id}-000-{'b' * 32}"
    service.create_review(
        review_id=review_id,
        job_id=job_id,
        file_index=0,
        diarization=diarization,
        labels={"speaker:00": "Intervenant 1"},
        matches={"speaker:00": SpeakerMatch(None, None, False, 0, 0, False, "unknown")},
        structured_filename=structured_path.name,
        transcript_filename=transcript_path.name,
    )
    # Rename without consent: no biometric profile, and pending voice samples are purged.
    service.apply(review_id, {"speaker:00": "Julie"}, set(), self_speaker_name="Romain")
    assert profiles.list_profiles() == ()
    assert "Julie" in transcript_path.read_text(encoding="utf-8")
    assert service.load_review(review_id)["clusters"][0]["embeddings"] == []

    # A fresh review with explicit consent persists the voiceprint once.
    second_review = f"{job_id}-000-{'c' * 32}"
    service.create_review(
        review_id=second_review,
        job_id=job_id,
        file_index=0,
        diarization=diarization,
        labels={"speaker:00": "Intervenant 1"},
        matches={"speaker:00": SpeakerMatch(None, None, False, 0, 0, False, "unknown")},
        structured_filename=structured_path.name,
        transcript_filename=transcript_path.name,
    )
    service.apply(second_review, {"speaker:00": "Julie"}, {"speaker:00"}, self_speaker_name="Romain")
    assert profiles.list_profiles()[0].name == "Julie"
    assert service.load_review(second_review)["clusters"][0]["embeddings"] == []
