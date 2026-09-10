"""Safety regressions for automatic speaker identification and profile enrichment."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from transcripteur_whisper.core.paths import AppPaths
from transcripteur_whisper.models.speaker import SpeakerCluster
from transcripteur_whisper.services.speaker_profile_service import SpeakerProfileService


class TestOnlyCodec:
    __test__ = False

    def protect(self, data):
        return b"TEST-ONLY:" + data

    def unprotect(self, data):
        if not data.startswith(b"TEST-ONLY:"):
            raise ValueError("corrupt or foreign profile")
        return data[len(b"TEST-ONLY:"):]


def vec(index: int, dim: int = 192) -> tuple[float, ...]:
    value = np.zeros(dim, dtype=np.float32)
    value[index] = 1.0
    return tuple(map(float, value))


def service(tmp_path: Path) -> SpeakerProfileService:
    return SpeakerProfileService(AppPaths.create(tmp_path), codec=TestOnlyCodec())


def test_large_profile_bank_cannot_win_from_one_contaminated_reference(tmp_path):
    profiles = service(tmp_path)
    profiles.enroll("Paul", "model", [vec(0)] * 12)
    # Julie has many genuine references plus one accidental Paul-like sample.
    profiles.enroll("Julie", "model", [vec(1)] * 11 + [vec(0)])

    cluster = SpeakerCluster("q", "mixed", 30.0, tuple([vec(0)] * 6))
    match = profiles.match_cluster_for_model(cluster, "model")

    assert match.accepted
    assert match.name == "Paul"


def test_mixed_cluster_is_not_given_a_confident_name(tmp_path):
    profiles = service(tmp_path)
    profiles.enroll("Paul", "model", [vec(0)] * 10)
    profiles.enroll("Julie", "model", [vec(1)] * 10)

    cluster = SpeakerCluster("q", "mixed", 30.0, tuple([vec(0)] * 3 + [vec(1)] * 3))
    match = profiles.match_cluster_for_model(cluster, "model")

    assert not match.accepted


def test_established_profile_rejects_incoherent_enrichment(tmp_path):
    profiles = service(tmp_path)
    profiles.enroll("Paul", "model", [vec(0)] * 8)

    with pytest.raises(ValueError, match="contaminer"):
        profiles.enroll("Paul", "model", [vec(1), vec(1), vec(2), vec(2)])

    stored, = profiles.list_profiles()
    assert stored.name == "Paul"
    assert all(int(np.argmax(item)) == 0 for item in stored.embeddings)


def test_established_profile_accepts_coherent_enrichment(tmp_path):
    profiles = service(tmp_path)
    profiles.enroll("Paul", "model", [vec(0)] * 8)
    updated = profiles.enroll("Paul", "model", [vec(0), vec(0), vec(0)])

    assert updated.name == "Paul"
    assert len(updated.embeddings) == 11
