"""Deterministic regressions. No voice data, neural weights, network or plaintext production fallback."""
from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from transcripteur_whisper.core.paths import AppPaths
from transcripteur_whisper.models.speaker import DiarizationResult, DiarizationTurn, SpeakerCluster
from transcripteur_whisper.models.transcript import StructuredTranscript, TranscriptSegment, TranscriptWord
from transcripteur_whisper.services import diarization_worker as worker
from transcripteur_whisper.services import speaker_persistence as persistence
from transcripteur_whisper.services.diarization_model_service import MODEL_SET_ID
from transcripteur_whisper.services.diarization_options import validate_diarization_options
from transcripteur_whisper.services.diarization_service import DiarizationError, DiarizationService
from transcripteur_whisper.services.recording_sources import find_recording_sources
from transcripteur_whisper.services.speaker_audio import AudioExcerpt, excerpts_for
from transcripteur_whisper.services.speaker_profile_service import SpeakerProfileService
from transcripteur_whisper.services.speaker_review_service import SpeakerReviewService, SpeakerReviewStore
from transcripteur_whisper.services.speaker_review_state import SpeakerReviewState
from transcripteur_whisper.services.transcript_alignment import _best_turn, align_transcript
from transcripteur_whisper.services.transcript_sources import combine_sources, shift_source


class TestOnlyCodec:
    __test__ = False

    def protect(self, data):
        return b"TEST-ONLY:" + data

    def unprotect(self, data):
        if not data.startswith(b"TEST-ONLY:"):
            raise ValueError("corrupt or foreign profile")
        return data[len(b"TEST-ONLY:"):]


def vec(index, dim=192):
    result = np.zeros(dim, dtype=np.float32)
    result[index] = 1
    return tuple(map(float, result))


@pytest.fixture
def review(tmp_path):
    paths = AppPaths.create(tmp_path)
    codec = TestOnlyCodec()
    profiles = SpeakerProfileService(paths, codec=codec)
    store = SpeakerReviewStore(paths, codec=codec)
    service = SpeakerReviewService(paths, profiles=profiles, store=store)
    job_id = 'a' * 32
    directory = paths.results / job_id
    directory.mkdir()
    json_path, text_path = directory / 'meeting.structured.json', directory / 'meeting.txt'
    turns = tuple(DiarizationTurn(i * 10, i * 10 + 9, f's{i}', 'mixed') for i in range(4))
    clusters = tuple(SpeakerCluster(f's{i}', 'mixed', 9, (vec(i * 2), vec(i * 2 + 1))) for i in range(4))
    structured = StructuredTranscript(tuple(TranscriptSegment(
        item.start, item.end, f'Phrase {i}.', (TranscriptWord(item.start, item.end, f'Phrase {i}.'),),
        speaker_key=item.cluster_id, speaker_name=f'Intervenant {i + 1}', speaker_source='mixed',
    ) for i, item in enumerate(turns)), 'fr')
    json_path.write_text(json.dumps(structured.to_dict()), encoding='utf-8')
    text_path.write_text(structured.speaker_text(), encoding='utf-8')
    audio = tmp_path / 'meeting.wav'
    audio.write_bytes(b'fixture-no-decoder-used')
    review_id = job_id + '-000-' + 'b' * 32
    service.create_review(
        review_id=review_id, job_id=job_id, file_index=0,
        diarization=DiarizationResult('test-model', turns, clusters),
        labels={f's{i}': f'Intervenant {i + 1}' for i in range(4)}, matches={},
        structured_filename=json_path.name, transcript_filename=text_path.name, source_audio=audio,
    )
    return SimpleNamespace(paths=paths, profiles=profiles, store=store, service=service,
                           review_id=review_id, json_path=json_path, text_path=text_path,
                           structured=structured, audio=audio)


def disk_state(review):
    return {path: persistence.read_optional(path) for path in (
        review.profiles.path, review.json_path, review.text_path, review.store._path(review.review_id),
    )}


def apply_state(review, state, remember=(), **kwargs):
    return review.service.apply(
        review.review_id, {g.group_id: g.name for g in state.ordered_groups()}, set(remember),
        groups=state.serialized_groups(), self_speaker_name='Romain', **kwargs,
    )


def test_merge_preview_undo_and_cancel_are_non_destructive(review):
    before = disk_state(review)
    state = SpeakerReviewState(review.service.load_review(review.review_id))
    state.merge(['s0', 's2'], 'Paul')
    assert len(state.groups) == 3
    assert [s.speaker_name for s in state.preview(review.structured).segments] == [
        'Paul', 'Intervenant 2', 'Paul', 'Intervenant 4']
    state.merge(['s0', 's3'], 'Paul')
    assert len(state.groups) == 2 and len(state.embeddings('s0')) == 6
    state.undo()
    assert len(state.groups) == 3 and state.groups['s0'].members == ('s0', 's2')
    state.undo()
    assert len(state.groups) == 4
    assert disk_state(review) == before


def test_merge_apply_all_segments_preserves_words_and_original_keys(review):
    payload = review.service.load_review(review.review_id)
    state = SpeakerReviewState(payload)
    state.merge(['s0', 's2', 's3'], 'Paul')
    apply_state(review, state, expected_revision=payload['_revision'],
                expected_transcript_revision=payload['_transcript_revision'])
    updated = review.service.transcript(review.review_id)
    assert [s.speaker_key for s in updated.segments] == ['s0', 's1', 's0', 's0']
    assert [s.speaker_original_key for s in updated.segments] == ['s0', 's1', 's2', 's3']
    assert [(s.start, s.end, s.words, s.text) for s in updated.segments] == [
        (s.start, s.end, s.words, s.text) for s in review.structured.segments]
    assert review.profiles.list_profiles() == ()
    assert review.text_path.read_text().strip() == updated.speaker_text()
    assert all(not c['embeddings'] for c in review.service.load_review(review.review_id)['clusters'])
    # Splitting remains possible even AFTER reopening a saved review.
    reopened = SpeakerReviewState(review.service.load_review(review.review_id))
    reopened.split('s0')
    reopened.groups['s2'].name = 'Julie'
    apply_state(review, reopened)
    assert review.service.transcript(review.review_id).segments[2].speaker_name == 'Julie'
    assert review.profiles.list_profiles() == ()


def test_all_merged_variants_enrolled_and_existing_profile_enriched(review):
    review.profiles.enroll('Paul', 'test-model', [vec(20), vec(21)])
    state = SpeakerReviewState(review.service.load_review(review.review_id))
    state.merge(['s0', 's2', 's3'], 'Paul')
    apply_state(review, state, ['s0'])
    profile, = review.profiles.list_profiles()
    assert len(profile.embeddings) == 8
    assert {int(np.argmax(item)) for item in profile.embeddings} == {0, 1, 4, 5, 6, 7, 20, 21}
    assert not profile.is_self


def test_multi_variant_recognition_survives_opposed_centroid_and_rejects_mixed_query(review):
    vectors = [vec(0), vec(1), vec(2), vec(3)]
    review.profiles.enroll('Paul', 'test-model', vectors)
    query = SpeakerCluster('a', 'mixed', 9, (vec(0), vec(1)))
    assert review.profiles.match_cluster_for_model(query, 'test-model').accepted
    review.profiles.enroll('Julie', 'test-model', [vec(8), vec(9)])
    mixed = SpeakerCluster('b', 'mixed', 9, (vec(0), vec(8)))
    assert not review.profiles.match_cluster_for_model(mixed, 'test-model').accepted
    assert not review.profiles.match_cluster_for_model(query, 'other-model').accepted
    assert not review.profiles.match_cluster_for_model(SpeakerCluster('c', 'mixed', 9, (vec(0, 32), vec(1, 32))),
                                                       'test-model').accepted
    assert not review.profiles.match_cluster_for_model(SpeakerCluster('c', 'mixed', float('nan'), (vec(0), vec(1))),
                                                       'test-model').accepted


def test_profile_capacity_preserves_diversity_instead_of_fifo(review):
    review.profiles.enroll('Paul', 'test-model', [vec(90), vec(91)])
    review.profiles.enroll('Paul', 'test-model', [vec(0)] * 80)
    profile, = review.profiles.list_profiles()
    assert len(profile.embeddings) == 64
    assert {0, 90, 91}.issubset({int(np.argmax(v)) for v in profile.embeddings})


@pytest.mark.parametrize('bad_name', ['', '   ', 'x' * 81, 'Paul\nJulie', 'Paul\t', '\x00Paul', 'A\u202eB'])
def test_invalid_names_rejected_without_writes(review, bad_name):
    before = disk_state(review)
    with pytest.raises((ValueError, TypeError)):
        review.service.apply(review.review_id, {'s0': bad_name}, set(), self_speaker_name='Romain')
    assert disk_state(review) == before


def test_missing_variant_or_duplicate_groups_rejected(review):
    state = SpeakerReviewState(review.service.load_review(review.review_id))
    groups = state.serialized_groups()
    with pytest.raises(ValueError):
        state.set_groups(groups[:-1])
    groups[0]['members'] += ['s1']
    with pytest.raises(ValueError):
        state.set_groups(groups)
    with pytest.raises(ValueError):
        state.merge(['s0', 'nonexistent'], 'Paul')


def test_homonyms_require_explicit_merge_before_enrollment(review):
    before = disk_state(review)
    with pytest.raises(ValueError, match='Fusionnez'):
        review.service.apply(review.review_id, {'s0': 'Paul', 's1': 'Paul'}, {'s0', 's1'}, self_speaker_name='Romain')
    assert disk_state(review) == before
    # Identical DISPLAY names alone do not merge speaker identity.
    transcript = StructuredTranscript((TranscriptSegment(0, 1, 'A', speaker_key='a', speaker_name='Paul'),
                                       TranscriptSegment(1, 2, 'B', speaker_key='b', speaker_name='Paul')))
    assert transcript.speaker_text().count('Paul') == 2


def test_batch_enrollment_validation_precedes_all_writes(review):
    review.profiles.enroll('Wrong model', 'old-model', [vec(10), vec(11)])
    before = disk_state(review)
    with pytest.raises(ValueError):
        review.service.apply(review.review_id, {'s0': 'Paul', 's1': 'Wrong model'}, {'s0', 's1'},
                             self_speaker_name='Romain')
    assert disk_state(review) == before


def test_dpapi_failure_does_not_write_transcript_or_profile(review, monkeypatch):
    before = disk_state(review)
    def fail(_data):
        raise OSError('DPAPI unavailable')
    monkeypatch.setattr(review.profiles.codec, 'protect', fail)
    with pytest.raises(OSError):
        review.service.apply(review.review_id, {'s0': 'Paul'}, {'s0'}, self_speaker_name='Romain')
    assert disk_state(review) == before


def test_disk_failure_rolls_back_profile_json_txt_and_review(review, monkeypatch):
    before = disk_state(review)
    original = persistence.atomic_bytes
    failed = False
    def fail_once(path, data):
        nonlocal failed
        if Path(path) == review.text_path and not failed:
            failed = True
            raise OSError('simulated disk full')
        original(path, data)
    monkeypatch.setattr(persistence, 'atomic_bytes', fail_once)
    with pytest.raises(OSError):
        review.service.apply(review.review_id, {'s0': 'Paul'}, {'s0'}, self_speaker_name='Romain')
    assert failed and disk_state(review) == before
    assert not list(review.service.transaction.root.iterdir())


def test_interrupted_transaction_is_recovered_on_next_open(review, monkeypatch):
    before = disk_state(review)
    original = persistence.atomic_bytes
    def interrupt(path, data):
        if Path(path) == review.text_path:
            raise KeyboardInterrupt('simulate terminated process')
        original(path, data)
    with monkeypatch.context() as patch:
        patch.setattr(persistence, 'atomic_bytes', interrupt)
        with pytest.raises(KeyboardInterrupt):
            review.service.apply(review.review_id, {'s0': 'Paul'}, {'s0'}, self_speaker_name='Romain')
    assert review.json_path.read_bytes() != before[review.json_path]
    review.service.transaction.recover()
    assert disk_state(review) == before


def test_recovery_refuses_to_overwrite_external_edits(review, monkeypatch):
    original = persistence.atomic_bytes
    def interrupt(path, data):
        if Path(path) == review.text_path:
            raise KeyboardInterrupt()
        original(path, data)
    with monkeypatch.context() as patch:
        patch.setattr(persistence, 'atomic_bytes', interrupt)
        with pytest.raises(KeyboardInterrupt):
            review.service.apply(review.review_id, {'s0': 'Paul'}, set(), self_speaker_name='Romain')
    review.json_path.write_bytes(b'manual edit')
    with pytest.raises(RuntimeError, match='suspendue'):
        review.service.transaction.recover()
    assert review.json_path.read_bytes() == b'manual edit'
    assert list(review.service.transaction.root.iterdir())


def test_stale_window_and_modified_transcript_are_rejected(review):
    old = review.service.load_review(review.review_id)
    review.service.apply(review.review_id, {'s0': 'Paul'}, set(), self_speaker_name='Romain')
    before = disk_state(review)
    with pytest.raises(RuntimeError, match='autre'):
        review.service.apply(review.review_id, {'s0': 'Julie'}, set(), self_speaker_name='Romain',
                             expected_revision=old['_revision'])
    assert disk_state(review) == before
    new = review.service.load_review(review.review_id)
    review.text_path.write_text('Manual correction', encoding='utf-8')
    with pytest.raises(RuntimeError, match='entre-temps'):
        review.service.apply(review.review_id, {}, set(), self_speaker_name='Romain',
                             expected_transcript_revision=new['_transcript_revision'])
    assert review.text_path.read_text() == 'Manual correction'


def test_expiration_clears_biometrics_but_keeps_renaming_available(review):
    payload = review.store.load(review.review_id)
    payload['created_at'] = (datetime.now(timezone.utc) - timedelta(days=8)).isoformat()
    review.store.save(review.review_id, payload)
    current = review.service.load_review(review.review_id)
    assert all(not item['embeddings'] for item in current['clusters'])
    before = disk_state(review)
    with pytest.raises(ValueError, match='empreinte'):
        review.service.apply(review.review_id, {'s0': 'Paul'}, {'s0'}, self_speaker_name='Romain')
    assert disk_state(review) == before
    review.service.apply(review.review_id, {'s0': 'Paul'}, set(), self_speaker_name='Romain')
    assert 'Paul' in review.text_path.read_text()


def test_corrupt_profile_is_not_silently_reset(review):
    review.profiles.path.write_bytes(b'not encrypted or invalid')
    before = disk_state(review)
    with pytest.raises(Exception):
        review.profiles.enroll('Paul', 'test-model', [vec(0), vec(1)])
    assert disk_state(review) == before


def test_concurrent_enrollments_do_not_lose_updates(review):
    def enroll(index):
        store = SpeakerProfileService(review.paths, codec=TestOnlyCodec())
        return store.enroll(f'Person {index}', 'test-model', [vec(index), vec(index)])
    with ThreadPoolExecutor(max_workers=6) as executor:
        list(executor.map(enroll, range(12)))
    assert len(review.profiles.list_profiles()) == 12


@pytest.mark.parametrize('job_id', ['../escape', 'a' * 31, 'A' * 32, 'a' * 32 + '/..'])
def test_review_paths_reject_traversal(review, job_id):
    payload = review.store.load(review.review_id)
    payload['job_id'] = job_id
    with pytest.raises(ValueError):
        review.service._paths(payload)


def test_symlink_transcript_escape_is_rejected(review, tmp_path):
    outside = tmp_path / 'outside.json'
    outside.write_text('{}')
    review.json_path.unlink()
    try:
        review.json_path.symlink_to(outside)
    except OSError:
        pytest.skip('symlink privilege unavailable on this Windows account')
    with pytest.raises(ValueError):
        review.service.load_review(review.review_id)


def test_audio_multiple_clips_and_variants_survive_merge(review):
    payload = review.service.load_review(review.review_id)
    clips = excerpts_for(payload, ['s0', 's2', 's3'])
    assert {clip.cluster_id for clip in clips} == {'s0', 's2', 's3'}
    assert len(clips) == 6 and all(0 < c.end - c.start <= 6 for c in clips)
    assert all(c.path == review.audio for c in clips)
    for speaker in {c.cluster_id for c in clips}:
        mine = sorted((c.start, c.end) for c in clips if c.cluster_id == speaker)
        assert all(a[1] <= b[0] for a, b in zip(mine, mine[1:]))


def test_audio_source_offset_and_clean_overlap_selection(review):
    payload = review.service.load_review(review.review_id)
    payload['clusters'] = [{'cluster_id': 'self', 'source': 'self'}, {'cluster_id': 'remote', 'source': 'system'}]
    payload['turns'] = [{'cluster_id': 'self', 'source': 'self', 'start': 3, 'end': 12},
                        {'cluster_id': 'remote', 'source': 'system', 'start': 3, 'end': 12}]
    payload['audio_sources'] = {'self': {'path': str(review.audio), 'offset': 3},
                                'mixed': {'path': str(review.audio), 'offset': 0}}
    clip = excerpts_for(payload, ['self'])[0]
    assert clip.start == 0 and clip.global_start == 3 and clip.isolated and not clip.overlap_possible
    mixed = excerpts_for(payload, ['self'], source_override=review.audio)[0]
    assert mixed.start == 3 and not mixed.isolated and mixed.overlap_possible


@pytest.mark.parametrize('start,end', [(float('nan'), 4), (-1, 4), (4, 4), (3, 10), (1, float('inf'))])
def test_audio_invalid_bounds_rejected(tmp_path, start, end):
    with pytest.raises(ValueError):
        AudioExcerpt(tmp_path / 'a.wav', start, end, start, 'a', False)


def test_source_choice_persists_only_on_apply_and_checks_existence(review, tmp_path):
    chosen = tmp_path / 'chosen.wav'
    chosen.write_bytes(b'audio')
    payload = review.service.load_review(review.review_id)
    before = disk_state(review)
    assert excerpts_for(payload, ['s0'], source_override=chosen)[0].path == chosen
    assert disk_state(review) == before
    with pytest.raises(FileNotFoundError):
        review.service.apply(review.review_id, {}, set(), self_speaker_name='Romain', source_override=tmp_path / 'absent')
    assert disk_state(review) == before
    review.service.apply(review.review_id, {}, set(), self_speaker_name='Romain', source_override=chosen)
    assert review.service.load_review(review.review_id)['audio_sources']['mixed']['path'] == str(chosen)


@pytest.mark.parametrize('expected', [-1, 33, True, '4', None, 4.5])
def test_invalid_expected_count(expected):
    with pytest.raises(ValueError):
        validate_diarization_options(expected, 0.60)


@pytest.mark.parametrize('threshold', [float('nan'), float('inf'), 0.19, 0.91, True, '0.6'])
def test_invalid_clustering_threshold(threshold):
    with pytest.raises(ValueError):
        validate_diarization_options(0, threshold)


def test_embedding_windows_are_exclusive_nonoverlapping_and_independent():
    turns = [DiarizationTurn(0, 30, 'a'), DiarizationTurn(9, 18, 'b'), DiarizationTurn(0, 8, 'a')]
    windows = worker._embedding_windows(turns, 'a')
    assert 2 <= len(windows) <= 6
    assert all(end <= 8.97 or start >= 18.03 for start, end in windows)
    assert all(a[1] <= b[0] for a, b in zip(windows, windows[1:]))
    assert all(1.5 <= end - start <= 8 for start, end in windows)


def test_one_short_sample_cannot_fake_multiple_query_embeddings(monkeypatch):
    monkeypatch.setattr(worker, '_embedding', lambda *args: vec(0))
    samples = np.ones(16000 * 10, np.float32) * 0.1
    values = worker._cluster_embeddings(None, samples, 16000, [DiarizationTurn(0, 2, 'a')], 'a')
    assert len(values) == 1
    short = [DiarizationTurn(i, i + 0.6, 'a') for i in range(4)]
    values = worker._cluster_embeddings(None, samples, 16000, short, 'a')
    assert len(values) == 1
    assert worker._cluster_embeddings(None, np.zeros(160000, np.float32), 16000, short, 'a') == ()


@pytest.mark.parametrize('mic_speaks,total,remote_count', [(True, 4, 3), (False, 4, 4), (True, 0, -1), (False, 0, -1)])
def test_worker_count_subtracts_microphone_only_when_it_speaks(tmp_path, monkeypatch, mic_speaks, total, remote_count):
    model = tmp_path / 'model.onnx'
    model.write_bytes(b'mock model')
    calls = []
    def process(path, **kwargs):
        calls.append(kwargs)
        source = kwargs['source']
        if source == 'self' and not mic_speaks:
            return [], []
        return [DiarizationTurn(0, 5, source, source)], [SpeakerCluster(source, source, 5, ())]
    monkeypatch.setattr(worker, '_process_one', process)
    result = worker.process_request(dict(schema_version=1, expected_speakers=total, clustering_threshold=.65,
                                        segmentation_model=str(model), embedding_model=str(model),
                                        microphone_source='mic.wav', system_source='system.wav'))
    assert result.model_set_id == MODEL_SET_ID
    assert calls[0]['num_speakers'] == 1 and calls[1]['num_speakers'] == remote_count
    assert all(item['clustering_threshold'] == .65 for item in calls)


def test_worker_expected_one_rejects_two_active_channels(tmp_path, monkeypatch):
    model = tmp_path / 'model.onnx'
    model.touch()
    monkeypatch.setattr(worker, '_process_one', lambda path, **kwargs: (
        [DiarizationTurn(0, 5, kwargs['source'], kwargs['source'])], []))
    with pytest.raises(ValueError, match='Une voix'):
        worker.process_request(dict(schema_version=1, expected_speakers=1, segmentation_model=str(model),
                                    embedding_model=str(model), microphone_source='mic', system_source='sys'))


def test_silence_does_not_invoke_neural_clustering(tmp_path, monkeypatch):
    monkeypatch.setattr(worker, '_decode_audio', lambda path: (np.zeros(16000, np.float32), 16000))
    monkeypatch.setattr(worker, '_make_diarizer', lambda *args: pytest.fail('silence must skip model'))
    assert worker._process_one(tmp_path, source='mixed', prefix='x', offset=0,
                               segmentation_model=tmp_path, embedding_model=tmp_path, num_speakers=-1) == ([], [])


def test_alignment_does_not_guess_winner_during_mixed_overlap():
    turns = (DiarizationTurn(0, 8, 'self', 'self'), DiarizationTurn(0, 8, 'remote', 'system'))
    assert _best_turn(1, 2, turns) is None
    assert _best_turn(1, 2, turns, 'self').cluster_id == 'self'
    assert _best_turn(1, 2, turns, 'system').cluster_id == 'remote'
    assert _best_turn(20, 21, turns) is None


def test_native_transcripts_keep_source_and_offsets_through_alignment():
    local = StructuredTranscript((TranscriptSegment(1, 2, 'Oui', (TranscriptWord(1, 2, 'Oui'),)),), 'fr')
    mic, system = shift_source(local, 2, 'self'), shift_source(local, 2, 'system')
    merged = combine_sources([mic, system])
    turns = (DiarizationTurn(3, 4, 'self', 'self'), DiarizationTurn(3, 4, 'remote', 'system'))
    aligned = align_transcript(merged, DiarizationResult('m', turns, ()), {'self': 'Moi', 'remote': 'Paul'}, {})
    assert [s.speaker_name for s in aligned.segments] == ['Moi', 'Paul']
    assert all(s.start == 3 and s.words[0].start == 3 for s in aligned.segments)
    assert local.segments[0].start == 1


def test_missing_native_track_falls_back_to_full_mixed_source(review):
    recording_id = 'c' * 32
    directory = review.paths.results / 'recording_sources' / recording_id
    directory.mkdir(parents=True)
    (directory / 'microphone.wav').write_bytes(b'mic')
    payload = dict(schema_version=1, recording_id=recording_id, microphone_file='microphone.wav',
                   system_file='system.wav', microphone_offset=0, system_offset=0, sample_rate=48000)
    manifest = directory / 'recording.json'
    manifest.write_text(json.dumps(payload))
    mixed = review.paths.results / f'native_recording_{recording_id}.wav'
    assert find_recording_sources(review.paths, mixed) is None
    (directory / 'system.wav').write_bytes(b'system')
    assert find_recording_sources(review.paths, mixed) is not None
    payload['system_offset'] = float('nan')
    manifest.write_text(json.dumps(payload))
    assert find_recording_sources(review.paths, mixed) is None


def test_orchestrator_passes_options_reports_worker_error_and_cleans_embeddings(review, monkeypatch):
    from transcripteur_whisper.services import diarization_service as module
    segmentation, embedding = review.paths.models / 's.onnx', review.paths.models / 'e.onnx'
    segmentation.touch()
    embedding.touch()
    models = SimpleNamespace(ensure=lambda *args: (segmentation, embedding))
    observed = {}
    class Process:
        returncode = 1
        def __init__(self, command, **kwargs):
            request, response = Path(command[-2]), Path(command[-1])
            observed.update(json.loads(request.read_text()))
            response.write_text(json.dumps({'ok': False, 'error': 'Expected speaker count inconsistent'}))
        def poll(self):
            return 1
    monkeypatch.setattr(module.subprocess, 'Popen', Process)
    service = DiarizationService(review.paths, models=models)
    with pytest.raises(DiarizationError, match='inconsistent'):
        service.run(review.audio, job_id='a' * 32, file_index=0, expected_speakers=4,
                    clustering_threshold=.65, cancel_check=lambda: None)
    assert observed['expected_speakers'] == 4 and observed['clustering_threshold'] == .65
    assert not list(review.paths.temp.rglob('response.json'))
    assert not list(review.paths.temp.rglob('request.json'))


def test_delete_all_erases_voiceprints_without_destroying_review_metadata(review):
    review.profiles.enroll('Paul', 'test-model', [vec(0), vec(1)])
    review.profiles.delete_all()
    assert review.profiles.list_profiles() == ()
    payload = review.service.load_review(review.review_id)
    assert len(payload['clusters']) == 4
    assert all(not cluster['embeddings'] for cluster in payload['clusters'])
    review.service.apply(review.review_id, {'s0': 'Paul'}, set(), self_speaker_name='Romain')
    assert 'Paul' in review.text_path.read_text()


def test_deleted_isolated_audio_falls_back_to_existing_mixed_audio(review):
    payload = review.service.load_review(review.review_id)
    payload['clusters'][0]['source'] = 'self'
    payload['audio_sources']['self'] = {'path': str(review.paths.temp / 'missing.wav'), 'offset': 3}
    clip = excerpts_for(payload, ['s0'])[0]
    assert clip.path == review.audio and not clip.isolated


def test_legacy_review_can_use_original_audio_without_preexisting_audio_metadata(review):
    path = review.store._path(review.review_id)
    payload = review.store.load(review.review_id)
    payload.pop('turns')
    payload.pop('audio_sources')
    payload['schema_version'] = 1
    path.write_bytes(review.store.codec.protect(json.dumps(payload).encode()))
    restored = review.service.load_review(review.review_id)
    assert len(restored['turns']) == 4
    assert excerpts_for(restored, ['s0']) == []
    assert excerpts_for(restored, ['s0'], source_override=review.audio)


def test_windows_dpapi_real_roundtrip():
    import sys
    if sys.platform != 'win32':
        pytest.skip('Real Windows DPAPI cannot execute on Linux; release gate requires this test on Windows.')
    from transcripteur_whisper.services.dpapi import WindowsDpapiCodec
    codec = WindowsDpapiCodec()
    clear = b'synthetic-speaker-profile-test-not-real-voice'
    encrypted = codec.protect(clear)
    assert encrypted != clear and clear not in encrypted
    assert codec.unprotect(encrypted) == clear
    with pytest.raises(RuntimeError):
        codec.unprotect(b'not-a-dpapi-ciphertext')


def test_cross_process_profile_enrollment_is_serialized(review):
    import os
    import subprocess
    import sys
    script = r'''
import sys
from pathlib import Path
from transcripteur_whisper.core.paths import AppPaths
from transcripteur_whisper.services.speaker_profile_service import SpeakerProfileService
class Codec:
    def protect(self, value): return b"TEST-ONLY:" + value
    def unprotect(self, value):
        assert value.startswith(b"TEST-ONLY:")
        return value[10:]
paths = AppPaths.create(Path(sys.argv[1]))
v = [1.0] + [0.0] * 31
SpeakerProfileService(paths, codec=Codec()).enroll(sys.argv[2], "test-model", [v, v])
'''
    import transcripteur_whisper.services.speaker_profile_service as module
    environment = os.environ.copy()
    environment['PYTHONPATH'] = str(Path(module.__file__).parents[2])
    children = [subprocess.Popen([sys.executable, '-c', script, str(review.paths.root), f'Child {i}'],
                                 env=environment, stdout=subprocess.PIPE, stderr=subprocess.PIPE) for i in range(5)]
    for child in children:
        stdout, stderr = child.communicate(timeout=30)
        assert child.returncode == 0, (stdout, stderr)
    assert len(review.profiles.list_profiles()) == 5
