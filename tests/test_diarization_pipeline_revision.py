"""Actual orchestration code, using explicit fake neural/API/media dependencies."""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

from transcripteur_whisper.core.paths import AppPaths
from transcripteur_whisper.models.jobs import JobCancelled, ProgressReporter
from transcripteur_whisper.models.speaker import DiarizationResult, DiarizationTurn, SpeakerCluster
from transcripteur_whisper.models.transcription import TranscriptionOptions
from transcripteur_whisper.services import speaker_review_service
from transcripteur_whisper.services.speaker_profile_service import SpeakerProfileService


class TestOnlyCodec:
    __test__ = False
    def protect(self, data):
        return b'TEST:' + data
    def unprotect(self, data):
        assert data.startswith(b'TEST:')
        return data[5:]


@pytest.fixture
def pipeline(tmp_path, monkeypatch):
    # Only expensive/external boundaries are replaced, not transcript/review logic.
    for name, attributes in {
        'integrations.openai_service': {'make_openai_client': lambda *args: pytest.fail('No external fallback allowed')},
        'services.document_service': {'generate_document': lambda *args, **kw: 'Document mock'},
        'services.media_service': {'MediaService': object},
        'services.model_service': {'ModelService': object},
    }.items():
        full = 'transcripteur_whisper.' + name
        module = ModuleType(full)
        module.__dict__.update(attributes)
        monkeypatch.setitem(sys.modules, full, module)
    path = Path(speaker_review_service.__file__).with_name('transcription_service.py')
    spec = importlib.util.spec_from_file_location('transcripteur_whisper.services._pipeline_revision_test', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    paths = AppPaths.create(tmp_path)
    profiles = SpeakerProfileService(paths, codec=TestOnlyCodec())
    review_store = speaker_review_service.SpeakerReviewStore(paths, codec=TestOnlyCodec())
    review_service = speaker_review_service.SpeakerReviewService(paths, profiles=profiles, store=review_store)
    monkeypatch.setattr(module, 'SpeakerProfileService', lambda *args: profiles)
    monkeypatch.setattr(module, 'SpeakerReviewService', lambda *args, **kw: review_service)
    media_calls = []
    class Media:
        def prepare_local_media(self, job_id, index, source, stem, check):
            media_calls.append(source)
            return source, None
    class Model:
        calls = []
        def transcribe(self, path, **options):
            self.calls.append((str(path), options))
            text = 'Micro' if 'microphone' in str(path) else 'System'
            word = SimpleNamespace(start=1, end=2, word=text)
            segment = SimpleNamespace(start=1, end=2, text=text, words=[word])
            return iter([segment]), SimpleNamespace(duration=3, language='fr')
    diarization_calls = []
    class Diarizer:
        def run(self, source, **kwargs):
            diarization_calls.append(kwargs)
            return DiarizationResult('test', (
                DiarizationTurn(1, 3, 'self', 'self'),
                DiarizationTurn(1, 5, 'remote', 'system'),
            ), (SpeakerCluster('self', 'self', 2, ()), SpeakerCluster('remote', 'system', 3, ())))
    service = module.TranscriptionService(paths, media=Media(), models=object(), diarization=Diarizer())
    source = tmp_path / 'source.wav'
    source.write_bytes(b'test fake whisper input')
    metadata = dict(path=str(source), name=source.name, output_stem='test')
    job = dict(id='a' * 32, files=[metadata], output_name='Test')
    logs = []
    reporter = ProgressReporter(lambda: None, logs.append,
                                lambda index, **kwargs: metadata.update(kwargs), lambda **kw: job.update(kw))
    return SimpleNamespace(paths=paths, module=module, service=service, profiles=profiles,
                           reviews=review_service, source=source, metadata=metadata, job=job,
                           model=Model(), reporter=reporter, logs=logs, media_calls=media_calls,
                           diarization_calls=diarization_calls)


def run(pipeline, options):
    pipeline.service._run_file(pipeline.job, 0, pipeline.metadata, options, pipeline.model, None, '', pipeline.reporter)


def test_disabled_diarization_preserves_legacy_path(pipeline):
    run(pipeline, TranscriptionOptions())
    assert pipeline.metadata['status'] == 'done'
    assert pipeline.diarization_calls == []
    assert 'word_timestamps' not in pipeline.model.calls[0][1]
    assert 'speaker_review_id' not in pipeline.metadata
    assert Path(pipeline.metadata['transcription_path']).read_text().strip() == 'System'


def test_native_pipeline_preserves_channels_creates_review_and_exportable_names(pipeline):
    recording_id = 'b' * 32
    directory = pipeline.paths.results / 'recording_sources' / recording_id
    directory.mkdir(parents=True)
    for name in ('microphone.wav', 'system.wav'):
        (directory / name).write_bytes(b'mocked audio')
    (directory / 'recording.json').write_text(json.dumps(dict(
        schema_version=1, recording_id=recording_id, microphone_file='microphone.wav', system_file='system.wav',
        microphone_offset=0, system_offset=1, sample_rate=48000,
    )))
    pipeline.source = pipeline.paths.results / f'native_recording_{recording_id}.wav'
    pipeline.source.write_bytes(b'mixed audio')
    pipeline.metadata.update(path=str(pipeline.source), name=pipeline.source.name)
    run(pipeline, TranscriptionOptions(diarization_enabled=True, self_speaker_name='Romain',
                                       expected_speakers=4, clustering_threshold=.65))
    assert pipeline.metadata['status'] == 'done', pipeline.metadata.get('error')
    assert len(pipeline.model.calls) == 2 and pipeline.media_calls == []
    assert all(call[1]['word_timestamps'] for call in pipeline.model.calls)
    assert pipeline.diarization_calls[0]['expected_speakers'] == 4
    assert pipeline.diarization_calls[0]['clustering_threshold'] == .65
    review_id = pipeline.metadata['speaker_review_id']
    structured = pipeline.reviews.transcript(review_id)
    assert [s.speaker_name for s in structured.segments] == ['Romain', 'Intervenant 1']
    assert [s.start for s in structured.segments] == [1, 2]
    payload = pipeline.reviews.load_review(review_id)
    assert payload['audio_sources']['system']['offset'] == 1
    assert pipeline.profiles.list_profiles() == ()
    pipeline.reviews.apply(review_id, {'remote': 'Paul'}, set(), self_speaker_name='Romain')
    assert 'Paul' in Path(pipeline.metadata['transcription_path']).read_text()


def test_diarization_error_keeps_plain_partial_transcript(pipeline):
    def fail(*args, **kwargs):
        raise RuntimeError('worker failure')
    pipeline.service.diarization.run = fail
    run(pipeline, TranscriptionOptions(diarization_enabled=True))
    assert pipeline.metadata['status'] == 'partial'
    assert pipeline.metadata['error'] == 'worker failure'
    assert Path(pipeline.metadata['transcription_path']).read_text().strip() == 'System'
    assert not pipeline.metadata.get('speaker_review_id')


def test_cancellation_is_not_swallowed_or_faked_as_success(pipeline):
    pipeline.reporter = ProgressReporter(lambda: (_ for _ in ()).throw(JobCancelled()), pipeline.logs.append,
                                        lambda i, **kw: pipeline.metadata.update(kw), lambda **kw: None)
    with pytest.raises(JobCancelled):
        run(pipeline, TranscriptionOptions(diarization_enabled=True))
    assert pipeline.metadata['status'] == 'cancelled'
    assert not pipeline.diarization_calls
