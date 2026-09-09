"""Batch transcription orchestration, optional diarization and persistent partial results."""
from __future__ import annotations

import json
import shutil
import uuid
from pathlib import Path

from ..integrations import openai_service, whisper_local
from ..models.jobs import JobCancelled, ProgressReporter
from ..models.transcript import StructuredTranscript
from ..models.transcription import TranscriptionOptions
from .diarization_service import DiarizationService
from .document_service import generate_document
from .files import output_filename, write_text_atomic
from .media_service import MediaService
from .model_service import ModelService
from .recording_sources import find_recording_sources
from .speaker_profile_service import SpeakerProfileService
from .speaker_review_service import SpeakerReviewService
from .transcript_alignment import align_transcript, build_labels
from .transcript_sources import combine_sources, shift_source


class TranscriptionService:
    def __init__(self, paths, *, media=None, models=None, diarization=None):
        self.paths = paths
        self.media = media or MediaService(paths)
        self.models = models or ModelService(paths)
        self.diarization = diarization or DiarizationService(paths)

    def run(
        self,
        job: dict,
        options: TranscriptionOptions,
        api_key: str,
        reporter: ProgressReporter,
    ) -> None:
        model = None
        client = None
        try:
            if not options.use_api:
                model_path = self.models.download(
                    options.model,
                    reporter.cancel_check,
                    lambda value: reporter.job(model_download_progress=value),
                    reporter.log,
                )
                reporter.cancel_check()
                reporter.log("Chargement du modèle local (CPU int8)…")
                model = whisper_local.load_model(model_path)
            # Existing OpenAI client remains the document-generation client. The
            # diarized transcription endpoint uses isolated direct HTTP code so
            # it does not require a risky SDK upgrade.
            if (options.use_api and not options.diarization_enabled) or options.output_type != "transcription":
                client = openai_service.make_openai_client(api_key)
            for index, metadata in enumerate(job["files"]):
                reporter.cancel_check()
                self._run_file(job, index, metadata, options, model, client, api_key, reporter)
        finally:
            if client is not None:
                try:
                    client.close()
                except Exception:
                    pass
            del model

    def _run_file(self, job, index, metadata, options, model, client, api_key, reporter):
        cleanup = None
        result_dir = self.paths.results / job["id"]
        transcript_path = result_dir / output_filename(job, "transcription", index)
        structured_path = transcript_path.with_suffix(".structured.json")
        document_requested = options.output_type != "transcription"
        # Transcription uses 70%, diarization 20%, optional document 10%.
        transcription_weight = 0.70 if options.diarization_enabled else (0.85 if document_requested else 1.0)
        diarization_weight = 0.20 if options.diarization_enabled else 0.0
        document_base = transcription_weight + diarization_weight

        def partial(progress: float, text: str, **changes) -> None:
            if text:
                write_text_atomic(transcript_path, text)
                changes.update(
                    out_path=str(transcript_path),
                    transcription_path=str(transcript_path),
                    transcription_available=True,
                )
            overall = progress * transcription_weight
            reporter.file(index, progress=overall, **changes)
            reporter.job(progress=(index + overall) / len(job["files"]))

        reporter.file(index, status="running", stage="conversion", error=None)
        reporter.log(f"Traitement : {metadata['name']}")
        structured: StructuredTranscript | None = None
        text = ""
        try:
            source = Path(metadata["path"])
            if options.use_api:
                chunks, cleanup, duration = self.media.prepare_api_chunks(
                    job["id"], index, source, metadata["output_stem"], reporter.cancel_check
                )
                reporter.file(index, stage="transcription_api", duration=duration, segment_count=len(chunks))
                if options.diarization_enabled:
                    reporter.log(
                        "Transcription OpenAI avec gpt-4o-transcribe-diarize ; "
                        "l'identité finale des voix reste déterminée localement."
                    )
                    offsets: list[tuple[Path, float]] = []
                    offset = 0.0
                    for chunk in chunks:
                        offsets.append((chunk, offset))
                        chunk_duration = self.media.probe(chunk)
                        if chunk_duration is None:
                            raise RuntimeError(f"Durée du segment {chunk.name} impossible à déterminer.")
                        offset += float(chunk_duration)

                    def on_chunk(position: int, partial_text: str) -> None:
                        partial(position / max(1, len(chunks)), partial_text, segment_index=position)
                        reporter.log(f"Segment diarizé OpenAI {position}/{len(chunks)} reçu.")

                    structured = openai_service.transcribe_diarized_chunks(
                        api_key,
                        offsets,
                        options.language,
                        reporter.cancel_check,
                        on_chunk,
                    )
                    text = structured.plain_text()
                else:
                    reporter.log(f"Transcription OpenAI : {len(chunks)} segment(s) de dix minutes au maximum.")

                    def on_chunk(position: int, partial_text: str) -> None:
                        partial(position / max(1, len(chunks)), partial_text, segment_index=position)
                        reporter.log(f"Segment {position}/{len(chunks)} reçu.")

                    text = openai_service.transcribe_chunks(
                        client,
                        chunks,
                        options.model,
                        options.language,
                        reporter.cancel_check,
                        on_chunk,
                    )
            else:
                native = find_recording_sources(self.paths, source) if options.diarization_enabled else None
                reporter.file(index, stage="transcription_local")
                reporter.log("VAD Silero - beam_size=5")
                if native is not None:
                    tracks = [(path, offset, kind) for path, offset, kind in (
                        (native.microphone_path, native.microphone_offset, "self"),
                        (native.system_path, native.system_offset, "system"),
                    ) if path is not None]
                    transcripts = []
                    for track_index, (path, offset, kind) in enumerate(tracks):
                        reporter.cancel_check()
                        reporter.log(f"Transcription de la piste {kind} sans melange des canaux.")

                        def track_partial(value, content, position=track_index):
                            previous = combine_sources(transcripts).plain_text()
                            partial((position + value) / len(tracks),
                                    "\n".join(part for part in (previous, content) if part))

                        track = whisper_local.transcribe_structured(
                            model, path, options.language, reporter.cancel_check, track_partial
                        )
                        transcripts.append(shift_source(track, offset, kind))
                    structured = combine_sources(transcripts)
                    text = structured.plain_text()
                else:
                    processing, cleanup = self.media.prepare_local_media(
                        job["id"], index, source, metadata["output_stem"], reporter.cancel_check
                    )
                    if options.diarization_enabled:
                        structured = whisper_local.transcribe_structured(
                            model, processing, options.language, reporter.cancel_check, partial
                        )
                        text = structured.plain_text()
                    else:
                        text = whisper_local.transcribe(
                            model, processing, options.language, reporter.cancel_check, partial
                        )

            write_text_atomic(transcript_path, text)
            reporter.file(
                index,
                out_path=str(transcript_path),
                transcription_path=str(transcript_path),
                transcription_available=True,
                stage="recomposition",
            )
            reporter.cancel_check()

            if options.diarization_enabled:
                if structured is None:
                    raise RuntimeError("La transcription horodatée nécessaire à la diarisation est absente.")
                reporter.file(index, stage="diarization")
                reporter.log("Diarisation locale et comparaison avec les profils vocaux confirmés…")

                def diarization_progress(value: float) -> None:
                    overall = transcription_weight + value * diarization_weight
                    reporter.file(index, progress=overall)
                    reporter.job(progress=(index + overall) / len(job["files"]))

                diarization = self.diarization.run(
                    source,
                    job_id=job["id"],
                    file_index=index,
                    expected_speakers=options.expected_speakers,
                    clustering_threshold=options.clustering_threshold,
                    cancel_check=reporter.cancel_check,
                    progress=diarization_progress,
                    log=reporter.log,
                )
                reporter.cancel_check()
                profiles = SpeakerProfileService(self.paths)
                labels, matches = build_labels(diarization, profiles, options.self_speaker_name)
                aligned = align_transcript(structured, diarization, labels, matches)
                text = aligned.speaker_text()
                write_text_atomic(transcript_path, text)
                write_text_atomic(
                    structured_path,
                    json.dumps(aligned.to_dict(), ensure_ascii=False, indent=2),
                )
                review_id = f"{job['id']}-{index:03d}-{uuid.uuid4().hex}"
                review_service = SpeakerReviewService(self.paths, profiles=profiles)
                review_service.create_review(
                    review_id=review_id,
                    job_id=job["id"],
                    file_index=index,
                    diarization=diarization,
                    labels=labels,
                    matches=matches,
                    structured_filename=structured_path.name,
                    transcript_filename=transcript_path.name,
                    source_audio=source,
                )
                reporter.file(
                    index,
                    stage="speaker_review_ready",
                    structured_transcript_path=str(structured_path),
                    speaker_review_id=review_id,
                    speaker_review_available=True,
                    progress=document_base,
                )
                reporter.log(
                    "Locuteurs séparés. Les identifications incertaines restent « Intervenant N » "
                    "jusqu'à confirmation humaine."
                )

            reporter.cancel_check()
            if document_requested and text:
                if client is None:
                    client = openai_service.make_openai_client(api_key)
                reporter.file(index, stage="document")
                reporter.log(f"Génération du document : {options.output_type}")
                processed = generate_document(
                    client,
                    options.output_type,
                    text,
                    cancel_check=reporter.cancel_check,
                    log=reporter.log,
                )
                document = result_dir / output_filename(job, options.output_type, index)
                write_text_atomic(document, processed)
                reporter.file(
                    index,
                    out_path=str(document),
                    document_path=str(document),
                    document_available=True,
                )
            reporter.file(index, status="done", stage="done", progress=1.0)
            reporter.job(progress=(index + 1) / len(job["files"]))
        except JobCancelled:
            reporter.file(index, status="cancelled", stage="cancelled")
            raise
        except Exception as exc:
            available = transcript_path.is_file() and transcript_path.stat().st_size > 0
            status = "partial" if available else "error"
            reporter.file(index, status=status, stage=status, error=str(exc))
            reporter.log(
                f"{'Résultat partiel conservé' if available else 'Échec'} pour {metadata['name']} : {exc}"
            )
        finally:
            if cleanup is not None:
                shutil.rmtree(cleanup, ignore_errors=True)
