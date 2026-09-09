"""Desktop replacements for HTTP boundary and job lifecycle regressions."""
from __future__ import annotations

import json
import tempfile
import threading
import time
import unittest
import zipfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from transcripteur_whisper.core.paths import AppPaths
from transcripteur_whisper.integrations import openai_service, whisper_local
from transcripteur_whisper.models.jobs import JobCancelled, JobError, ValidationError
from transcripteur_whisper.models.transcription import TranscriptionOptions
from transcripteur_whisper.services import job_service, model_service
from transcripteur_whisper.services.files import write_text_atomic
from transcripteur_whisper.services.job_service import JobService
from transcripteur_whisper.services.transcription_service import TranscriptionService


class JobServiceTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.paths = AppPaths.create(Path(self.directory.name))
        self.source = Path(self.directory.name) / "source.wav"
        self.source.write_bytes(b"audio")
        self.engine = Mock()
        self.service = JobService(self.paths, transcription=self.engine)
        self.addCleanup(self.service.shutdown, True)

    def wait_job(self, job_id):
        deadline = time.monotonic() + 3
        while time.monotonic() < deadline:
            snapshot = self.service.snapshot(job_id)
            if snapshot["status"] in {"done", "partial", "error", "cancelled"}:
                return snapshot
            time.sleep(0.005)
        self.fail("Job did not finish")

    def complete(self, job, options, api_key, reporter):
        for index, _metadata in enumerate(job["files"]):
            result = self.paths.results / job["id"] / f"transcription_{index}.txt"
            write_text_atomic(result, f"Texte {index}")
            reporter.file(index, status="done", stage="done", progress=1.0,
                          out_path=str(result), transcription_path=str(result), transcription_available=True)

    def test_invalid_extension_rejected_without_creating_job(self):
        executable = self.source.with_suffix(".exe")
        executable.write_bytes(b"not media")
        with self.assertRaisesRegex(ValidationError, "Format non pris en charge"):
            self.service.submit([executable], TranscriptionOptions())
        self.assertEqual(self.service.history(), [])

    def test_file_and_total_size_limits_apply_before_queue(self):
        with patch.object(job_service, "MAX_UPLOAD_BYTES", 2):
            with self.assertRaisesRegex(ValidationError, "taille"):
                self.service.submit([self.source], TranscriptionOptions())
        with patch.object(job_service, "MAX_JOB_UPLOAD_BYTES", 5):
            with self.assertRaisesRegex(ValidationError, "taille"):
                self.service.submit([self.source, self.source], TranscriptionOptions())
        self.assertEqual(self.service.history(), [])

    def test_options_validate_model_language_mode_and_key(self):
        cases = [TranscriptionOptions(model="unknown"), TranscriptionOptions(language="unknown"),
                 TranscriptionOptions(mode="unknown"), TranscriptionOptions(output_type="unknown"),
                 TranscriptionOptions(mode="api", model="whisper-1")]
        for options in cases:
            with self.subTest(options=options), self.assertRaises(ValidationError):
                self.service.submit([self.source], options)

    def test_unknown_job_cancel_fails_clearly(self):
        with self.assertRaisesRegex(JobError, "introuvable"):
            self.service.cancel("not-a-job")

    def test_snapshot_is_detached_and_hides_source_and_cancellation_state(self):
        self.engine.run.side_effect = self.complete
        job_id = self.service.submit([self.source], TranscriptionOptions())
        snapshot = self.wait_job(job_id)
        self.assertNotIn("cancel_requested", snapshot)
        self.assertNotIn("path", snapshot["files"][0])
        self.assertNotIn("output_stem", snapshot["files"][0])
        snapshot["files"][0]["name"] = "changed"
        self.assertEqual(self.service.snapshot(job_id)["files"][0]["name"], self.source.name)
        self.assertTrue(self.source.exists())

    def test_results_persist_across_restart_and_export_ordered_txt_and_zip(self):
        self.engine.run.side_effect = self.complete
        job_id = self.service.submit([self.source, self.source], TranscriptionOptions(output_name="Réunion"))
        self.assertEqual(self.wait_job(job_id)["status"], "done")
        self.service.shutdown(wait=True)
        restarted = JobService(self.paths, transcription=Mock())
        self.addCleanup(restarted.shutdown, True)
        self.assertEqual(restarted.history()[0]["id"], job_id)
        text = restarted.export(job_id, Path(self.directory.name) / "export.txt").read_text(encoding="utf-8")
        self.assertLess(text.index("Texte 0"), text.index("Texte 1"))
        archive = restarted.export(job_id, Path(self.directory.name) / "export.zip")
        with zipfile.ZipFile(archive) as z:
            self.assertEqual(z.namelist(), ["transcription_0.txt", "transcription_1.txt"])
        manifest = json.loads((self.paths.results / job_id / "job.json").read_text(encoding="utf-8"))
        self.assertNotIn("path", manifest["files"][0])

    def test_pending_cancel_and_shutdown_do_not_wait_for_blocked_worker(self):
        started, release = threading.Event(), threading.Event()

        def blocking(*_args):
            started.set()
            release.wait(5)

        self.engine.run.side_effect = blocking
        self.addCleanup(release.set)
        running = self.service.submit([self.source], TranscriptionOptions())
        self.assertTrue(started.wait(1))
        pending = self.service.submit([self.source], TranscriptionOptions())
        self.assertEqual(self.service.cancel(pending)["status"], "cancelled")
        before = time.monotonic()
        self.service.shutdown(wait=False)
        self.assertLess(time.monotonic() - before, 1)
        self.assertEqual(self.service.snapshot(running)["status"], "cancelling")
        release.set()
        self.assertEqual(self.wait_job(running)["status"], "cancelled")

    def test_manifest_write_failure_does_not_retain_job_capacity(self):
        self.engine.run.side_effect = self.complete
        with patch.object(job_service, "MAX_QUEUED_JOBS", 1):
            job_id = self.service.submit([self.source], TranscriptionOptions())
            self.wait_job(job_id)
            self.service.shutdown(True)
        self.assertFalse(self.service.busy)
        with patch.object(job_service, "write_text_atomic", side_effect=OSError("disk")):
            self.service._persist(job_id)
        self.assertFalse(self.service.busy)

    def test_api_key_is_redacted_from_logs_errors_and_manifest(self):
        credential = "memory-only-test-credential"

        def failure(job, _options, api_key, reporter):
            reporter.log(f"Key accidentally echoed: {api_key}")
            reporter.file(0, error=f"Credential {api_key}")
            raise RuntimeError(api_key)

        self.engine.run.side_effect = failure
        job_id = self.service.submit([self.source], TranscriptionOptions(mode="api", model="whisper-1"), credential)
        snapshot = self.wait_job(job_id)
        self.service.shutdown(True)
        self.assertNotIn(credential, json.dumps(snapshot))
        self.assertNotIn(credential, (self.paths.results / job_id / "job.json").read_text(encoding="utf-8"))

    def test_running_manifest_recovers_without_deleting_results_or_recordings(self):
        self.engine.run.side_effect = self.complete
        job_id = self.service.submit([self.source], TranscriptionOptions())
        self.wait_job(job_id)
        self.service.shutdown(True)
        manifest = self.paths.results / job_id / "job.json"
        job = json.loads(manifest.read_text(encoding="utf-8"))
        job["status"] = "running"
        manifest.write_text(json.dumps(job), encoding="utf-8")
        salvage = self.paths.temp / "native_recordings" / "partial.wav"
        salvage.parent.mkdir(parents=True)
        salvage.write_bytes(b"recoverable")
        restarted = JobService(self.paths, transcription=Mock())
        self.addCleanup(restarted.shutdown, True)
        recovered = restarted.snapshot(job_id)
        self.assertEqual(recovered["status"], "cancelled")
        self.assertTrue(Path(recovered["files"][0]["out_path"]).exists())
        self.assertTrue(salvage.exists())


class EngineRegressionTests(unittest.TestCase):
    def test_document_failure_keeps_completed_transcription_and_closes_client(self):
        with tempfile.TemporaryDirectory() as directory:
            paths = AppPaths.create(Path(directory))
            source = Path(directory) / "source.wav"
            source.write_bytes(b"audio")
            chunk = paths.temp / "part.mp3"
            chunk.write_bytes(b"mp3")
            client = Mock()
            client.audio.transcriptions.create.return_value = SimpleNamespace(text="Décision validée.")
            client.responses.create.side_effect = RuntimeError("document unavailable")
            media = Mock()
            media.prepare_api_chunks.return_value = ([chunk], None, 1.0)
            engine = TranscriptionService(paths, media=media)
            service = JobService(paths, transcription=engine)
            with patch.object(openai_service, "make_openai_client", return_value=client):
                job_id = service.submit([source], TranscriptionOptions(mode="api", model="whisper-1", output_type="resume"), "fake-credential")
                deadline = time.monotonic() + 3
                while service.busy and time.monotonic() < deadline:
                    time.sleep(0.005)
                service.shutdown(True)
            snapshot = service.snapshot(job_id)
            self.assertEqual(snapshot["status"], "partial")
            self.assertTrue(snapshot["has_transcriptions"])
            self.assertFalse(snapshot["has_documents"])
            self.assertEqual(Path(snapshot["files"][0]["out_path"]).read_text(encoding="utf-8"), "Décision validée.\n")
            client.close.assert_called_once_with()

    def test_cloud_cancellation_preserves_completed_chunk_before_next_request(self):
        with tempfile.TemporaryDirectory() as directory:
            chunks = [Path(directory) / f"part{i}.mp3" for i in range(2)]
            for chunk in chunks:
                chunk.write_bytes(b"mp3")
            client = Mock()
            client.audio.transcriptions.create.return_value = SimpleNamespace(text="Déjà transcrit")
            received = []

            def check():
                if received:
                    raise JobCancelled()

            with self.assertRaises(JobCancelled):
                openai_service.transcribe_chunks(client, chunks, "whisper-1", None, check,
                                                  lambda index, text: received.append((index, text)))
            self.assertEqual(received, [(1, "Déjà transcrit")])
            self.assertEqual(client.audio.transcriptions.create.call_count, 1)
            self.assertNotIn("language", client.audio.transcriptions.create.call_args.kwargs)

    def test_cloud_order_context_and_partial_results_survive_later_failure(self):
        for fail_second in (False, True):
            with self.subTest(fail_second=fail_second), tempfile.TemporaryDirectory() as directory:
                paths = AppPaths.create(Path(directory))
                chunks = [paths.temp / f"part_{i:03d}.mp3" for i in (1, 2)]
                for chunk in chunks:
                    chunk.write_bytes(b"mp3")
                calls = []

                def create(**kwargs):
                    calls.append({**kwargs, "filename": Path(kwargs["file"].name).name})
                    if fail_second and len(calls) == 2:
                        raise RuntimeError("connection lost")
                    return SimpleNamespace(text=f"texte {Path(kwargs['file'].name).name}")

                client = SimpleNamespace(audio=SimpleNamespace(transcriptions=SimpleNamespace(create=create)), close=lambda: None)
                engine = TranscriptionService(paths, media=Mock(), models=Mock())
                engine.media.prepare_api_chunks.return_value = (chunks, None, 700.0)
                service = JobService(paths, transcription=engine)
                source = Path(directory) / "source.mp4"
                source.write_bytes(b"source")
                with patch.object(openai_service, "make_openai_client", return_value=client):
                    job_id = service.submit([source], TranscriptionOptions(mode="api", model="whisper-1"), "fake-credential")
                    deadline = time.monotonic() + 3
                    while service.busy and time.monotonic() < deadline:
                        time.sleep(0.005)
                    service.shutdown(True)
                snapshot = service.snapshot(job_id)
                self.assertEqual(snapshot["status"], "partial" if fail_second else "done")
                self.assertEqual([call["filename"] for call in calls], ["part_001.mp3", "part_002.mp3"])
                self.assertNotIn("prompt", calls[0])
                self.assertEqual(calls[1]["prompt"], "texte part_001.mp3")
                text = Path(snapshot["files"][0]["transcription_path"]).read_text(encoding="utf-8")
                self.assertEqual(text, "texte part_001.mp3\n" if fail_second else "texte part_001.mp3\n\ntexte part_002.mp3\n")

    def test_local_partial_text_survives_iterator_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            paths = AppPaths.create(Path(directory))
            source = Path(directory) / "source.wav"
            source.write_bytes(b"audio")
            model = Mock()

            def segments():
                yield SimpleNamespace(text="Première phrase", end=1)
                raise RuntimeError("backend lost")

            model.transcribe.return_value = (segments(), SimpleNamespace(duration=10))
            engine = TranscriptionService(paths, models=Mock())
            service = JobService(paths, transcription=engine)
            with patch.object(whisper_local, "load_model", return_value=model):
                job_id = service.submit([source], TranscriptionOptions())
                deadline = time.monotonic() + 3
                while service.busy and time.monotonic() < deadline:
                    time.sleep(0.005)
                service.shutdown(True)
            snapshot = service.snapshot(job_id)
            self.assertEqual(snapshot["status"], "partial")
            self.assertEqual(Path(snapshot["files"][0]["out_path"]).read_text(encoding="utf-8"), "Première phrase\n")
            model.transcribe.assert_called_once_with(str(source), language="fr", beam_size=5, vad_filter=True)


class ModelDownloadTests(unittest.TestCase):
    def test_model_directory_equal_to_legacy_is_adopted_without_truncating_sources(self):
        with tempfile.TemporaryDirectory() as directory:
            project = Path(directory)
            paths = AppPaths.create(project)
            target = paths.models / "small"
            target.mkdir()
            originals = {name: f"original-{name}".encode() for name in ("model.bin", "config.json", "tokenizer.json")}
            for name, data in originals.items():
                (target / name).write_bytes(data)
            fake_module = project / "src" / "transcripteur_whisper" / "services" / "model_service.py"
            with patch.object(model_service, "__file__", str(fake_module)), patch.object(model_service, "snapshot_try_candidates") as remote:
                service = model_service.ModelService(paths)
                self.assertEqual(service.download("small"), target)
                self.assertTrue(service.available("small"))
                remote.assert_not_called()
            for name, expected in originals.items():
                self.assertEqual((target / name).read_bytes(), expected)

    def test_model_import_preserves_individually_hardlinked_source_files(self):
        with tempfile.TemporaryDirectory() as directory:
            project = Path(directory)
            paths = AppPaths.create(project / "user-data")
            source = project / "models" / "small"
            target = paths.models / "small"
            source.mkdir(parents=True)
            target.mkdir()
            originals = {name: f"original-{name}".encode() for name in ("model.bin", "config.json", "tokenizer.json")}
            for name, data in originals.items():
                (source / name).write_bytes(data)
                (target / name).hardlink_to(source / name)
            fake_module = project / "src" / "transcripteur_whisper" / "services" / "model_service.py"
            with patch.object(model_service, "__file__", str(fake_module)), patch.object(model_service, "snapshot_try_candidates") as remote:
                service = model_service.ModelService(paths)
                service.download("small")
                self.assertTrue(service.available("small"))
                remote.assert_not_called()
            for name, expected in originals.items():
                self.assertEqual((source / name).read_bytes(), expected)
                self.assertEqual((target / name).read_bytes(), expected)

    def test_sequential_transport_pins_revision_and_downloads_only_runtime_files(self):
        information = SimpleNamespace(sha="immutable-revision", siblings=[
            SimpleNamespace(rfilename="model.bin"), SimpleNamespace(rfilename="tokenizer.json"),
            SimpleNamespace(rfilename="README.md")])
        with patch("huggingface_hub.HfApi") as api, patch("huggingface_hub.hf_hub_download") as download:
            api.return_value.model_info.return_value = information
            model_service.snapshot_try_candidates("org/model", Path("target"), lambda: None, lambda _: None)
        self.assertEqual([c.kwargs["filename"] for c in download.call_args_list], ["model.bin", "tokenizer.json"])
        self.assertTrue(all(c.kwargs["revision"] == "immutable-revision" for c in download.call_args_list))

    def test_download_cancels_during_byte_progress_without_trying_another_repository(self):
        cancelled = threading.Event()

        def check():
            if cancelled.is_set():
                raise JobCancelled()

        def download(**kwargs):
            with kwargs["tqdm_class"](total=100, name="hub-internal-name") as progress:
                cancelled.set()
                progress.update(10)

        with patch.object(model_service, "snapshot_download", side_effect=download) as snapshot:
            with self.assertRaises(JobCancelled):
                model_service.snapshot_try_candidates(["org/first", "org/second"], Path("target"), check, lambda _: None)
        self.assertEqual(snapshot.call_count, 1)

    def test_huggingface_uses_current_arguments_and_cancellation(self):
        with patch.object(model_service, "snapshot_download") as download:
            model_service.snapshot_try_candidates("org/model", Path("target"), lambda: None, lambda _: None)
        kwargs = download.call_args.kwargs
        self.assertEqual(kwargs["repo_id"], "org/model")
        self.assertNotIn("resume_download", kwargs)
        self.assertNotIn("local_dir_use_symlinks", kwargs)
        with patch.object(model_service, "snapshot_download") as download:
            with self.assertRaises(JobCancelled):
                model_service.snapshot_try_candidates("org/model", Path("target"), Mock(side_effect=JobCancelled), lambda _: None)
            download.assert_not_called()

    def test_incomplete_download_has_no_completion_marker_and_can_resume(self):
        with tempfile.TemporaryDirectory() as directory:
            paths = AppPaths.create(Path(directory))
            service = model_service.ModelService(paths)

            def fake_download(_repo, target, _check, _log):
                for name in ("model.bin", "config.json", "tokenizer.json"):
                    (target / name).write_bytes(b"data")

            with patch.object(model_service, "snapshot_try_candidates", side_effect=fake_download):
                result = service.download("base")
            self.assertTrue(service.available("base"))
            (result / ".download_complete").unlink()
            with patch.object(model_service, "snapshot_try_candidates", side_effect=JobCancelled):
                with self.assertRaises(JobCancelled):
                    service.download("base")
            self.assertFalse(service.available("base"))
            self.assertTrue((result / "model.bin").exists())


if __name__ == "__main__":
    unittest.main()
