from __future__ import annotations

import threading
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from transcripteur_whisper.services import document_service as document
from transcripteur_whisper.services.executor import DaemonJobExecutor


def completed_response(text: str) -> SimpleNamespace:
    return SimpleNamespace(status="completed", output_text=text)


class FakeResponses:
    def __init__(self, responder):
        self._responder = responder
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return self._responder(len(self.calls), kwargs)


class FakeClient:
    def __init__(self, responder):
        self.responses = FakeResponses(responder)


class DocumentPipelineTests(unittest.TestCase):
    def test_short_document_uses_one_separated_responses_call(self) -> None:
        transcript = "Une décision importante a été prise."
        client = FakeClient(lambda _index, _kwargs: completed_response("Résumé final"))

        with (
            patch.object(document, "DOCUMENT_MODEL", "gpt-document-test"),
            patch.object(document, "DOCUMENT_CHUNK_CHARS", 1_000),
            patch.object(document, "DOCUMENT_FINAL_OUTPUT_TOKENS", 777),
        ):
            result = document.generate_document(client, "resume", transcript)

        self.assertEqual(result, "Résumé final")
        self.assertEqual(len(client.responses.calls), 1)
        call = client.responses.calls[0]
        self.assertEqual(call["model"], "gpt-document-test")
        self.assertEqual(call["max_output_tokens"], 777)
        self.assertIs(call["store"], False)
        self.assertEqual(call["input"], transcript)
        self.assertNotIn(transcript, call["instructions"])
        self.assertNotIn("{texte}", call["instructions"])
        self.assertIn("résumé", call["instructions"].lower())

    def test_long_document_maps_reduces_and_finalizes_in_source_order(self) -> None:
        source_chunks = [character * 14 for character in "ABCD"]
        transcript = "".join(source_chunks)
        map_outputs = {
            1: "NOTE-A",
            2: "NOTE-B",
            3: "NOTE-C",
            4: "NOTE-D",
        }

        def responder(_call_index, kwargs):
            input_text = kwargs["input"]
            if input_text.startswith("EXTRAIT "):
                source_index = int(input_text.split()[1].split("/")[0])
                return completed_response(map_outputs[source_index])
            if input_text.startswith("NOTES DES EXTRAITS 1 À 2"):
                return completed_response("R-12")
            if input_text.startswith("NOTES DES EXTRAITS 3 À 4"):
                return completed_response("R-34")
            return completed_response("DOCUMENT FINAL COMPLET")

        client = FakeClient(responder)
        with (
            patch.object(document, "DOCUMENT_MODEL", "gpt-long-test"),
            patch.object(document, "DOCUMENT_CHUNK_CHARS", 14),
            patch.object(document, "DOCUMENT_MAP_OUTPUT_TOKENS", 321),
            patch.object(document, "DOCUMENT_FINAL_OUTPUT_TOKENS", 654),
        ):
            result = document.generate_document(client, "compte_rendu", transcript)

        self.assertEqual(result, "DOCUMENT FINAL COMPLET")
        calls = client.responses.calls
        self.assertEqual(len(calls), 7)

        map_calls = calls[:4]
        reduce_calls = calls[4:6]
        final_call = calls[6]
        self.assertEqual(
            [call["input"].split("\n\n", 1)[1] for call in map_calls],
            source_chunks,
        )
        self.assertTrue(all(call["max_output_tokens"] == 321 for call in calls[:6]))
        self.assertEqual(reduce_calls[0]["input"], "NOTES DES EXTRAITS 1 À 2\n\nNOTE-A\n\nNOTE-B")
        self.assertEqual(reduce_calls[1]["input"], "NOTES DES EXTRAITS 3 À 4\n\nNOTE-C\n\nNOTE-D")
        self.assertEqual(final_call["input"], "R-12\n\nR-34")
        self.assertEqual(final_call["max_output_tokens"], 654)
        self.assertTrue(all(call["model"] == "gpt-long-test" for call in calls))
        self.assertTrue(all(call["store"] is False for call in calls))

    def test_incomplete_and_empty_responses_are_rejected(self) -> None:
        cases = (
            (
                SimpleNamespace(
                    status="incomplete",
                    incomplete_details=SimpleNamespace(reason="max_output_tokens"),
                    output_text="Texte partiel",
                ),
                "incomplète",
            ),
            (completed_response("   "), "vide"),
        )

        for response, expected_message in cases:
            with self.subTest(expected_message=expected_message):
                client = FakeClient(lambda _index, _kwargs, value=response: value)
                with self.assertRaisesRegex(RuntimeError, expected_message):
                    document.generate_document(client, "resume", "Texte court")
                self.assertEqual(len(client.responses.calls), 1)

    def test_cancellation_after_response_discards_the_result(self) -> None:
        class CancelledAfterResponse(RuntimeError):
            pass

        cancellation = CancelledAfterResponse("annulation après l'appel")
        cancel_calls = 0

        def cancel_check() -> None:
            nonlocal cancel_calls
            cancel_calls += 1
            if cancel_calls == 2:
                raise cancellation

        client = FakeClient(lambda _index, _kwargs: completed_response("Résultat à ignorer"))
        with self.assertRaises(CancelledAfterResponse) as raised:
            document.generate_document(
                client,
                "resume",
                "Texte court",
                cancel_check=cancel_check,
            )

        self.assertIs(raised.exception, cancellation)
        self.assertEqual(cancel_calls, 2)
        self.assertEqual(len(client.responses.calls), 1)

    def test_split_text_uses_the_current_dynamic_default(self) -> None:
        text = "ABCDEFGHIJ"
        with patch.object(document, "DOCUMENT_CHUNK_CHARS", 4):
            chunks_of_four = document.split_text_for_model(text)
        with patch.object(document, "DOCUMENT_CHUNK_CHARS", 3):
            chunks_of_three = document.split_text_for_model(text)

        self.assertEqual(chunks_of_four, ["ABCD", "EFGH", "IJ"])
        self.assertEqual(chunks_of_three, ["ABC", "DEF", "GHI", "J"])
        self.assertEqual("".join(chunks_of_four), text)
        self.assertEqual("".join(chunks_of_three), text)

    def test_group_document_notes_keeps_whole_notes_and_provenance(self) -> None:
        notes = [
            (1, 1, "aaaa"),
            (2, 2, "bbbb"),
            (3, 4, "cc"),
        ]

        with patch.object(document, "DOCUMENT_CHUNK_CHARS", 10):
            groups = document.group_document_notes(notes)

        self.assertEqual(groups, [notes[:2], notes[2:]])
        self.assertEqual([note for group in groups for note in group], notes)
        self.assertEqual(groups[0][0][:2], (1, 1))
        self.assertEqual(groups[0][-1][:2], (2, 2))
        self.assertEqual(groups[1][0][:2], (3, 4))


class DaemonJobExecutorTests(unittest.TestCase):
    def test_workers_are_daemon_threads(self) -> None:
        executor = DaemonJobExecutor(2, "daemon-test")
        try:
            self.assertEqual(len(executor._threads), 2)
            self.assertTrue(all(worker.daemon for worker in executor._threads))
            self.assertTrue(all(worker.is_alive() for worker in executor._threads))
        finally:
            executor.shutdown(wait=True, cancel_futures=True)

    def test_shutdown_cancels_pending_future_and_runs_callback_once(self) -> None:
        executor = DaemonJobExecutor(1, "cancel-test")
        started = threading.Event()
        release = threading.Event()
        callback_called = threading.Event()
        callback_futures = []

        def blocking_task() -> str:
            started.set()
            release.wait()
            return "running task complete"

        running = executor.submit(blocking_task)
        self.assertTrue(started.wait(2), "La tâche bloquante n'a pas démarré.")
        pending = executor.submit(lambda: "must not run")

        def on_done(future) -> None:
            callback_futures.append(future)
            callback_called.set()

        pending.add_done_callback(on_done)
        try:
            executor.shutdown(wait=False, cancel_futures=True)
            self.assertTrue(pending.cancelled())
            self.assertTrue(callback_called.wait(1))
            self.assertEqual(callback_futures, [pending])
            self.assertFalse(running.done())
        finally:
            release.set()
            executor.shutdown(wait=False, cancel_futures=True)
            for worker in executor._threads:
                worker.join(timeout=2)

        self.assertEqual(running.result(timeout=1), "running task complete")
        self.assertTrue(all(not worker.is_alive() for worker in executor._threads))

    def test_shutdown_wait_false_returns_while_running_task_is_blocked(self) -> None:
        executor = DaemonJobExecutor(1, "nonblocking-test")
        started = threading.Event()
        release = threading.Event()
        shutdown_returned = threading.Event()
        shutdown_errors = []

        def blocking_task() -> str:
            started.set()
            release.wait()
            return "released"

        running = executor.submit(blocking_task)
        self.assertTrue(started.wait(2), "La tâche bloquante n'a pas démarré.")

        def invoke_shutdown() -> None:
            try:
                executor.shutdown(wait=False, cancel_futures=True)
            except BaseException as exc:  # pragma: no cover - diagnostic de test
                shutdown_errors.append(exc)
            finally:
                shutdown_returned.set()

        shutdown_thread = threading.Thread(target=invoke_shutdown, daemon=True)
        shutdown_thread.start()
        try:
            self.assertTrue(
                shutdown_returned.wait(1),
                "shutdown(wait=False) a attendu la tâche bloquée.",
            )
            self.assertEqual(shutdown_errors, [])
            self.assertFalse(running.done())
            self.assertTrue(executor._threads[0].daemon)
            self.assertTrue(executor._threads[0].is_alive())
        finally:
            release.set()
            shutdown_thread.join(timeout=1)
            executor.shutdown(wait=False, cancel_futures=True)
            for worker in executor._threads:
                worker.join(timeout=2)

        self.assertEqual(running.result(timeout=1), "released")
        self.assertFalse(executor._threads[0].is_alive())


if __name__ == "__main__":
    unittest.main()
