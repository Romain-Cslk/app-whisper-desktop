"""Bounded daemon workers keep desktop shutdown responsive."""
from __future__ import annotations

import queue
import threading
from concurrent.futures import Future
from typing import Any, Callable


class DaemonJobExecutor:
    """Petit exécuteur dont les workers ne retiennent pas le processus GUI.

    L'API volontairement réduite (`submit`/`shutdown`) suffit aux jobs de
    l'application et conserve la sémantique standard de `Future.cancel()`.
    """

    _STOP = object()

    def __init__(self, max_workers: int, thread_name_prefix: str):
        if max_workers < 1:
            raise ValueError("max_workers doit être positif")
        self._queue: "queue.Queue[Any]" = queue.Queue()
        self._lock = threading.Lock()
        self._shutdown = False
        self._threads = [
            threading.Thread(
                target=self._worker,
                name=f"{thread_name_prefix}-{index + 1}",
                daemon=True,
            )
            for index in range(max_workers)
        ]
        for worker in self._threads:
            worker.start()

    def submit(self, fn: Callable, /, *args, **kwargs) -> Future:
        future: Future = Future()
        with self._lock:
            if self._shutdown:
                raise RuntimeError("L'exécuteur de jobs est arrêté.")
            self._queue.put((future, fn, args, kwargs))
        return future

    def _worker(self) -> None:
        while True:
            item = self._queue.get()
            try:
                if item is self._STOP:
                    return
                future, fn, args, kwargs = item
                if not future.set_running_or_notify_cancel():
                    continue
                try:
                    result = fn(*args, **kwargs)
                except BaseException as exc:
                    future.set_exception(exc)
                else:
                    future.set_result(result)
            finally:
                self._queue.task_done()
            # Do not retain a completed job's API credential while idle.
            item = fn = future = None
            args = ()
            kwargs = {}

    def shutdown(self, wait: bool = True, *, cancel_futures: bool = False) -> None:
        with self._lock:
            first_shutdown = not self._shutdown
            self._shutdown = True
            if first_shutdown and cancel_futures:
                while True:
                    try:
                        queued = self._queue.get_nowait()
                    except queue.Empty:
                        break
                    try:
                        if queued is not self._STOP:
                            queued[0].cancel()
                    finally:
                        self._queue.task_done()
            if first_shutdown:
                for _ in self._threads:
                    self._queue.put(self._STOP)

        if wait:
            for worker in self._threads:
                worker.join()
