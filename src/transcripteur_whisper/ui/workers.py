"""Small task pool that always delivers callbacks on its owning GUI thread."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any
from uuid import uuid4

from PySide6.QtCore import QObject, QRunnable, Qt, QThreadPool, Signal, Slot


class TaskSignals(QObject):
    result = Signal(str, object)
    failed = Signal(str, str)
    finished = Signal(str)


class Task(QRunnable):
    def __init__(self, identifier: str, function: Callable[[], Any]) -> None:
        super().__init__()
        self.identifier = identifier
        self.function = function
        self.signals = TaskSignals()

    def run(self) -> None:
        try:
            result = self.function()
        except Exception as exc:
            self.signals.failed.emit(self.identifier, str(exc))
        else:
            self.signals.result.emit(self.identifier, result)
        finally:
            self.function = lambda: None
            self.signals.finished.emit(self.identifier)


@dataclass
class PendingTask:
    task: Task
    success: Callable[[Any], None] | None
    failure: Callable[[str], None] | None
    finish: Callable[[], None] | None


class TaskRunner(QObject):
    """Keep workers alive and route their signals through QObject slots.

    No lambda is connected directly to a worker signal: that would make thread
    affinity ambiguous. Callback closures run only inside the slots below.
    """

    idle = Signal()
    error = Signal(str)

    def __init__(self, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self.pool = QThreadPool(self)
        self.pool.setMaxThreadCount(4)
        self._pending: dict[str, PendingTask] = {}

    @property
    def active_count(self) -> int:
        return len(self._pending)

    def submit(
        self,
        function: Callable[[], Any],
        success: Callable[[Any], None] | None = None,
        failure: Callable[[str], None] | None = None,
        finish: Callable[[], None] | None = None,
    ) -> str:
        identifier = uuid4().hex
        task = Task(identifier, function)
        self._pending[identifier] = PendingTask(task, success, failure, finish)
        task.signals.result.connect(self._result, Qt.ConnectionType.QueuedConnection)
        task.signals.failed.connect(self._failed, Qt.ConnectionType.QueuedConnection)
        task.signals.finished.connect(self._finished, Qt.ConnectionType.QueuedConnection)
        self.pool.start(task)
        return identifier

    @Slot(str, object)
    def _result(self, identifier: str, value: Any) -> None:
        pending = self._pending.get(identifier)
        if pending and pending.success:
            pending.success(value)

    @Slot(str, str)
    def _failed(self, identifier: str, message: str) -> None:
        pending = self._pending.get(identifier)
        if pending and pending.failure:
            pending.failure(message)
        else:
            self.error.emit(message)

    @Slot(str)
    def _finished(self, identifier: str) -> None:
        pending = self._pending.pop(identifier, None)
        if pending and pending.finish:
            pending.finish()
        if not self._pending:
            self.idle.emit()
