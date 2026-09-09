"""Business failures and thread-safe progress contracts."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable


class JobError(RuntimeError):
    """An operation could not be completed."""


class ValidationError(JobError):
    """Input or configuration is not supported."""


class JobCancelled(JobError):
    """Cooperative cancellation requested by the user."""


TERMINAL_STATUSES = frozenset({"done", "partial", "error", "cancelled"})


@dataclass(frozen=True)
class ProgressReporter:
    """The engine reports updates without retaining UI or job storage."""

    cancel_check: Callable[[], None]
    log: Callable[[str], None]
    file: Callable[..., None]
    job: Callable[..., None]

