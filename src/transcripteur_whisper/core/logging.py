"""Rotating, redacted diagnostics including otherwise uncaught exceptions."""

from __future__ import annotations

import logging
import re
import sys
import threading
from logging.handlers import RotatingFileHandler
from typing import Callable

from .paths import AppPaths


def redact(text: str) -> str:
    text = re.sub(r"\bsk-[A-Za-z0-9_-]+", "[clé masquée]", text)
    return re.sub(r"(?i)(authorization\s*[:=]\s*(?:bearer\s+)?)[^\s,;]+", r"\1[masqué]", text)


class RedactingFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        return redact(super().format(record))


def configure_logging(paths: AppPaths) -> None:
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    for handler in tuple(root.handlers):
        if getattr(handler, "_whisper_handler", False):
            root.removeHandler(handler)
            handler.close()
    handler = RotatingFileHandler(paths.logs / "app.log", maxBytes=5 * 1024**2, backupCount=3, encoding="utf-8")
    handler._whisper_handler = True
    handler.setFormatter(RedactingFormatter("%(asctime)s %(levelname)s %(name)s %(message)s"))
    root.addHandler(handler)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)


def install_exception_handler(notify: Callable[[str], None]) -> None:
    def handle(exc_type: type[BaseException], exc: BaseException, traceback: object) -> None:
        logging.getLogger("transcripteur_whisper").critical("Unhandled exception", exc_info=(exc_type, exc, traceback))
        notify("Une erreur inattendue est survenue. Les détails sont disponibles dans le dossier des journaux.")

    sys.excepthook = handle
    threading.excepthook = lambda args: handle(args.exc_type, args.exc_value, args.exc_traceback)
