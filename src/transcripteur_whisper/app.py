"""Qt application entry point, isolated workers and executable smoke validation."""

from __future__ import annotations

import argparse
import importlib
import json
import logging
import os
import sys
from pathlib import Path

from transcripteur_whisper import __version__
from transcripteur_whisper.core.logging import configure_logging, install_exception_handler
from transcripteur_whisper.core.paths import AppPaths


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Transcripteur Whisper")
    parser.add_argument("--version", action="version", version=__version__)
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--smoke-visible", action="store_true")
    parser.add_argument("--smoke-report", type=Path)
    parser.add_argument(
        "--diarization-worker",
        nargs=2,
        metavar=("REQUEST_JSON", "RESPONSE_JSON"),
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--speaker-sample-worker",
        nargs=2,
        metavar=("REQUEST_JSON", "RESPONSE_JSON"),
        help=argparse.SUPPRESS,
    )
    args = parser.parse_args(argv)
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

    # IMPORTANT: execute native inference workers before importing Qt, faster-whisper
    # or the app services. sherpa-onnx therefore gets a clean native-runtime process.
    if args.diarization_worker:
        from transcripteur_whisper.services.diarization_worker import worker_main

        request, response = (Path(value) for value in args.diarization_worker)
        return worker_main(request, response)
    if args.speaker_sample_worker:
        from transcripteur_whisper.services.speaker_sample_worker import worker_main

        request, response = (Path(value) for value in args.speaker_sample_worker)
        return worker_main(request, response)

    if args.smoke_test and not args.smoke_visible:
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    if sys.stdout is None:
        sys.stdout = open(os.devnull, "w", encoding="utf-8")
    if sys.stderr is None:
        sys.stderr = open(os.devnull, "w", encoding="utf-8")
    from PySide6.QtCore import QObject, QTimer, QUrl, Signal, Slot
    from PySide6.QtGui import QDesktopServices, QFont, QIcon
    from PySide6.QtWidgets import QApplication, QMessageBox

    app = QApplication.instance() or QApplication(sys.argv[:1])
    app.setStyle("Fusion")
    app.setFont(QFont("Segoe UI", 10))
    app.setApplicationName("Transcripteur Whisper")
    app.setOrganizationName("TranscripteurWhisper")
    app.setApplicationVersion(__version__)
    try:
        paths = AppPaths.create()
        configure_logging(paths)
        app.setWindowIcon(QIcon(str(paths.asset_path("icon.ico"))))
    except Exception as exc:
        if not args.smoke_test:
            QMessageBox.critical(
                None, "Démarrage impossible", f"Impossible de préparer les dossiers utilisateur : {exc}"
            )
        return 1

    class CrashNotifier(QObject):
        raised = Signal(str)

        @Slot(str)
        def show_error(self, message: str) -> None:
            if args.smoke_test:
                app.exit(1)
                return
            dialog = QMessageBox(QMessageBox.Icon.Critical, "Erreur de l'application", message)
            open_logs = dialog.addButton("Ouvrir les journaux", QMessageBox.ButtonRole.ActionRole)
            dialog.addButton(QMessageBox.StandardButton.Close)
            dialog.exec()
            if dialog.clickedButton() is open_logs:
                QDesktopServices.openUrl(QUrl.fromLocalFile(str(paths.logs)))

    notifier = CrashNotifier()
    notifier.raised.connect(notifier.show_error)
    install_exception_handler(notifier.raised.emit)
    report: dict[str, object] = {
        "version": __version__,
        "frozen": bool(getattr(sys, "frozen", False)),
        "ok": False,
    }
    exit_code = 0
    try:
        from transcripteur_whisper.ui.main_window import MainWindow

        window = MainWindow(paths, monitor=False) if args.smoke_test else MainWindow(paths)
        window.show()
        if args.smoke_test:
            versions = {}
            for name in (
                "PySide6",
                "faster_whisper",
                "ctranslate2",
                "av",
                "soundcard",
                "sounddevice",
                "soundfile",
                "numpy",
                "onnxruntime",
                "openai",
            ):
                module = importlib.import_module(name)
                versions[name] = getattr(module, "__version__", "imported")
            from av import Codec
            from faster_whisper.vad import get_vad_model
            from PySide6.QtGui import QPixmap
            from PySide6.QtMultimedia import QMediaDevices

            Codec("libmp3lame", "w")
            get_vad_model()
            for name in ("icon.ico", "logo.png", "logo_white.png"):
                if not paths.asset_path(name).is_file() or QPixmap(str(paths.asset_path(name))).isNull():
                    raise RuntimeError(f"Ressource manquante ou invalide : {name}")
            monitor = QMediaDevices()
            report.update(
                imports=versions,
                qt_multimedia=type(monitor).__name__,
                window=type(window).__name__,
                assets=True,
                paths=True,
                vad=True,
                mp3_encoder=True,
                diarization_isolated=True,
            )
            QTimer.singleShot(600, window.close)
            QTimer.singleShot(10000, lambda: app.exit(2))
        exit_code = app.exec()
        report["ok"] = exit_code == 0
        logging.getLogger(__name__).info("Application closed with code %s", exit_code)
    except Exception:
        logging.getLogger(__name__).exception("Application startup/smoke failed")
        report["error"] = "Startup/smoke failed; see app.log"
        if not args.smoke_test:
            notifier.show_error("Impossible de démarrer l'application. Consultez les journaux pour le détail.")
        exit_code = 1
    finally:
        if args.smoke_report:
            args.smoke_report.parent.mkdir(parents=True, exist_ok=True)
            args.smoke_report.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return exit_code
