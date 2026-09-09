"""Opt-in offline integration: Windows synthetic speech -> cached Whisper -> TXT.

Run from the checkout with an existing models/small directory:
    python scripts/validate_local_model.py --report build/local-model-smoke.json
No microphone is opened and all model/cache test data is temporary.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from transcripteur_whisper.core.paths import AppPaths  # noqa: E402
from transcripteur_whisper.models.transcription import TranscriptionOptions  # noqa: E402
from transcripteur_whisper.services.job_service import JobService  # noqa: E402
from transcripteur_whisper.services.model_service import ModelService  # noqa: E402


def synthesize_speech(destination: Path) -> None:
    script = """
Add-Type -AssemblyName System.Speech
$validationSynth = New-Object System.Speech.Synthesis.SpeechSynthesizer
try {
    $validationSynth.SetOutputToWaveFile($env:WHISPER_VALIDATION_WAV)
    $validationSynth.Speak('Bonjour. Ceci est un test de transcription locale. La réunion est prévue demain à dix heures. Le projet fonctionne sans connexion internet.')
} finally {
    $validationSynth.Dispose()
}
"""
    environment = dict(os.environ, WHISPER_VALIDATION_WAV=str(destination))
    result = subprocess.run(
        ["powershell.exe", "-NoProfile", "-NonInteractive", "-WindowStyle", "Hidden", "-Command", script],
        env=environment, capture_output=True, text=True, timeout=30,
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
    )
    if result.returncode or not destination.is_file():
        raise RuntimeError(f"Windows speech synthesis failed: {result.stderr.strip()}")


def run_validation() -> dict:
    start = time.monotonic()
    with tempfile.TemporaryDirectory(prefix="whisper-offline-validation-") as directory:
        paths = AppPaths.create(Path(directory))
        sample = paths.temp / "synthetic-speech.wav"
        synthesize_speech(sample)
        # The actual download service imports the existing checkout model into
        # the user-layout test cache. Any attempted remote download is a failure.
        with patch("transcripteur_whisper.services.model_service.snapshot_try_candidates",
                   side_effect=AssertionError("Offline validation attempted a network download")):
            service = JobService(paths)
            try:
                job_id = service.submit([sample], TranscriptionOptions(model="small", language="fr"))
                deadline = time.monotonic() + 180
                while service.busy and time.monotonic() < deadline:
                    time.sleep(0.1)
                result = service.snapshot(job_id)
                if service.busy:
                    raise RuntimeError("Local validation exceeded 180 seconds")
                if result["status"] != "done":
                    raise RuntimeError(json.dumps(result, ensure_ascii=False))
            finally:
                service.shutdown(wait=True)
            transcript = Path(result["files"][0]["transcription_path"]).read_text(encoding="utf-8").strip()
            if not transcript:
                raise RuntimeError("Synthetic speech produced an empty transcription")
            restarted = JobService(paths)
            try:
                cache = ModelService(paths)
                available_after_restart = cache.available("small")
                marker = cache.model_path("small") / ".download_complete"
                modified_before = marker.stat().st_mtime_ns
                cache.download("small")
                cache_reused = marker.stat().st_mtime_ns == modified_before
                exported = restarted.export(job_id, paths.root / "export.txt", "transcription")
                persisted = exported.read_text(encoding="utf-8").strip() == transcript
            finally:
                restarted.shutdown(wait=True)
        if not (available_after_restart and cache_reused and persisted):
            raise RuntimeError("Cache reuse or persistent TXT export failed")
        return {
            "status": "passed", "utc": datetime.now(timezone.utc).isoformat(),
            "python": sys.version, "duration_seconds": round(time.monotonic() - start, 2),
            "input": "Windows System.Speech synthetic speech; no microphone",
            "backend": "faster-whisper small / CPU int8 / VAD Silero / beam_size 5",
            "network_downloads": 0, "transcript": transcript,
            "cache_available_after_restart": available_after_restart,
            "cache_reused_without_download": cache_reused,
            "persistent_export_verified": persisted,
        }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, default=Path("build/local-model-smoke.json"))
    args = parser.parse_args()
    try:
        evidence = run_validation()
    except Exception as exc:
        evidence = {"status": "failed", "error": str(exc), "utc": datetime.now(timezone.utc).isoformat()}
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(evidence, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(evidence, ensure_ascii=True))
    return 0 if evidence["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
