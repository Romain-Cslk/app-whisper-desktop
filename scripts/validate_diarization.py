"""Packaging smoke for the isolated sherpa-onnx worker."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--models-dir", required=True, type=Path)
    parser.add_argument("--executable", type=Path)
    parser.add_argument("--work-dir", type=Path)
    args = parser.parse_args()
    models = args.models_dir.resolve()
    segmentation = models / "segmentation.onnx"
    embedding = models / "embedding.onnx"
    if not segmentation.is_file() or not embedding.is_file():
        raise RuntimeError("Diarization model assets are missing")
    root = args.work_dir or Path(tempfile.mkdtemp(prefix="whisper-diarization-smoke-"))
    root.mkdir(parents=True, exist_ok=True)
    request = root / "request.json"
    response = root / "response.json"
    request.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "action": "probe",
                "segmentation_model": str(segmentation),
                "embedding_model": str(embedding),
            }
        ),
        encoding="utf-8",
    )
    if args.executable:
        command = [str(args.executable.resolve()), "--diarization-worker", str(request), str(response)]
    else:
        command = [sys.executable, "-m", "transcripteur_whisper", "--diarization-worker", str(request), str(response)]
    completed = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=180, check=False)
    if completed.returncode != 0 or not response.is_file():
        raise RuntimeError(
            "Diarization worker smoke failed: "
            + completed.stderr.decode("utf-8", errors="replace")[-4000:]
        )
    payload = json.loads(response.read_text(encoding="utf-8"))
    if not payload.get("ok") or int(payload.get("embedding_dimension") or 0) < 16:
        raise RuntimeError(f"Invalid diarization smoke response: {payload}")
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
