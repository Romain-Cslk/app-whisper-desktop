"""CLI diagnostic for the local CPU/CUDA backend."""
from __future__ import annotations

import argparse
import json

from transcripteur_whisper.services.compute_backend import validate_compute_backend


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--diarization", action="store_true")
    args = parser.parse_args()
    info = validate_compute_backend(args.device, require_diarization=args.diarization)
    print(
        json.dumps(
            {
                "ok": True,
                "device": info.device,
                "whisper_compute_type": info.whisper_compute_type,
                "sherpa_provider": info.sherpa_provider,
                "cuda_device_count": info.cuda_device_count,
                "sherpa_version": info.sherpa_version,
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
