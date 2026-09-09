"""Prepare pinned diarization assets for the Windows bundle.

The release URLs and reviewed SHA-256 digests are pinned. A build fails if
upstream bytes change. Runtime then verifies every copied asset against the
manifest embedded in that build.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import tarfile
from pathlib import Path

import requests

MODEL_SET_ID = "sherpa-pyannote3-wespeaker-campp-v1"
SEG_ARCHIVE_SHA256 = "24615ee884c897d9d2ba09bb4d30da6bb1b15e685065962db5b02e76e4996488"
SEG_MODEL_SHA256 = "220ad67ca923bef2fa91f2390c786097bf305bceb5e261d4af67b38e938e1079"
EMB_SHA256 = "c46fad10b5f81e1aa4a60c162714208577093655076c5450f8c469e522ec54ef"
SEG_URL = (
    "https://github.com/k2-fsa/sherpa-onnx/releases/download/"
    "speaker-segmentation-models/sherpa-onnx-pyannote-segmentation-3-0.tar.bz2"
)
EMB_URL = (
    "https://github.com/k2-fsa/sherpa-onnx/releases/download/"
    "speaker-recongition-models/wespeaker_en_voxceleb_CAM%2B%2B.onnx"
)


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def download(url: str, path: Path, expected_sha256: str) -> None:
    if path.is_file() and digest(path) == expected_sha256:
        return
    temporary = path.with_suffix(path.suffix + ".part")
    temporary.unlink(missing_ok=True)
    with requests.get(url, stream=True, timeout=(15, 120), allow_redirects=True) as response:
        response.raise_for_status()
        with temporary.open("wb") as stream:
            for chunk in response.iter_content(1024 * 1024):
                if chunk:
                    stream.write(chunk)
            stream.flush()
    if temporary.stat().st_size <= 0 or digest(temporary) != expected_sha256:
        temporary.unlink(missing_ok=True)
        raise RuntimeError(f"SHA-256 mismatch for pinned asset: {url}")
    temporary.replace(path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("destination", type=Path)
    parser.add_argument("--refresh", action="store_true")
    args = parser.parse_args()
    target = args.destination.resolve()
    target.mkdir(parents=True, exist_ok=True)
    segmentation = target / "segmentation.onnx"
    embedding = target / "embedding.onnx"
    manifest_path = target / "manifest.json"

    if not args.refresh and segmentation.is_file() and embedding.is_file() and manifest_path.is_file():
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if (
                manifest.get("model_set_id") == MODEL_SET_ID
                and digest(segmentation) == manifest["segmentation_sha256"]
                and digest(embedding) == manifest["embedding_sha256"]
            ):
                print("Diarization assets already verified.")
                return 0
        except (OSError, ValueError, KeyError):
            pass

    archive = target / "segmentation.tar.bz2"
    try:
        download(SEG_URL, archive, SEG_ARCHIVE_SHA256)
        with tarfile.open(archive, "r:bz2") as bundle:
            members = [m for m in bundle.getmembers() if m.isfile() and Path(m.name).name == "model.onnx"]
            if len(members) != 1:
                raise RuntimeError("Unexpected pyannote segmentation archive layout")
            source = bundle.extractfile(members[0])
            if source is None:
                raise RuntimeError("Missing segmentation model in archive")
            temporary = segmentation.with_suffix(".onnx.part")
            with temporary.open("wb") as output:
                shutil.copyfileobj(source, output)
            if temporary.stat().st_size <= 0 or digest(temporary) != SEG_MODEL_SHA256:
                temporary.unlink(missing_ok=True)
                raise RuntimeError("SHA-256 mismatch for extracted segmentation model")
            temporary.replace(segmentation)

        download(EMB_URL, embedding, EMB_SHA256)
        manifest = {
            "schema_version": 1,
            "model_set_id": MODEL_SET_ID,
            "segmentation_source": SEG_URL,
            "segmentation_archive_sha256": SEG_ARCHIVE_SHA256,
            "segmentation_sha256": SEG_MODEL_SHA256,
            "segmentation_size": segmentation.stat().st_size,
            "embedding_source": EMB_URL,
            "embedding_sha256": EMB_SHA256,
            "embedding_size": embedding.stat().st_size,
        }
        temporary_manifest = manifest_path.with_suffix(".json.tmp")
        temporary_manifest.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        temporary_manifest.replace(manifest_path)
        (target / "THIRD_PARTY_NOTICES.txt").write_text(
            "Speaker segmentation: pyannote/segmentation-3.0 (MIT).\n"
            "Speaker embedding: WeSpeaker CAM++ model distributed through sherpa-onnx.\n"
            "Before external/public redistribution, review the upstream model card/data attribution terms.\n",
            encoding="utf-8",
        )
    finally:
        archive.unlink(missing_ok=True)
    print(f"Segmentation SHA-256: {digest(segmentation)}")
    print(f"Embedding SHA-256:    {digest(embedding)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
