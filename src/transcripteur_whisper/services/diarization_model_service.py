"""Diarization model installation verified against the reviewed bundle manifest."""
from __future__ import annotations

import hashlib
import json
import shutil
import uuid
from pathlib import Path
from typing import Any, Callable

MODEL_SET_ID = "sherpa-pyannote3-wespeaker-campp-v1"


class DiarizationModelError(RuntimeError):
    pass


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class DiarizationModelService:
    """Installs exact model bytes that were verified at application build time.

    End users do not need Python, Hugging Face credentials or a model download.
    The build embeds model files plus their SHA-256 manifest. Runtime copies them
    atomically to the persistent models directory on first use and verifies both
    source and destination hashes.
    """

    def __init__(self, paths: Any) -> None:
        self.paths = paths
        self.directory = Path(paths.models) / "diarization" / MODEL_SET_ID
        self.segmentation = self.directory / "segmentation.onnx"
        self.embedding = self.directory / "embedding.onnx"
        self.manifest = self.directory / "manifest.json"

    def _bundle(self) -> tuple[Path, dict[str, Any]]:
        root = Path(self.paths.asset_path("diarization"))
        manifest_path = root / "manifest.json"
        try:
            payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise DiarizationModelError(
                "Les modèles de diarisation ne sont pas présents dans cette installation. "
                "Reconstruisez l'application avec scripts/prepare_diarization_assets.py."
            ) from exc
        required = {
            "schema_version",
            "model_set_id",
            "segmentation_sha256",
            "embedding_sha256",
        }
        if not required.issubset(payload) or int(payload["schema_version"]) != 1:
            raise DiarizationModelError("Manifest des modèles de diarisation invalide.")
        if payload["model_set_id"] != MODEL_SET_ID:
            raise DiarizationModelError("Espace d'empreintes vocales incompatible avec cette version.")
        segmentation = root / "segmentation.onnx"
        embedding = root / "embedding.onnx"
        if (
            not segmentation.is_file()
            or sha256_file(segmentation) != str(payload["segmentation_sha256"])
            or not embedding.is_file()
            or sha256_file(embedding) != str(payload["embedding_sha256"])
        ):
            raise DiarizationModelError("Un modèle de diarisation intégré est absent ou corrompu.")
        return root, payload

    def _user_available(self, manifest: dict[str, Any] | None = None) -> bool:
        try:
            if manifest is None:
                _root, manifest = self._bundle()
            return (
                self.segmentation.is_file()
                and sha256_file(self.segmentation) == str(manifest["segmentation_sha256"])
                and self.embedding.is_file()
                and sha256_file(self.embedding) == str(manifest["embedding_sha256"])
            )
        except (OSError, DiarizationModelError):
            return False

    def available(self) -> bool:
        try:
            self._bundle()
            return True
        except (DiarizationModelError, OSError, ValueError, TypeError):
            return False

    @staticmethod
    def _copy_atomic(source: Path, destination: Path) -> None:
        temporary = destination.with_name(f".{destination.name}.{uuid.uuid4().hex}.part")
        temporary.unlink(missing_ok=True)
        try:
            shutil.copyfile(source, temporary)
            temporary.replace(destination)
        finally:
            temporary.unlink(missing_ok=True)

    def ensure(
        self,
        cancel_check: Callable[[], None] | None = None,
        progress: Callable[[float], None] | None = None,
        log: Callable[[str], None] | None = None,
    ) -> tuple[Path, Path]:
        cancel_check = cancel_check or (lambda: None)
        progress = progress or (lambda _value: None)
        log = log or (lambda _message: None)
        root, manifest = self._bundle()
        if self._user_available(manifest):
            progress(1.0)
            return self.segmentation, self.embedding
        self.directory.mkdir(parents=True, exist_ok=True)
        log("Installation locale des modèles de diarisation inclus avec l'application…")
        cancel_check()
        self._copy_atomic(root / "segmentation.onnx", self.segmentation)
        if sha256_file(self.segmentation) != str(manifest["segmentation_sha256"]):
            self.segmentation.unlink(missing_ok=True)
            raise DiarizationModelError("La copie locale du modèle de segmentation est corrompue.")
        progress(0.50)
        cancel_check()
        self._copy_atomic(root / "embedding.onnx", self.embedding)
        if sha256_file(self.embedding) != str(manifest["embedding_sha256"]):
            self.embedding.unlink(missing_ok=True)
            raise DiarizationModelError("La copie locale du modèle vocal est corrompue.")
        progress(0.95)
        user_manifest = dict(manifest)
        temporary = self.manifest.with_name(f".manifest.{uuid.uuid4().hex}.tmp")
        temporary.write_text(json.dumps(user_manifest, indent=2), encoding="utf-8")
        temporary.replace(self.manifest)
        progress(1.0)
        return self.segmentation, self.embedding
