"""Safe Windows filenames and atomic persistent results."""
from __future__ import annotations

import re
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

from ..core.config import SUPPORTED_MEDIA_EXTENSIONS
from ..models.jobs import ValidationError


def safe_upload_name(raw_name: Optional[str], index: int) -> str:
    """Retourne un nom interne sûr, sans chemin ni caractères Windows réservés."""
    basename = re.split(r"[\\/]", raw_name or "")[-1].strip()
    basename = re.sub(r"[\x00-\x1f<>:\"/\\|?*]", "_", basename).rstrip(". ")
    if not basename:
        basename = f"media_{index:03d}"

    suffix = Path(basename).suffix.lower()
    if suffix not in SUPPORTED_MEDIA_EXTENSIONS:
        allowed = ", ".join(sorted(SUPPORTED_MEDIA_EXTENSIONS))
        raise ValidationError(f"Format non pris en charge pour '{basename}'. Formats acceptés : {allowed}",
        )

    stem = Path(basename).stem[:140].rstrip(". ") or f"media_{index:03d}"
    return f"{index:03d}_{stem}{suffix}"

def safe_output_name(raw_name: Optional[str]) -> str:
    """Nettoie le nom libre ou fournit un horodatage local."""
    name = re.split(r"[\\/]", raw_name or "")[-1].strip()
    name = re.sub(r"[\x00-\x1f<>:\"/\\|?*]", "_", name).rstrip(". ")
    if name.lower().endswith(".txt"):
        name = name[:-4].rstrip(". ")
    name = re.sub(r"\s+", " ", name)[:80].rstrip(". ")
    return name or datetime.now().astimezone().strftime("%d_%m_%Y_%Hh%M")

def output_filename(job: Dict[str, Any], kind: str, index: Optional[int] = None) -> str:
    """Construit un nom de sortie sûr à partir du type et du nom demandé."""
    output_name = job.get("output_name")
    if not output_name:
        files = job.get("files", [])
        if index is not None and index < len(files) and files[index].get("output_stem"):
            return f"{files[index]['output_stem']}_{kind}.txt"
        return f"{kind}.txt"

    name = f"{kind}_{output_name}"
    if index is not None and len(job.get("files", [])) > 1:
        name += f"_{index + 1:02d}"
    return f"{name}.txt"

def write_text_atomic(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        temporary.write_text(
            text + ("\n" if text and not text.endswith("\n") else ""),
            encoding="utf-8",
        )
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)

def format_seconds_hms(seconds: float) -> str:
    total = max(0, int(round(seconds)))
    hours, remainder = divmod(total, 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"
