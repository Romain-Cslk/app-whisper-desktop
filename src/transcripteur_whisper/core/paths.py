"""User data is independent of the source tree and installation location."""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path


def asset_path(name: str) -> Path:
    return Path(__file__).resolve().parents[1] / "assets" / name


@dataclass(frozen=True)
class AppPaths:
    root: Path
    config: Path
    models: Path
    logs: Path
    temp: Path
    results: Path
    assets: Path
    speakers: Path

    @classmethod
    def create(cls, root: Path | None = None) -> "AppPaths":
        explicit_root = root is not None
        if root is None:
            override = os.getenv("WHISPER_DATA_DIR")
            if override:
                root = Path(override).expanduser()
            elif sys.platform == "win32":
                root = Path(os.getenv("LOCALAPPDATA") or Path.home() / "AppData" / "Local") / "TranscripteurWhisper"
            else:
                root = Path(os.getenv("XDG_DATA_HOME") or Path.home() / ".local" / "share") / "TranscripteurWhisper"
        root = root.resolve()
        models = Path(
            (os.getenv("WHISPER_MODELS_DIR") if not explicit_root else None) or root / "models"
        ).expanduser().resolve()
        paths = cls(
            root=root,
            config=root / "config",
            models=models,
            logs=root / "logs",
            temp=root / "temp",
            results=root / "results",
            assets=asset_path(""),
            speakers=root / "speakers",
        )
        for directory in (
            paths.root,
            paths.config,
            paths.models,
            paths.logs,
            paths.temp,
            paths.results,
            paths.speakers,
        ):
            directory.mkdir(parents=True, exist_ok=True)
        return paths

    def asset_path(self, name: str) -> Path:
        return self.assets / name
