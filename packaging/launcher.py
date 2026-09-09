"""PyInstaller entry point (absolute import also works outside package execution)."""

from transcripteur_whisper.app import main

if __name__ == "__main__":
    raise SystemExit(main())
