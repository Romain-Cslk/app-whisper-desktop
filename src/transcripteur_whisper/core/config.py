"""Runtime limits and the original transcription/document configuration."""
from __future__ import annotations

import os
from typing import Dict, List

# HTTP progress callbacks allow cancellation; the native Xet transfer does not
# offer an equivalent bounded cancellation contract to this desktop lifecycle.
# Set this before importing Whisper/Hugging Face (also in application bootstrap).
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

MAX_API_CHUNK_SECONDS = 10 * 60
API_AUDIO_SAMPLE_RATE = 16_000
API_AUDIO_BIT_RATE = 64_000
MAX_API_CHUNK_BYTES = 24 * 1024 * 1024
API_CHUNK_CONTENT_SECONDS = MAX_API_CHUNK_SECONDS - 1
MAX_UPLOAD_BYTES = int(os.getenv("WHISPER_MAX_UPLOAD_MB", "2048")) * 1024 * 1024
MAX_JOB_UPLOAD_BYTES = int(os.getenv("WHISPER_MAX_JOB_UPLOAD_MB", "4096")) * 1024 * 1024
MAX_FILES_PER_JOB = int(os.getenv("WHISPER_MAX_FILES", "50"))
MAX_JOB_LOG_ENTRIES = 500
MAX_API_JOB_WORKERS = max(1, int(os.getenv("WHISPER_API_JOB_WORKERS", "2")))
MAX_QUEUED_JOBS = max(MAX_API_JOB_WORKERS + 1, int(os.getenv("WHISPER_MAX_QUEUED_JOBS", "20")))
DOCUMENT_MODEL = os.getenv("WHISPER_DOCUMENT_MODEL", "gpt-5-mini").strip() or "gpt-5-mini"
DOCUMENT_CHUNK_CHARS = max(20_000, int(os.getenv("WHISPER_DOCUMENT_CHUNK_CHARS", "80000")))
DOCUMENT_MAP_OUTPUT_TOKENS = 3_000
DOCUMENT_FINAL_OUTPUT_TOKENS = 12_000
DOCUMENT_MAX_REDUCTION_ROUNDS = 8
OPENAI_TIMEOUT_SECONDS = max(30.0, float(os.getenv("WHISPER_OPENAI_TIMEOUT_SECONDS", "180")))
OPENAI_MAX_RETRIES = max(0, int(os.getenv("WHISPER_OPENAI_MAX_RETRIES", "1")))
AUDIO_EXTENSIONS = {".aac", ".flac", ".m4a", ".mp3", ".mpga", ".ogg", ".wav"}
VIDEO_EXTENSIONS = {".mkv", ".mov", ".mp4", ".webm"}
SUPPORTED_MEDIA_EXTENSIONS = AUDIO_EXTENSIONS | VIDEO_EXTENSIONS
MODELS_LOCAL: Dict[str, str] = {
    "Base": "base",
    "Small": "small",
    "Medium": "medium",
    "Large v3 (CPU lourd)": "large-v3",
    "Large v3 Turbo (recommandé)": "large-v3-turbo",
}
MODELS_CLOUD: List[str] = [
    "gpt-4o-transcribe",
    "gpt-4o-mini-transcribe",
    "whisper-1",
    "gpt-transcribe",
]
LANGS: Dict[str, str] = {
    "Français": "fr", "Anglais": "en", "Espagnol": "es", "Allemand": "de",
    "Italien": "it", "Portugais": "pt", "Néerlandais": "nl", "Russe": "ru",
    "Arabe": "ar", "Chinois": "zh", "Japonais": "ja",
}
DEFAULT_LANG = "Français"
OUTPUT_PROMPTS: Dict[str, str] = {
    "resume": (
        "À partir de cette transcription, rédige un résumé concis qui mette en avant uniquement :\n"
        "- les points clés évoqués,\n"
        "- les décisions majeures,\n"
        "- les éléments stratégiques ou critiques à retenir.\n"
        "Exclue tout détail opérationnel ou annexe. Présente le texte sous forme de liste à puces claire et synthétique.\n\n"
        "{texte}"
    ),
    "compte_rendu": (
        "À partir de cette transcription, rédige un compte rendu structuré comprenant :\n"
        "- Contexte de la réunion (objet, date, participants),\n"
        "- Points abordés (classés par thème),\n"
        "- Décisions prises,\n"
        "- Actions à réaliser avec les responsables et échéances associées.\n"
        "La rédaction doit être claire, factuelle et sans interprétation.\n\n"
        "{texte}"
    ),
    "note_de_cadrage": (
        "À partir de cette transcription, rédige une note de cadrage de projet structurée avec les sections suivantes :\n"
        "- Contexte : rappel de la situation et du besoin,\n"
        "- Objectifs : finalités attendues,\n"
        "- Périmètre : inclusions et exclusions,\n"
        "- Enjeux : valeur ajoutée et impacts,\n"
        "- Livrables attendus,\n"
        "- Planning prévisionnel (jalons clés),\n"
        "- Acteurs et rôles (sponsors, responsables, contributeurs),\n"
        "- Risques identifiés et contraintes.\n"
        "La note doit être rédigée de manière synthétique et opérationnelle.\n\n"
        "{texte}"
    ),
    "cahier_des_charges": (
        "À partir de cette transcription, rédige un cahier des charges comprenant :\n"
        "- Contexte et objectifs,\n"
        "- Besoins fonctionnels (détaillés par grandes fonctionnalités attendues),\n"
        "- Besoins techniques (infrastructure, compatibilités, performances),\n"
        "- Contraintes (budgétaires, réglementaires, organisationnelles),\n"
        "- Critères de validation et d’acceptation.\n"
        "Le document doit être clair, structuré par rubriques numérotées, et utilisable comme base contractuelle.\n\n"
        "{texte}"
    ),
    "procedure_technique": (
        "À partir de cette transcription, rédige une procédure technique comprenant :\n"
        "- Objectif de la procédure,\n"
        "- Prérequis et conditions initiales,\n"
        "- Étapes détaillées numérotées,\n"
        "- Points de contrôle et vérifications à effectuer,\n"
        "- Erreurs fréquentes et solutions associées.\n"
        "La rédaction doit être claire, pas-à-pas et utilisable directement par un consultant technique.\n\n"
        "{texte}"
    ),
    "rapport_analyse": (
        "À partir de cette transcription, rédige un rapport d’analyse comprenant :\n"
        "- Contexte et problématique,\n"
        "- Analyse détaillée des points discutés,\n"
        "- Scénarios ou options identifiées (avantages / inconvénients),\n"
        "- Recommandations formulées,\n"
        "- Conclusion et perspectives.\n"
        "La rédaction doit être factuelle, hiérarchisée et adaptée à une prise de décision.\n\n"
        "{texte}"
    ),
    "support_formation": (
        "À partir de cette transcription, rédige un support de formation clair et pédagogique comprenant :\n"
        "- Objectifs pédagogiques,\n"
        "- Notions clés expliquées simplement,\n"
        "- Exemples concrets d’application,\n"
        "- Bonnes pratiques et erreurs à éviter,\n"
        "- Résumé final.\n"
        "Le ton doit être didactique et structuré, avec des titres et sous-titres clairs.\n\n"
        "{texte}"
    ),
}
MODEL_APPROX_SIZE = {
    "base":     150 * 1024**2,   # ~150 MB
    "small":    470 * 1024**2,   # ~470 MB
    "medium":  1500 * 1024**2,   # ~1.5 GB
    "large-v2": 2900 * 1024**2,  # ~2.9 GB
    "large-v3": 3100 * 1024**2,  # ~3.1 GB
}
REPO_MAP = {
    "base": "Systran/faster-whisper-base",
    "small": "Systran/faster-whisper-small",
    "medium": "Systran/faster-whisper-medium",
    "large-v2": "Systran/faster-whisper-large-v2",
    "large-v3": "Systran/faster-whisper-large-v3",
    "large-v3-turbo": [
        "mobiuslabsgmbh/faster-whisper-large-v3-turbo",
        "deepdml/faster-whisper-large-v3-turbo-ct2",
        "Purfview/faster-whisper-large-v3-turbo",
    ],
}

MODEL_APPROX_SIZE["large-v3-turbo"] = 1600 * 1024**2
DEFAULT_MODEL_LOCAL = "Base"
