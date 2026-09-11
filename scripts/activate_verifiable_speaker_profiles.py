"""Activate strict, replayable speaker-profile evidence using existing local data.

What it does:
- keeps every existing profile vector in profiles.v1.dpapi;
- quarantines legacy vectors that have no replayable audio;
- reconstructs recoverable historical audio evidence using the manual-review tool;
- imports existing human GOOD/BAD/UNSURE decisions when present;
- stores each recoverable verification clip encrypted with Windows DPAPI;
- enables strict matching so quarantined/BAD/UNSURE vectors no longer identify people.

No profile vector is deleted. A timestamped backup of the evidence sidecar is created
before activation when one already exists.
"""
from __future__ import annotations

import argparse
import importlib.util
import shutil
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

from transcripteur_whisper.core.paths import AppPaths
from transcripteur_whisper.services.speaker_evidence import EvidenceRecordRequest, SpeakerEvidenceStore
from transcripteur_whisper.services.speaker_profile_service import SpeakerProfileService


def load_review_tool(repo: Path):
    path = repo / "scripts" / "review_speaker_profile_audio.py"
    module_name = "speaker_audio_review_tool"
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError("Impossible de charger l'outil de revue audio.")
    module = importlib.util.module_from_spec(spec)
    # dataclasses resolves postponed annotations through sys.modules while the module
    # is being executed. Register it before exec_module(), exactly like a normal import.
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(module_name, None)
        raise
    return module


def main() -> int:
    parser = argparse.ArgumentParser(description="Active la quarantaine des empreintes vocales non vérifiables.")
    parser.add_argument("--provider", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--min-link-similarity", type=float, default=0.97)
    args = parser.parse_args()
    if not 0.5 <= args.min_link_similarity <= 1.0:
        raise SystemExit("--min-link-similarity doit etre compris entre 0.5 et 1.0")

    repo = Path.cwd().resolve()
    paths = AppPaths.create()
    profiles_service = SpeakerProfileService(paths)
    evidence = SpeakerEvidenceStore(paths, codec=profiles_service.codec)
    profiles = list(profiles_service.list_profiles())
    if not profiles:
        raise SystemExit("Aucun profil vocal disponible.")

    if evidence.path.is_file():
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        backup = evidence.path.with_name(f"{evidence.path.name}.{stamp}.bak")
        shutil.copy2(evidence.path, backup)
        print(f"Sauvegarde evidence : {backup}")

    review_tool = load_review_tool(repo)
    decisions_path = Path(paths.speakers) / review_tool.DECISIONS_FILE
    decisions = review_tool.load_decisions(decisions_path)

    print("\n=== ACTIVATION DES PROFILS VOCAUX VERIFIABLES ===\n")
    print("1/3 - Mise en quarantaine logique de toutes les empreintes legacy...")
    evidence.save_quarantine(profiles)

    by_model: dict[str, list[Any]] = defaultdict(list)
    for profile in profiles:
        by_model[profile.model_set_id].append(profile)

    candidate_cache: dict[str, list[Any]] = {}
    requests: list[EvidenceRecordRequest] = []
    linked_by_profile: dict[str, int] = defaultdict(int)
    status_by_profile: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))

    print("2/3 - Reconstruction des extraits audio encore recuperables...")
    for model_set_id, model_profiles in by_model.items():
        candidates = review_tool.reconstruct_candidates(
            paths,
            provider=args.provider,
            profile_model_set_id=model_set_id,
        )
        candidate_cache[model_set_id] = candidates
        print(f"  {model_set_id}: {len(candidates)} extrait(s) candidat(s)")

        for profile in model_profiles:
            links = review_tool.link_profile_embeddings(
                profile.embeddings,
                candidates,
                args.min_link_similarity,
            )
            stored = (
                decisions.get("profiles", {})
                .get(profile.profile_id, {})
                .get("embeddings", {})
            )
            for index, (vector, link) in enumerate(zip(profile.embeddings, links), 1):
                candidate, similarity = link
                decision = stored.get(str(index), {}) if isinstance(stored, dict) else {}
                human = str(decision.get("decision") or "").upper()
                if human not in {"GOOD", "BAD", "UNSURE"}:
                    human = "UNVERIFIED"

                if candidate is None:
                    # It stays QUARANTINED from step 1. Do not manufacture evidence.
                    status_by_profile[profile.name]["QUARANTINED"] += 1
                    continue

                linked_by_profile[profile.name] += 1
                status_by_profile[profile.name][human] += 1
                requests.append(EvidenceRecordRequest(
                    profile_id=profile.profile_id,
                    embedding=vector,
                    status=human,
                    excerpt=candidate.excerpt,
                    metadata={
                        "profile_name": profile.name,
                        "model_set_id": profile.model_set_id,
                        "review_id": candidate.review_id,
                        "source": candidate.source,
                        "source_name": candidate.review_name,
                        "historical_name": candidate.historical_name,
                        "link_similarity": similarity,
                        "actual_name": str(decision.get("actual_name") or ""),
                        "note": str(decision.get("note") or ""),
                    },
                ))

    print("3/3 - Chiffrement et stockage des preuves audio...")
    if requests:
        evidence.save_records(requests)

    print("\n=== RESULTAT ===\n")
    print(f"{'PROFIL':<24} {'TOTAL':>5} {'AUDIO':>5} {'GOOD':>5} {'BAD':>5} {'UNSURE':>7} {'UNVERIF':>8} {'QUAR':>5}")
    print("-" * 78)
    for profile in profiles:
        counts = status_by_profile[profile.name]
        print(
            f"{profile.name:<24.24} {len(profile.embeddings):>5} {linked_by_profile[profile.name]:>5} "
            f"{counts['GOOD']:>5} {counts['BAD']:>5} {counts['UNSURE']:>7} "
            f"{counts['UNVERIFIED']:>8} {counts['QUARANTINED']:>5}"
        )

    print("\nMode strict ACTIVE.")
    print("- GOOD et UNVERIFIED avec audio chiffre sont eligibles au matching.")
    print("- BAD, UNSURE et QUARANTINED sont exclus du matching automatique.")
    print("- aucune empreinte n'a ete supprimee du profil source.")
    print(f"- preuves audio chiffrees : {evidence.audio_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
