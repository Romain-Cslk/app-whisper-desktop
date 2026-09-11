"""Activate strict, replayable speaker-profile evidence using existing local data.

What it does:
- keeps every existing profile vector in profiles.v1.dpapi;
- quarantines legacy vectors that have no replayable audio;
- reconstructs recoverable historical audio evidence using the manual-review tool;
- imports existing human GOOD/BAD/UNSURE decisions when present;
- when BAD has a known ``actual_name``, promotes that exact replayable embedding as GOOD
  evidence for the target profile;
- stores every accepted verification clip encrypted with Windows DPAPI;
- enables strict matching so only human-approved GOOD vectors can identify people.

No legacy profile vector is deleted. A timestamped backup of the evidence sidecar is created
before activation when one already exists.
"""
from __future__ import annotations

import argparse
import importlib.util
import shutil
import sys
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from transcripteur_whisper.core.paths import AppPaths
from transcripteur_whisper.services.speaker_evidence import (
    EvidenceRecordRequest,
    SpeakerEvidenceStore,
    embedding_key,
)
from transcripteur_whisper.services.speaker_profile_service import SpeakerProfileService


@dataclass(frozen=True)
class LinkedItem:
    source_profile: Any
    index: int
    vector: Any
    candidate: Any
    similarity: float
    human: str
    actual_name: str
    note: str


def load_review_tool(repo: Path):
    path = repo / "scripts" / "review_speaker_profile_audio.py"
    module_name = "speaker_audio_review_tool"
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError("Impossible de charger l'outil de revue audio.")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(module_name, None)
        raise
    return module


def main() -> int:
    parser = argparse.ArgumentParser(description="Active les empreintes vocales humaines et vérifiables.")
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

    profiles_by_name = {profile.name.casefold(): profile for profile in profiles}

    if evidence.path.is_file():
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        backup = evidence.path.with_name(f"{evidence.path.name}.{stamp}.bak")
        shutil.copy2(evidence.path, backup)
        print(f"Sauvegarde evidence : {backup}")

    review_tool = load_review_tool(repo)
    decisions_path = Path(paths.speakers) / review_tool.DECISIONS_FILE
    decisions = review_tool.load_decisions(decisions_path)

    print("\n=== ACTIVATION DES PROFILS VOCAUX VERIFIABLES ===\n")
    print("1/4 - Mise en quarantaine logique des empreintes legacy non verifiees...")
    evidence.save_quarantine(profiles)

    by_model: dict[str, list[Any]] = defaultdict(list)
    for profile in profiles:
        by_model[profile.model_set_id].append(profile)

    linked_items: list[LinkedItem] = []
    origin_audio_count: dict[str, int] = defaultdict(int)
    origin_quarantine_count: dict[str, int] = defaultdict(int)

    print("2/4 - Reconstruction des extraits audio encore recuperables...")
    for model_set_id, model_profiles in by_model.items():
        candidates = review_tool.reconstruct_candidates(
            paths,
            provider=args.provider,
            profile_model_set_id=model_set_id,
        )
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
                    origin_quarantine_count[profile.name] += 1
                    continue

                origin_audio_count[profile.name] += 1
                linked_items.append(LinkedItem(
                    source_profile=profile,
                    index=index,
                    vector=vector,
                    candidate=candidate,
                    similarity=float(similarity),
                    human=human,
                    actual_name=str(decision.get("actual_name") or "").strip(),
                    note=str(decision.get("note") or ""),
                ))

    # Resolve who the human says each exact embedding belongs to. This protects against
    # accidentally making one identical vector GOOD for two different people.
    ownership: dict[str, set[str]] = defaultdict(set)
    invalid_targets: list[tuple[str, int, str]] = []
    for item in linked_items:
        target = None
        if item.human == "GOOD":
            target = item.source_profile
        elif item.human == "BAD" and item.actual_name:
            target = profiles_by_name.get(item.actual_name.casefold())
            if target is None:
                invalid_targets.append((item.source_profile.name, item.index, item.actual_name))
                continue
            if target.model_set_id != item.source_profile.model_set_id:
                invalid_targets.append((item.source_profile.name, item.index, item.actual_name))
                continue
        if target is not None:
            ownership[embedding_key(item.vector)].add(target.profile_id)

    conflicts = {key: owners for key, owners in ownership.items() if len(owners) > 1}
    requests: list[EvidenceRecordRequest] = []
    reassigned_count: dict[str, int] = defaultdict(int)
    conflict_count = 0

    print("3/4 - Application des decisions humaines et reattributions...")
    for item in linked_items:
        key = embedding_key(item.vector)
        conflict = key in conflicts
        source_status = item.human
        if conflict and source_status == "GOOD":
            # Never let an identical vector identify two people until the conflict is resolved.
            source_status = "UNSURE"
            conflict_count += 1

        common_metadata = {
            "profile_name": item.source_profile.name,
            "model_set_id": item.source_profile.model_set_id,
            "review_id": item.candidate.review_id,
            "source": item.candidate.source,
            "source_name": item.candidate.review_name,
            "historical_name": item.candidate.historical_name,
            "link_similarity": item.similarity,
            "actual_name": item.actual_name,
            "note": item.note,
        }
        requests.append(EvidenceRecordRequest(
            profile_id=item.source_profile.profile_id,
            embedding=item.vector,
            status=source_status,
            excerpt=item.candidate.excerpt,
            metadata=common_metadata,
        ))

        if item.human != "BAD" or not item.actual_name or conflict:
            continue
        target = profiles_by_name.get(item.actual_name.casefold())
        if target is None or target.profile_id == item.source_profile.profile_id:
            continue
        if target.model_set_id != item.source_profile.model_set_id:
            continue

        requests.append(EvidenceRecordRequest(
            profile_id=target.profile_id,
            embedding=item.vector,
            status="GOOD",
            excerpt=item.candidate.excerpt,
            metadata={
                "profile_name": target.name,
                "model_set_id": target.model_set_id,
                "review_id": item.candidate.review_id,
                "source": item.candidate.source,
                "source_name": item.candidate.review_name,
                "historical_name": item.candidate.historical_name,
                "link_similarity": item.similarity,
                "actual_name": target.name,
                "note": item.note,
                "reassigned_from_profile": item.source_profile.name,
                "reassigned_from_index": item.index,
            },
        ))
        reassigned_count[target.name] += 1

    print("4/4 - Chiffrement et stockage des preuves audio...")
    if requests:
        evidence.save_records(requests)

    print("\n=== RESULTAT PAR IDENTITE ===\n")
    print(
        f"{'PROFIL':<24} {'LEGACY':>6} {'AUDIO_SRC':>9} {'GOOD':>5} "
        f"{'BAD':>5} {'UNSURE':>7} {'UNVERIF':>8} {'QUAR':>5} {'RECUP':>5}"
    )
    print("-" * 92)
    for profile in profiles:
        counts = evidence.status_counts(profile.profile_id)
        print(
            f"{profile.name:<24.24} {len(profile.embeddings):>6} {origin_audio_count[profile.name]:>9} "
            f"{counts['GOOD']:>5} {counts['BAD']:>5} {counts['UNSURE']:>7} "
            f"{counts['UNVERIFIED']:>8} {counts['QUARANTINED']:>5} "
            f"{reassigned_count[profile.name]:>5}"
        )

    total_reassigned = sum(reassigned_count.values())
    print(f"\nReattributions humaines promues en GOOD : {total_reassigned}")
    if reassigned_count:
        for name in sorted(reassigned_count, key=str.casefold):
            print(f"- {name}: +{reassigned_count[name]}")

    if invalid_targets:
        print("\nReattributions ignorees (profil cible introuvable/incompatible) :")
        for source_name, index, target_name in invalid_targets:
            print(f"- {source_name} #{index} -> {target_name}")

    if conflicts:
        print(
            f"\nATTENTION : {len(conflicts)} empreinte(s) ont des identites humaines contradictoires. "
            "Elles ne sont pas promues automatiquement."
        )

    print("\nMode strict ACTIVE.")
    print("- seules les empreintes GOOD avec audio chiffre sont eligibles au matching.")
    print("- une BAD reattribuee a une personne connue devient GOOD pour cette personne.")
    print("- UNVERIFIED, BAD, UNSURE et QUARANTINED sont exclus du matching automatique.")
    print("- aucune empreinte legacy n'a ete supprimee du profil source.")
    print(f"- preuves audio chiffrees : {evidence.audio_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
