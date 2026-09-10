"""Non-destructive global audit of encrypted local speaker profiles.

The script never modifies or exports biometric vectors. It prints only similarity
statistics and writes a CSV containing profile names, embedding indexes and scores.
"""
from __future__ import annotations

import csv
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from transcripteur_whisper.core.paths import AppPaths
from transcripteur_whisper.services.speaker_profile_service import SpeakerProfileService

TOP_K = 3
STRONG_OTHER_SCORE = 0.85
STRONG_NEGATIVE_MARGIN = -0.08
LIKELY_OTHER_SCORE = 0.80
LIKELY_NEGATIVE_MARGIN = -0.05
TRUSTED_MARGIN = 0.04
MIN_OWN_SCORE = 0.68


def normalize(values) -> np.ndarray:
    value = np.asarray(values, dtype=np.float32)
    norm = float(np.linalg.norm(value))
    if norm <= 1e-8 or not np.isfinite(norm):
        raise ValueError("invalid embedding")
    return value / norm


def topk_mean(values: np.ndarray, k: int = TOP_K) -> float:
    if values.size == 0:
        return -1.0
    take = min(k, values.size)
    return float(np.sort(values)[-take:].mean())


def classify(own_top3: float, other_top3: float, margin: float) -> str:
    if other_top3 >= STRONG_OTHER_SCORE and margin <= STRONG_NEGATIVE_MARGIN:
        return "FOREIGN_STRONG"
    if other_top3 >= LIKELY_OTHER_SCORE and margin <= LIKELY_NEGATIVE_MARGIN:
        return "FOREIGN_LIKELY"
    if own_top3 < MIN_OWN_SCORE or margin < TRUSTED_MARGIN:
        return "AMBIGUOUS"
    return "TRUSTED"


def main() -> int:
    service = SpeakerProfileService(AppPaths.create())
    profiles = list(service.list_profiles())
    if not profiles:
        print("Aucun profil vocal trouve.")
        return 1

    matrices: dict[str, np.ndarray] = {}
    by_id = {profile.profile_id: profile for profile in profiles}
    for profile in profiles:
        vectors = [normalize(item) for item in profile.embeddings]
        matrices[profile.profile_id] = np.asarray(vectors, dtype=np.float32)

    rows = []
    summary = defaultdict(Counter)
    transfers = Counter()

    for profile in profiles:
        own = matrices[profile.profile_id]
        for index, vector in enumerate(own):
            own_scores = own @ vector
            own_scores = np.delete(own_scores, index)
            own_top3 = topk_mean(own_scores)
            own_max = float(own_scores.max()) if own_scores.size else -1.0
            own_median = float(np.median(own_scores)) if own_scores.size else -1.0

            best_other_name = "-"
            best_other_top3 = -1.0
            best_other_max = -1.0

            for other in profiles:
                if other.profile_id == profile.profile_id or other.model_set_id != profile.model_set_id:
                    continue
                other_matrix = matrices[other.profile_id]
                if other_matrix.ndim != 2 or other_matrix.shape[1] != own.shape[1]:
                    continue
                scores = other_matrix @ vector
                candidate_top3 = topk_mean(scores)
                if candidate_top3 > best_other_top3:
                    best_other_top3 = candidate_top3
                    best_other_max = float(scores.max())
                    best_other_name = other.name

            margin = own_top3 - best_other_top3 if best_other_top3 >= -0.5 else 1.0
            status = classify(own_top3, best_other_top3, margin)
            summary[profile.name][status] += 1
            if status.startswith("FOREIGN_"):
                transfers[(profile.name, best_other_name, status)] += 1

            rows.append({
                "profile": profile.name,
                "embedding_index": index + 1,
                "status": status,
                "own_top3": own_top3,
                "own_max": own_max,
                "own_median": own_median,
                "best_other_profile": best_other_name,
                "other_top3": best_other_top3,
                "other_max": best_other_max,
                "margin": margin,
            })

    print("\n=== AUDIT GLOBAL DES PROFILS VOCAUX ===\n")
    print(f"{'PROFIL':<24} {'TOTAL':>5} {'TRUSTED':>8} {'AMBIG':>7} {'FOREIGN':>8} {'STRONG':>7}")
    print("-" * 68)
    for profile in profiles:
        counts = summary[profile.name]
        foreign = counts['FOREIGN_LIKELY'] + counts['FOREIGN_STRONG']
        print(
            f"{profile.name:<24.24} {len(profile.embeddings):>5} "
            f"{counts['TRUSTED']:>8} {counts['AMBIGUOUS']:>7} "
            f"{foreign:>8} {counts['FOREIGN_STRONG']:>7}"
        )

    print("\n=== TRANSFERTS SUSPECTS LES PLUS FORTS ===\n")
    if not transfers:
        print("Aucun transfert suspect detecte.")
    else:
        for (source, target, status), count in transfers.most_common():
            print(f"{source:<24.24} -> {target:<24.24} {status:<15} x{count}")

    severe = sorted(
        (row for row in rows if row['status'] in {'FOREIGN_STRONG', 'FOREIGN_LIKELY'}),
        key=lambda row: (0 if row['status'] == 'FOREIGN_STRONG' else 1, row['margin']),
    )
    print("\n=== EMPREINTES A CONTROLER EN PRIORITE ===\n")
    for row in severe:
        print(
            f"{row['profile']} #{row['embedding_index']:02d} -> {row['best_other_profile']} | "
            f"own={row['own_top3']:.3f} other={row['other_top3']:.3f} "
            f"margin={row['margin']:.3f} [{row['status']}]"
        )

    output = Path.cwd() / "audit_profils_global.csv"
    with output.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nCSV : {output}")
    print("Aucune empreinte n'a ete modifiee ou supprimee.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
