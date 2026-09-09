"""Align timestamped transcription with local speaker turns."""
from __future__ import annotations

from dataclasses import replace

from ..models.speaker import DiarizationResult, DiarizationTurn, SpeakerMatch
from ..models.transcript import StructuredTranscript, TranscriptSegment, TranscriptWord
from .speaker_profile_service import SpeakerProfileService


def _overlap(start: float, end: float, turn: DiarizationTurn) -> float:
    return max(0.0, min(end, turn.end) - max(start, turn.start))


def _best_turn(start: float, end: float, turns: tuple[DiarizationTurn, ...],
               source_hint: str | None = None) -> DiarizationTurn | None:
    turns = tuple(turn for turn in turns if source_hint is None or turn.source == source_hint)
    midpoint = (start + end) / 2.0 if end > start else start
    end = max(end, start + 0.02)
    scored = sorted(((_overlap(start, end, turn), turn) for turn in turns),
                    key=lambda item: item[0], reverse=True)
    positive = [(amount, turn) for amount, turn in scored if amount > 0]
    if positive:
        amount, winner = positive[0]
        rival = next((value for value, turn in positive[1:] if turn.cluster_id != winner.cluster_id), 0.0)
        if rival and amount - rival <= max(0.025, 0.10 * amount):
            # Mixed-channel overlapping speech has no defensible automatic winner.
            return None
        return winner
    nearest = sorted(((min(abs(midpoint - turn.start), abs(midpoint - turn.end)), turn)
                      for turn in turns), key=lambda item: item[0])
    if not nearest or nearest[0][0] > 0.45:
        return None
    distance, winner = nearest[0]
    if any(turn.cluster_id != winner.cluster_id and other - distance <= 0.025
           for other, turn in nearest[1:]):
        return None
    return winner


def build_labels(
    diarization: DiarizationResult,
    profiles: SpeakerProfileService,
    self_name: str,
) -> tuple[dict[str, str], dict[str, SpeakerMatch]]:
    self_name = profiles.validate_name(self_name or "Moi")
    matches: dict[str, SpeakerMatch] = {}
    labels: dict[str, str] = {}
    first_seen = {cluster.cluster_id: float("inf") for cluster in diarization.clusters}
    for turn in diarization.turns:
        first_seen[turn.cluster_id] = min(first_seen.get(turn.cluster_id, turn.start), turn.start)
    unknown = 1
    for cluster in sorted(diarization.clusters, key=lambda item: first_seen.get(item.cluster_id, float("inf"))):
        if cluster.source == "self":
            labels[cluster.cluster_id] = self_name
            matches[cluster.cluster_id] = SpeakerMatch(
                None, self_name, True, 1.0, 1.0, False, "piste microphone personnelle"
            )
            continue
        match = profiles.match_cluster_for_model(cluster, diarization.model_set_id)
        matches[cluster.cluster_id] = match
        if match.accepted and match.name:
            labels[cluster.cluster_id] = self_name if match.is_self else match.name
        else:
            labels[cluster.cluster_id] = f"Intervenant {unknown}"
            unknown += 1
    return labels, matches


def _speaker_meta(
    turn: DiarizationTurn | None,
    labels: dict[str, str],
    matches: dict[str, SpeakerMatch],
) -> tuple[str | None, str, str | None, float | None]:
    if turn is None:
        return None, "Intervenant inconnu", None, None
    match = matches.get(turn.cluster_id)
    confidence = match.score if match and match.accepted else None
    return turn.cluster_id, labels.get(turn.cluster_id, "Intervenant inconnu"), turn.source, confidence


def align_transcript(
    transcript: StructuredTranscript,
    diarization: DiarizationResult,
    labels: dict[str, str],
    matches: dict[str, SpeakerMatch],
) -> StructuredTranscript:
    turns = diarization.turns
    output: list[TranscriptSegment] = []
    for segment in transcript.segments:
        if segment.words:
            groups: list[tuple[DiarizationTurn | None, list[TranscriptWord]]] = []
            for word in segment.words:
                turn = _best_turn(word.start, word.end, turns, segment.speaker_source)
                if groups and (
                    (groups[-1][0] is None and turn is None)
                    or (
                        groups[-1][0] is not None
                        and turn is not None
                        and groups[-1][0].cluster_id == turn.cluster_id
                    )
                ):
                    groups[-1][1].append(word)
                else:
                    groups.append((turn, [word]))
            for turn, words in groups:
                text = "".join(word.text for word in words).strip()
                if not text:
                    continue
                key, name, source, confidence = _speaker_meta(turn, labels, matches)
                output.append(
                    TranscriptSegment(
                        start=words[0].start,
                        end=words[-1].end,
                        text=text,
                        words=tuple(words),
                        speaker_key=key,
                        speaker_name=name,
                        speaker_source=source,
                        speaker_confidence=confidence,
                    )
                )
        else:
            turn = _best_turn(segment.start, segment.end, turns, segment.speaker_source)
            key, name, source, confidence = _speaker_meta(turn, labels, matches)
            output.append(
                replace(
                    segment,
                    speaker_key=key,
                    speaker_name=name,
                    speaker_source=source,
                    speaker_confidence=confidence,
                )
            )
    return StructuredTranscript(tuple(output), transcript.language)
