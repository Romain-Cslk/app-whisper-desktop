"""Isolated sherpa-onnx worker.

This module intentionally imports sherpa_onnx only inside the worker process.
The main process can therefore keep faster-whisper/onnxruntime loaded without
sharing their native runtime state with sherpa-onnx.
"""
from __future__ import annotations

import json
import math
import sys
import traceback
from pathlib import Path
from typing import Any

import numpy as np

from ..models.speaker import DiarizationResult, DiarizationTurn, SpeakerCluster
from .diarization_model_service import MODEL_SET_ID
from .diarization_options import DEFAULT_CLUSTERING_THRESHOLD, validate_diarization_options
from .speaker_audio import _subtract

TARGET_SAMPLE_RATE = 16_000
MIN_EMBED_CLIP_SECONDS = 1.5
MAX_EMBED_CLIP_SECONDS = 8.0
MAX_EMBEDDINGS_PER_CLUSTER = 6


def _decode_audio(path: Path) -> tuple[np.ndarray, int]:
    import av
    from av.audio.resampler import AudioResampler

    chunks: list[np.ndarray] = []
    with av.open(str(path)) as container:
        stream = next((item for item in container.streams if item.type == "audio"), None)
        if stream is None:
            raise RuntimeError(f"Aucune piste audio dans {path.name}.")
        resampler = AudioResampler(format="fltp", layout="mono", rate=TARGET_SAMPLE_RATE)
        for frame in container.decode(stream):
            for converted in resampler.resample(frame):
                data = np.asarray(converted.to_ndarray(), dtype=np.float32)
                if data.ndim == 2:
                    data = data[0]
                data = np.ascontiguousarray(data.reshape(-1), dtype=np.float32)
                if data.size:
                    chunks.append(data)
        for converted in resampler.resample(None):
            data = np.asarray(converted.to_ndarray(), dtype=np.float32)
            if data.ndim == 2:
                data = data[0]
            data = np.ascontiguousarray(data.reshape(-1), dtype=np.float32)
            if data.size:
                chunks.append(data)
    if not chunks:
        raise RuntimeError(f"Aucun échantillon audio exploitable dans {path.name}.")
    samples = np.concatenate(chunks)
    if not np.isfinite(samples).all():
        samples = np.nan_to_num(samples, nan=0.0, posinf=0.0, neginf=0.0)
    return np.clip(samples, -1.0, 1.0).astype(np.float32, copy=False), TARGET_SAMPLE_RATE


def _make_diarizer(
    segmentation_model: Path,
    embedding_model: Path,
    num_speakers: int,
    clustering_threshold: float = DEFAULT_CLUSTERING_THRESHOLD,
    provider: str = "cpu",
):
    if provider == "cuda":
        from .compute_backend import prepare_sherpa_windows_runtime

        prepare_sherpa_windows_runtime(provider)
    import sherpa_onnx

    config = sherpa_onnx.OfflineSpeakerDiarizationConfig(
        segmentation=sherpa_onnx.OfflineSpeakerSegmentationModelConfig(
            pyannote=sherpa_onnx.OfflineSpeakerSegmentationPyannoteModelConfig(
                model=str(segmentation_model), window_shift_ratio=0.1
            ),
            num_threads=2,
            provider=provider,
        ),
        embedding=sherpa_onnx.SpeakerEmbeddingExtractorConfig(
            model=str(embedding_model), num_threads=2, provider=provider
        ),
        clustering=sherpa_onnx.FastClusteringConfig(num_clusters=num_speakers, threshold=clustering_threshold),
        min_duration_on=0.3,
        min_duration_off=0.5,
    )
    if not config.validate():
        raise RuntimeError("Configuration sherpa-onnx de diarisation invalide.")
    return sherpa_onnx.OfflineSpeakerDiarization(config)


def _make_extractor(embedding_model: Path, provider: str = "cpu"):
    if provider == "cuda":
        from .compute_backend import prepare_sherpa_windows_runtime

        prepare_sherpa_windows_runtime(provider)
    import sherpa_onnx

    config = sherpa_onnx.SpeakerEmbeddingExtractorConfig(
        model=str(embedding_model), num_threads=2, provider=provider
    )
    if not config.validate():
        raise RuntimeError("Configuration sherpa-onnx d'empreinte vocale invalide.")
    return sherpa_onnx.SpeakerEmbeddingExtractor(config)


def _embedding(extractor: Any, samples: np.ndarray, sample_rate: int) -> tuple[float, ...] | None:
    if samples.size < int(MIN_EMBED_CLIP_SECONDS * sample_rate):
        return None
    stream = extractor.create_stream()
    stream.accept_waveform(sample_rate=sample_rate, waveform=np.ascontiguousarray(samples, dtype=np.float32))
    stream.input_finished()
    if hasattr(extractor, "is_ready") and not extractor.is_ready(stream):
        return None
    value = np.asarray(extractor.compute(stream), dtype=np.float32)
    if value.ndim != 1 or value.size < 16 or not np.isfinite(value).all():
        return None
    norm = float(np.linalg.norm(value))
    if not math.isfinite(norm) or norm <= 1e-8:
        return None
    value /= norm
    return tuple(float(item) for item in value)


def _clean_turns(raw_turns: list[DiarizationTurn]) -> list[DiarizationTurn]:
    valid = [item for item in raw_turns if math.isfinite(item.start) and math.isfinite(item.end)
             and item.start >= 0 and item.duration >= 0.12]
    by_speaker: dict[tuple[str, str], list[DiarizationTurn]] = {}
    for turn in sorted(valid, key=lambda item: (item.start, item.end)):
        key = (turn.cluster_id, turn.source)
        group = by_speaker.setdefault(key, [])
        if group and turn.start <= group[-1].end:
            old = group[-1]
            group[-1] = DiarizationTurn(old.start, max(old.end, turn.end), old.cluster_id, old.source)
        else:
            group.append(turn)
    return sorted((turn for group in by_speaker.values() for turn in group),
                  key=lambda item: (item.start, item.end, item.cluster_id))


def _overlaps_other(turn: DiarizationTurn, all_turns: list[DiarizationTurn]) -> bool:
    return any(other.cluster_id != turn.cluster_id and
               min(turn.end, other.end) - max(turn.start, other.start) > 0.08 for other in all_turns)


def _embedding_windows(turns: list[DiarizationTurn], cluster_id: str) -> list[tuple[float, float]]:
    """Non-overlapping, speaker-exclusive windows; no duplicated evidence."""
    windows = []
    for turn in _clean_turns(turns):
        if turn.cluster_id != cluster_id:
            continue
        cuts = [(other.start - 0.03, other.end + 0.03) for other in turns
                if other.cluster_id != cluster_id and other.end > turn.start and other.start < turn.end]
        for start, end in _subtract(turn.start, turn.end, cuts):
            duration = end - start
            if duration < MIN_EMBED_CLIP_SECONDS:
                continue
            count = max(1, math.ceil(duration / 4.0))
            width = duration / count
            if width < MIN_EMBED_CLIP_SECONDS:
                count, width = 1, duration
            for index in range(count):
                windows.append((start + index * width, start + (index + 1) * width))
    windows.sort()
    if len(windows) > MAX_EMBEDDINGS_PER_CLUSTER:
        positions = np.linspace(0, len(windows) - 1, MAX_EMBEDDINGS_PER_CLUSTER, dtype=int)
        windows = [windows[int(index)] for index in positions]
    return windows


def _cluster_embeddings(
    extractor: Any, samples: np.ndarray, sample_rate: int,
    turns: list[DiarizationTurn], cluster_id: str,
) -> tuple[tuple[float, ...], ...]:
    embeddings = []
    for start, end in _embedding_windows(turns, cluster_id):
        left, right = max(0, round(start * sample_rate)), min(samples.size, round(end * sample_rate))
        clip = samples[left:right]
        if not clip.size or float(np.max(np.abs(clip))) < 1e-5:
            continue
        value = _embedding(extractor, clip, sample_rate)
        if value is not None:
            embeddings.append(value)
    # A single concatenated sample is permitted for explicit enrollment only.
    # Never manufacture a second query from audio already used above.
    if not embeddings:
        parts = []
        remaining = int(MAX_EMBED_CLIP_SECONDS * sample_rate)
        for turn in _clean_turns(turns):
            if turn.cluster_id != cluster_id or _overlaps_other(turn, turns):
                continue
            left, right = max(0, round(turn.start * sample_rate)), min(samples.size, round(turn.end * sample_rate))
            if right <= left:
                continue
            clip = samples[left:min(right, left + remaining)]
            if not clip.size or float(np.max(np.abs(clip))) < 1e-5:
                continue
            parts.append(clip)
            remaining -= clip.size
            if remaining <= 0:
                break
        if parts:
            value = _embedding(extractor, np.concatenate(parts), sample_rate)
            if value is not None:
                embeddings.append(value)
    return tuple(embeddings)


def _process_one(
    path: Path,
    *,
    source: str,
    prefix: str,
    offset: float,
    segmentation_model: Path,
    embedding_model: Path,
    num_speakers: int,
    clustering_threshold: float = DEFAULT_CLUSTERING_THRESHOLD,
    provider: str = "cpu",
) -> tuple[list[DiarizationTurn], list[SpeakerCluster]]:
    samples, sample_rate = _decode_audio(path)
    if samples.size == 0 or float(np.max(np.abs(samples))) < 1e-5:
        return [], []
    diarizer = _make_diarizer(
        segmentation_model, embedding_model, num_speakers, clustering_threshold, provider
    )
    result = diarizer.process(samples).sort_by_start_time()
    raw: list[DiarizationTurn] = []
    for item in result:
        speaker = int(item.speaker)
        cluster_id = "self" if source == "self" else f"{prefix}{speaker:02d}"
        raw.append(
            DiarizationTurn(
                start=max(0.0, float(item.start) + offset),
                end=min(samples.size / sample_rate, max(0.0, float(item.end))) + offset,
                cluster_id=cluster_id,
                source=source,
            )
        )
    shifted_turns = _clean_turns(raw)
    # Extract embeddings in source-local time, not shifted global time.
    local_turns = [
        DiarizationTurn(
            max(0.0, item.start - offset),
            max(0.0, item.end - offset),
            item.cluster_id,
            item.source,
        )
        for item in shifted_turns
    ]
    extractor = _make_extractor(embedding_model, provider)
    clusters: list[SpeakerCluster] = []
    for cluster_id in sorted({item.cluster_id for item in local_turns}):
        duration = sum(item.duration for item in local_turns if item.cluster_id == cluster_id)
        embeddings = _cluster_embeddings(extractor, samples, sample_rate, local_turns, cluster_id)
        clusters.append(SpeakerCluster(cluster_id, source, duration, embeddings))
    return shifted_turns, clusters


def process_request(payload: dict[str, Any]) -> DiarizationResult:
    if int(payload.get("schema_version", 0)) != 1:
        raise ValueError("Version de protocole de diarisation invalide.")
    expected, threshold = validate_diarization_options(
        payload.get("expected_speakers", 0),
        payload.get("clustering_threshold", DEFAULT_CLUSTERING_THRESHOLD),
    )
    provider = str(payload.get("provider", "cpu") or "cpu").lower()
    if provider not in {"cpu", "cuda"}:
        raise ValueError("Provider de diarisation invalide.")
    if payload.get("model_set_id", MODEL_SET_ID) != MODEL_SET_ID:
        raise ValueError("Espace vocal incompatible dans la requete.")
    for key in ("microphone_offset", "system_offset"):
        value = float(payload.get(key) or 0.0)
        if not math.isfinite(value) or value < 0:
            raise ValueError("Decalage de piste invalide.")
    segmentation = Path(str(payload["segmentation_model"])).resolve()
    embedding = Path(str(payload["embedding_model"])).resolve()
    if not segmentation.is_file() or not embedding.is_file():
        raise FileNotFoundError("Modèle de diarisation manquant.")
    all_turns: list[DiarizationTurn] = []
    all_clusters: list[SpeakerCluster] = []
    microphone = payload.get("microphone_source")
    system = payload.get("system_source")
    if microphone or system:
        if microphone:
            turns, clusters = _process_one(
                Path(str(microphone)),
                source="self",
                prefix="self:",
                offset=max(0.0, float(payload.get("microphone_offset") or 0.0)),
                segmentation_model=segmentation,
                embedding_model=embedding,
                num_speakers=1,
                clustering_threshold=threshold,
                provider=provider,
            )
            all_turns.extend(turns)
            all_clusters.extend(clusters)
        if system:
            turns, clusters = _process_one(
                Path(str(system)),
                source="system",
                prefix="remote:",
                offset=max(0.0, float(payload.get("system_offset") or 0.0)),
                segmentation_model=segmentation,
                embedding_model=embedding,
                num_speakers=max(1, expected - int(bool(all_turns))) if expected else -1,
                clustering_threshold=threshold,
                provider=provider,
            )
            all_turns.extend(turns)
            all_clusters.extend(clusters)
    else:
        turns, clusters = _process_one(
            Path(str(payload["audio_path"])),
            source="mixed",
            prefix="speaker:",
            offset=0.0,
            segmentation_model=segmentation,
            embedding_model=embedding,
            num_speakers=expected or -1,
            clustering_threshold=threshold,
            provider=provider,
        )
        all_turns.extend(turns)
        all_clusters.extend(clusters)
    if expected == 1 and any(turn.source == "self" for turn in all_turns) and any(
        turn.source == "system" for turn in all_turns
    ):
        raise ValueError("Une voix attendue, mais de la parole est presente sur le micro ET le systeme. "
                         "Choisissez Auto ou le nombre total de voix reellement presentes.")
    all_turns.sort(key=lambda item: (item.start, item.end, item.cluster_id))
    return DiarizationResult(MODEL_SET_ID, tuple(all_turns), tuple(all_clusters))


def probe_models(
    segmentation_model: Path, embedding_model: Path, provider: str = "cpu"
) -> dict[str, Any]:
    """Load native runtimes/models and run an embedding inference.

    The waveform need not be speech: this is a packaging smoke test, not a
    quality benchmark. Real meeting quality is validated separately.
    """
    _make_diarizer(segmentation_model, embedding_model, -1, provider=provider)
    extractor = _make_extractor(embedding_model, provider)
    t = np.arange(TARGET_SAMPLE_RATE * 2, dtype=np.float32) / TARGET_SAMPLE_RATE
    samples = (0.02 * np.sin(2 * np.pi * 220.0 * t)).astype(np.float32)
    value = _embedding(extractor, samples, TARGET_SAMPLE_RATE)
    if value is None:
        raise RuntimeError("Le smoke test n'a pas produit d'empreinte vocale.")
    return {"ok": True, "embedding_dimension": len(value), "model_set_id": MODEL_SET_ID}


def worker_main(request_path: Path, response_path: Path) -> int:
    response_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        request = json.loads(request_path.read_text(encoding="utf-8"))
        action = request.get("action", "diarize")
        if action == "probe":
            result: dict[str, Any] = probe_models(
                Path(str(request["segmentation_model"])),
                Path(str(request["embedding_model"])),
                str(request.get("provider", "cpu") or "cpu").lower(),
            )
        elif action == "diarize":
            result = {"ok": True, "result": process_request(request).to_dict()}
        else:
            raise ValueError("Action de worker inconnue.")
        temporary = response_path.with_suffix(response_path.suffix + ".tmp")
        temporary.write_text(json.dumps(result, ensure_ascii=False), encoding="utf-8")
        temporary.replace(response_path)
        return 0
    except BaseException as exc:
        error = {
            "ok": False,
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(limit=12),
        }
        try:
            response_path.write_text(json.dumps(error, ensure_ascii=False), encoding="utf-8")
        except Exception:
            pass
        return 1


if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise SystemExit("usage: diarization_worker.py request.json response.json")
    raise SystemExit(worker_main(Path(sys.argv[1]), Path(sys.argv[2])))
