"""One bounded local excerpt at a time; Qt owns decoding and audio output."""
from __future__ import annotations

from PySide6.QtCore import QObject, QTimer, QUrl, Signal
from PySide6.QtMultimedia import QAudioOutput, QMediaPlayer

from ...services.speaker_audio import AudioExcerpt


class SpeakerExcerptPlayer(QObject):
    status = Signal(str)
    failed = Signal(str)

    def __init__(self, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self.audio = QAudioOutput(self)
        self.audio.setVolume(0.8)
        self.player = QMediaPlayer(self)
        self.player.setAudioOutput(self.audio)
        self._clip: AudioExcerpt | None = None
        self._pending = False
        self._end_ms = 0
        self._timer = QTimer(self)
        self._timer.setInterval(20)
        self._timer.timeout.connect(self._check_end)
        self._loading = QTimer(self)
        self._loading.setSingleShot(True)
        self._loading.setInterval(15000)
        self._loading.timeout.connect(lambda: self._error("Le chargement de l'audio a expir\u00e9."))
        self.player.mediaStatusChanged.connect(self._media_status)
        self.player.durationChanged.connect(self._try_start)
        self.player.seekableChanged.connect(self._try_start)
        self.player.positionChanged.connect(self._check_end)
        self.player.errorOccurred.connect(lambda _code, message: self._error(message))

    def play_excerpt(self, clip: AudioExcerpt) -> None:
        self.stop()
        if not clip.path.is_file():
            self.failed.emit("Audio introuvable. Associez le fichier audio d'origine pour \u00e9couter les extraits.")
            return
        self._clip = clip
        self._pending = True
        self.status.emit("Chargement de l'extrait\u2026")
        self._loading.start()
        # Reset first so stale positions from the preceding source are never played.
        self.player.setSource(QUrl())
        self.player.setSource(QUrl.fromLocalFile(str(clip.path.resolve())))

    def _try_start(self, *_args) -> None:
        clip = self._clip
        if not self._pending or clip is None:
            return
        if self.player.mediaStatus() not in {
            QMediaPlayer.MediaStatus.LoadedMedia, QMediaPlayer.MediaStatus.BufferedMedia,
        }:
            return
        duration = self.player.duration()
        start = max(0, round(clip.start * 1000))
        if duration <= 0 or (start > 0 and not self.player.isSeekable()):
            return
        if start >= duration:
            self._error("L'extrait est hors de l'audio choisi. V\u00e9rifiez qu'il s'agit bien du fichier original.")
            return
        self._end_ms = min(duration, round(clip.end * 1000))
        self._pending = False
        self._loading.stop()
        self.player.setPosition(start)
        self.player.play()
        self._timer.start()
        warning = " (plusieurs voix peuvent se chevaucher)" if clip.overlap_possible else ""
        self.status.emit(f"Lecture de {clip.cluster_id}{warning}")

    def _media_status(self, status) -> None:
        if self._clip is None:
            return
        if status == QMediaPlayer.MediaStatus.InvalidMedia:
            self._error(self.player.errorString() or "Format audio non lisible par Qt.")
        elif status == QMediaPlayer.MediaStatus.EndOfMedia:
            self.stop()
        else:
            self._try_start()

    def _check_end(self, *_args) -> None:
        if self._clip is not None and not self._pending and self.player.position() >= self._end_ms:
            self.stop()

    def _error(self, message: str) -> None:
        if self._clip is None:
            return
        self.stop()
        self.failed.emit(message or "Impossible de lire cet extrait audio.")

    def stop(self) -> None:
        self._clip = None
        self._pending = False
        self._timer.stop()
        self._loading.stop()
        self.player.stop()
        self.status.emit("Lecture arr\u00eat\u00e9e.")

    def close(self) -> None:
        self.stop()
        self.player.setSource(QUrl())
