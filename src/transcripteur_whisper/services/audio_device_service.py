"""Core Audio endpoint discovery and safe mapping to the PortAudio cache.

SoundCard enumerates live Windows endpoint IDs; PortAudio retains its device
table until reinitialization. Refresh is serialized with opening/closing capture
and is NEVER performed while this application's streams can still be alive.
An endpoint's index may therefore be None during capture: it is available for
the *next* recording and is resolved again immediately before opening a stream.

PortAudio's public API has no endpoint ID. We match its WASAPI input by exact
normalized name and, where useful, channel count. Identically named endpoints
are deliberately refused rather than silently recording the wrong microphone.
Renaming such devices in Windows makes the mapping unambiguous. Drivers that
rename/recreate an endpoint on reconnect can require selecting it again.
"""

from __future__ import annotations

import hashlib
import threading
import unicodedata
from typing import Any

from transcripteur_whisper.audio import native_recorder as backend
from transcripteur_whisper.audio.windows_com import com_apartment
from transcripteur_whisper.models.audio_devices import AudioDevice, AudioDeviceSnapshot


class AudioDeviceError(RuntimeError):
    pass


def normalized_name(name: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", name).casefold().replace("®", "").split())


class AudioDeviceService:
    def __init__(self, *, sounddevice: Any = None, soundcard: Any = None):
        self._sd = backend.sd if sounddevice is None else sounddevice
        self._sc = backend.sc if soundcard is None else soundcard
        self.gate = threading.RLock()
        self._capture_active = False
        self._snapshot = AudioDeviceSnapshot()
        self._portaudio_fingerprint: tuple | None = None

    @property
    def current_snapshot(self) -> AudioDeviceSnapshot:
        # Immutable assignment is atomic; UI reads never wait on backend calls.
        return self._snapshot

    def set_capture_active(self, active: bool) -> None:
        """Called under gate by the sole capture owner, including failed stops."""
        with self.gate:
            self._capture_active = active

    def snapshot(self, *, force_refresh: bool = False) -> AudioDeviceSnapshot:
        """Blocking backend enumeration: always call on a worker thread."""
        with self.gate, com_apartment():
            if self._sd is None or self._sc is None:
                raise AudioDeviceError("Les bibliothèques audio Windows ne sont pas disponibles.")
            try:
                microphones = self._endpoints(self._sc.all_microphones(include_loopback=False))
                speakers = self._endpoints(self._sc.all_speakers())
                default_mic = self._default_id(self._sc.default_microphone)
                default_speaker = self._default_id(self._sc.default_speaker)
                fingerprint = (tuple(microphones), tuple(speakers), default_mic, default_speaker)
                if not self._capture_active and (force_refresh or fingerprint != self._portaudio_fingerprint):
                    self._refresh_portaudio()
                    self._portaudio_fingerprint = fingerprint
                pa_inputs = self._portaudio_inputs()
                inputs = tuple(
                    self._input(endpoint, microphones, pa_inputs, default_mic) for endpoint in microphones
                )
                outputs = tuple(
                    AudioDevice(
                        stable_key=self._key(endpoint, "output"),
                        backend_id=endpoint[0] or None,
                        backend_index=None,
                        name=endpoint[1],
                        kind="output",
                        host_api="Windows WASAPI",
                        channels=endpoint[2],
                        sample_rate=0,
                        is_default=bool(endpoint[0] and endpoint[0] == default_speaker),
                    )
                    for endpoint in speakers
                )
            except AudioDeviceError:
                raise
            except Exception as exc:
                # Keep the last snapshot if an enumeration fails transiently.
                raise AudioDeviceError(f"Détection des périphériques audio impossible : {exc}") from exc
            self._snapshot = AudioDeviceSnapshot(inputs, outputs)
            return self._snapshot

    def resolve(
        self, stable_key: str | None, kind: str, *, snapshot: AudioDeviceSnapshot | None = None
    ) -> AudioDevice:
        """Resolve against a fresh table. A missing explicit selection is an error."""
        current = snapshot if snapshot is not None else self.snapshot(force_refresh=True)
        devices = current.inputs if kind == "input" else current.outputs
        if stable_key:
            selected = next((item for item in devices if item.stable_key == stable_key), None)
        else:
            selected = next((item for item in devices if item.is_default), next(iter(devices), None))
        if selected is None:
            raise AudioDeviceError("Le périphérique audio sélectionné n'est plus disponible.")
        if kind == "input" and selected.backend_index is None:
            raise AudioDeviceError(
                f"Le microphone « {selected.name} » ne peut pas être identifié sans ambiguïté "
                "par WASAPI. Vérifiez son activation ou donnez-lui un nom unique dans Windows."
            )
        if kind == "output" and not selected.backend_id:
            raise AudioDeviceError("La sortie audio ne possède pas d'identifiant Windows utilisable.")
        return selected

    @staticmethod
    def _endpoints(devices: Any) -> list[tuple[str, str, int]]:
        return sorted(
            (str(getattr(device, "id", "")), str(device.name), backend._device_channel_count(device))
            for device in devices
        )

    @staticmethod
    def _default_id(get_default: Any) -> str:
        try:
            return str(getattr(get_default(), "id", ""))
        except Exception:
            # No default is normal when all devices of a kind are unplugged.
            return ""

    @staticmethod
    def _key(endpoint: tuple[str, str, int], kind: str) -> str:
        endpoint_id, name, channels = endpoint
        if endpoint_id:
            return f"wasapi:{kind}:{endpoint_id}"
        metadata = f"{kind}|windows wasapi|{normalized_name(name)}|{channels}"
        return f"name:{hashlib.sha256(metadata.encode()).hexdigest()[:24]}"

    def _refresh_portaudio(self) -> None:
        if self._capture_active:
            raise AudioDeviceError("Le cache audio ne peut pas être réinitialisé pendant une capture.")
        # sounddevice exposes no public refresh. Its private lifecycle calls are
        # isolated here and verified by tests; only the application owns streams.
        terminate = getattr(self._sd, "_terminate", None)
        initialize = getattr(self._sd, "_initialize", None)
        if terminate is None or initialize is None:
            raise AudioDeviceError(
                "Cette version de PortAudio ne permet pas de rafraîchir les périphériques."
            )
        initialized = getattr(self._sd, "_initialized", 1)
        if initialized > 1:
            raise AudioDeviceError("Le cache PortAudio est partagé avec une autre session audio.")
        if initialized > 0:
            terminate()
        initialize()

    def _portaudio_inputs(self) -> list[dict[str, Any]]:
        hostapis = self._sd.query_hostapis()
        found = []
        for index, item in enumerate(self._sd.query_devices()):
            host_api = str(hostapis[int(item.get("hostapi", 0))].get("name", ""))
            if "WASAPI" not in host_api.upper() or int(item.get("max_input_channels", 0)) <= 0:
                continue
            found.append(
                dict(
                    index=index,
                    name=normalized_name(str(item["name"])),
                    channels=int(item["max_input_channels"]),
                    sample_rate=int(round(float(item.get("default_samplerate", 0)))),
                    host_api=host_api,
                )
            )
        return found

    def _input(
        self,
        endpoint: tuple[str, str, int],
        endpoints: list[tuple[str, str, int]],
        pa_inputs: list[dict[str, Any]],
        default_id: str,
    ) -> AudioDevice:
        endpoint_id, name, channels = endpoint
        # Both sides must be unique: otherwise choosing the first equal name
        # would silently substitute a different USB microphone after renumbering.
        same_endpoints = [item for item in endpoints if normalized_name(item[1]) == normalized_name(name)]
        matches = [item for item in pa_inputs if item["name"] == normalized_name(name)]
        if len(matches) > 1:
            matches = [item for item in matches if item["channels"] == channels]
        if len(same_endpoints) > 1:
            same_endpoints = [item for item in same_endpoints if item[2] == channels]
        match = matches[0] if len(matches) == 1 and len(same_endpoints) == 1 else None
        return AudioDevice(
            stable_key=self._key(endpoint, "input"),
            backend_id=endpoint_id or None,
            backend_index=match["index"] if match else None,
            name=name,
            kind="input",
            host_api=match["host_api"] if match else "Windows WASAPI",
            channels=channels,
            sample_rate=match["sample_rate"] if match else 0,
            is_default=bool(endpoint_id and endpoint_id == default_id),
        )
