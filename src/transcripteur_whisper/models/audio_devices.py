"""Immutable audio identities; backend indices are deliberately transient."""

from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class AudioDevice:
    stable_key: str
    backend_id: str | None
    backend_index: int | None
    name: str
    kind: Literal["input", "output"]
    host_api: str
    channels: int
    sample_rate: int
    is_default: bool = False


@dataclass(frozen=True)
class AudioDeviceSnapshot:
    inputs: tuple[AudioDevice, ...] = ()
    outputs: tuple[AudioDevice, ...] = ()

    @property
    def all(self) -> tuple[AudioDevice, ...]:
        return self.inputs + self.outputs

    def find(self, stable_key: str) -> AudioDevice | None:
        return next((item for item in self.all if item.stable_key == stable_key), None)


def select_device(devices: tuple[AudioDevice, ...], selected_key: str | None) -> str | None:
    """Keep an explicit choice, otherwise use Windows default then first device."""
    if any(device.stable_key == selected_key for device in devices):
        return selected_key
    selected = next((device for device in devices if device.is_default), None)
    selected = selected or next(iter(devices), None)
    return selected.stable_key if selected else None
