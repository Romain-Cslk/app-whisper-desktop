"""Windows DPAPI protection for local biometric speaker profile data."""
from __future__ import annotations

import ctypes
import sys
from ctypes import wintypes


class SecureStorageError(RuntimeError):
    pass


class _DataBlob(ctypes.Structure):
    _fields_ = [("cbData", wintypes.DWORD), ("pbData", ctypes.POINTER(ctypes.c_ubyte))]


class WindowsDpapiCodec:
    """Encrypt/decrypt bytes for the current Windows user using DPAPI."""

    _FLAGS = 0x1  # CRYPTPROTECT_UI_FORBIDDEN
    _ENTROPY = b"TranscripteurWhisper.SpeakerProfiles.v1"

    def __init__(self) -> None:
        if sys.platform != "win32":
            raise SecureStorageError("Le stockage biométrique chiffré nécessite Windows.")
        self._crypt32 = ctypes.WinDLL("crypt32", use_last_error=True)
        self._kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        self._crypt32.CryptProtectData.argtypes = [
            ctypes.POINTER(_DataBlob), wintypes.LPCWSTR, ctypes.POINTER(_DataBlob),
            ctypes.c_void_p, ctypes.c_void_p, wintypes.DWORD, ctypes.POINTER(_DataBlob),
        ]
        self._crypt32.CryptProtectData.restype = wintypes.BOOL
        self._crypt32.CryptUnprotectData.argtypes = [
            ctypes.POINTER(_DataBlob), ctypes.POINTER(wintypes.LPWSTR), ctypes.POINTER(_DataBlob),
            ctypes.c_void_p, ctypes.c_void_p, wintypes.DWORD, ctypes.POINTER(_DataBlob),
        ]
        self._crypt32.CryptUnprotectData.restype = wintypes.BOOL
        self._kernel32.LocalFree.argtypes = [ctypes.c_void_p]
        self._kernel32.LocalFree.restype = ctypes.c_void_p

    @staticmethod
    def _input_blob(data: bytes) -> tuple[_DataBlob, ctypes.Array]:
        raw = bytes(data)
        buffer = ctypes.create_string_buffer(raw, len(raw) or 1)
        blob = _DataBlob(len(raw), ctypes.cast(buffer, ctypes.POINTER(ctypes.c_ubyte)))
        return blob, buffer

    def _entropy_blob(self) -> tuple[_DataBlob, ctypes.Array]:
        return self._input_blob(self._ENTROPY)

    def protect(self, data: bytes) -> bytes:
        if not data:
            raise SecureStorageError("Le contenu à chiffrer est vide.")
        source, source_buffer = self._input_blob(data)
        entropy, entropy_buffer = self._entropy_blob()
        destination = _DataBlob()
        # Keep source buffers alive until CryptProtectData returns.
        _ = (source_buffer, entropy_buffer)
        if not self._crypt32.CryptProtectData(
            ctypes.byref(source), "Transcripteur Whisper speaker profiles",
            ctypes.byref(entropy), None, None, self._FLAGS, ctypes.byref(destination)
        ):
            raise SecureStorageError(f"DPAPI CryptProtectData a échoué ({ctypes.get_last_error()}).")
        try:
            return ctypes.string_at(destination.pbData, destination.cbData)
        finally:
            if destination.pbData:
                self._kernel32.LocalFree(ctypes.cast(destination.pbData, ctypes.c_void_p))

    def unprotect(self, data: bytes) -> bytes:
        if not data:
            raise SecureStorageError("Le fichier biométrique chiffré est vide.")
        source, source_buffer = self._input_blob(data)
        entropy, entropy_buffer = self._entropy_blob()
        destination = _DataBlob()
        description = wintypes.LPWSTR()
        _ = (source_buffer, entropy_buffer)
        if not self._crypt32.CryptUnprotectData(
            ctypes.byref(source), ctypes.byref(description), ctypes.byref(entropy),
            None, None, self._FLAGS, ctypes.byref(destination)
        ):
            raise SecureStorageError(
                "Impossible de déchiffrer les profils vocaux pour cet utilisateur Windows."
            )
        try:
            return ctypes.string_at(destination.pbData, destination.cbData)
        finally:
            if destination.pbData:
                self._kernel32.LocalFree(ctypes.cast(destination.pbData, ctypes.c_void_p))
            if description:
                self._kernel32.LocalFree(ctypes.cast(description, ctypes.c_void_p))
