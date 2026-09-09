from __future__ import annotations

import json
import os
import sysconfig
from pathlib import Path

from transcripteur_whisper.services.compute_backend import prepare_sherpa_windows_runtime


def version_info(path: Path) -> dict[str, str | bool]:
    result: dict[str, str | bool] = {"path": str(path), "exists": path.is_file()}
    if path.is_file() and os.name == "nt":
        try:
            import ctypes
            from ctypes import wintypes
            size = ctypes.windll.version.GetFileVersionInfoSizeW(str(path), None)
            if size:
                buf = ctypes.create_string_buffer(size)
                ctypes.windll.version.GetFileVersionInfoW(str(path), 0, size, buf)
                lp = ctypes.c_void_p()
                length = wintypes.UINT()
                ctypes.windll.version.VerQueryValueW(buf, "\\", ctypes.byref(lp), ctypes.byref(length))
                class VS_FIXEDFILEINFO(ctypes.Structure):
                    _fields_ = [("dwSignature", wintypes.DWORD),("dwStrucVersion", wintypes.DWORD),
                                ("dwFileVersionMS", wintypes.DWORD),("dwFileVersionLS", wintypes.DWORD),
                                ("dwProductVersionMS", wintypes.DWORD),("dwProductVersionLS", wintypes.DWORD),
                                ("dwFileFlagsMask", wintypes.DWORD),("dwFileFlags", wintypes.DWORD),
                                ("dwFileOS", wintypes.DWORD),("dwFileType", wintypes.DWORD),
                                ("dwFileSubtype", wintypes.DWORD),("dwFileDateMS", wintypes.DWORD),
                                ("dwFileDateLS", wintypes.DWORD)]
                ffi = ctypes.cast(lp, ctypes.POINTER(VS_FIXEDFILEINFO)).contents
                result["file_version"] = ".".join(map(str, [ffi.dwFileVersionMS >> 16, ffi.dwFileVersionMS & 0xFFFF,
                                                               ffi.dwFileVersionLS >> 16, ffi.dwFileVersionLS & 0xFFFF]))
        except Exception as exc:
            result["version_error"] = str(exc)
    return result


def main() -> int:
    purelib = Path(sysconfig.get_paths()["purelib"])
    sherpa_ort = purelib / "sherpa_onnx" / "lib" / "onnxruntime.dll"
    system_ort = Path(os.environ.get("WINDIR", r"C:\Windows")) / "System32" / "onnxruntime.dll"
    resolved = prepare_sherpa_windows_runtime("cuda")
    print(json.dumps({
        "system32_onnxruntime": version_info(system_ort),
        "sherpa_bundled_onnxruntime": version_info(sherpa_ort),
        "worker_forced_onnxruntime": str(resolved) if resolved else None,
        "ok": resolved is not None and Path(resolved).resolve() == sherpa_ort.resolve(),
    }, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
