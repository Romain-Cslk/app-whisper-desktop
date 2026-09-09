"""Validation and deterministic Windows CUDA DLL preloading for local compute."""
from __future__ import annotations

import ctypes
import os
import platform
import sysconfig
from importlib.metadata import PackageNotFoundError, version
from dataclasses import dataclass
from pathlib import Path
from typing import Any

VALID_COMPUTE_DEVICES = {"cpu", "cuda"}
GPU_REPAIR_VERSION = "1.0.7"
_DLL_HANDLES: list[Any] = []
_DLL_DIR_HANDLES: list[Any] = []
_CUDA_PRELOADED = False


class ComputeBackendError(RuntimeError):
    """Raised when the requested local compute backend cannot be used."""


@dataclass(frozen=True)
class ComputeBackendInfo:
    device: str
    whisper_compute_type: str
    sherpa_provider: str
    cuda_device_count: int = 0
    sherpa_version: str = ""


def validate_compute_device(device: str) -> str:
    value = str(device or "cpu").strip().lower()
    if value not in VALID_COMPUTE_DEVICES:
        raise ValueError("Calcul local : choisissez CPU ou GPU NVIDIA CUDA.")
    return value


def _nvidia_component_directories() -> dict[str, Path]:
    purelib = Path(sysconfig.get_paths()["purelib"])
    root = purelib / "nvidia"
    candidates = {
        "cuda_runtime": root / "cuda_runtime" / "bin",
        "cublas": root / "cublas" / "bin",
        "cufft": root / "cufft" / "bin",
        "cudnn": root / "cudnn" / "bin",
        "nvjitlink": root / "nvjitlink" / "bin",
        "cuda_nvrtc": root / "cuda_nvrtc" / "bin",
        "curand": root / "curand" / "bin",
    }
    return {name: path.resolve() for name, path in candidates.items() if path.is_dir()}


def configure_cuda_dll_search() -> tuple[Path, ...]:
    """Expose venv-local NVIDIA DLL directories to this process and child workers."""
    if os.name != "nt":
        return ()
    directories = tuple(_nvidia_component_directories().values())
    existing = [item for item in os.environ.get("PATH", "").split(os.pathsep) if item]
    normalized = {os.path.normcase(os.path.abspath(item)) for item in existing}
    prefix: list[str] = []
    for directory in directories:
        value = str(directory)
        key = os.path.normcase(os.path.abspath(value))
        if key not in normalized:
            prefix.append(value)
            normalized.add(key)
        if hasattr(os, "add_dll_directory"):
            try:
                _DLL_DIR_HANDLES.append(os.add_dll_directory(value))
            except OSError:
                pass
    if prefix:
        os.environ["PATH"] = os.pathsep.join(prefix + existing)
    return directories


def _load_system_msvc() -> None:
    if os.name != "nt":
        return
    names = ["vcruntime140.dll", "msvcp140.dll"]
    if platform.machine().upper() != "ARM64":
        names.append("vcruntime140_1.dll")
    for name in names:
        try:
            _DLL_HANDLES.append(ctypes.CDLL(name))
        except OSError as exc:
            raise ComputeBackendError(
                f"Runtime Microsoft Visual C++ manquant ou invalide : {name}. "
                "Installez Microsoft Visual C++ 2015-2022 Redistributable x64. "
                f"Détail : {exc}"
            ) from exc


def _required_preload_sequence() -> tuple[tuple[str, str], ...]:
    """Windows CUDA/cuDNN preload order used by ONNX Runtime 1.23.x, plus optional JIT deps."""
    return (
        ("cublas", "cublasLt64_12.dll"),
        ("cublas", "cublas64_12.dll"),
        ("cufft", "cufft64_11.dll"),
        ("cuda_runtime", "cudart64_12.dll"),
        # These two are not direct ORT Windows preload entries, but cuDNN runtime
        # compiled engines can resolve them dynamically. Load them when installed.
        ("nvjitlink", "nvJitLink_120_0.dll"),
        ("cuda_nvrtc", "nvrtc64_120_0.dll"),
        ("curand", "curand64_10.dll"),
        # Exact cuDNN 9 preload set used by ONNX Runtime 1.23.2 on Windows.
        ("cudnn", "cudnn_engines_runtime_compiled64_9.dll"),
        ("cudnn", "cudnn_engines_precompiled64_9.dll"),
        ("cudnn", "cudnn_heuristic64_9.dll"),
        ("cudnn", "cudnn_ops64_9.dll"),
        ("cudnn", "cudnn_adv64_9.dll"),
        ("cudnn", "cudnn_graph64_9.dll"),
        ("cudnn", "cudnn64_9.dll"),
    )


def preload_sherpa_cuda_dependencies(provider: str = "cuda") -> dict[str, Path]:
    """Preload CUDA/cuDNN DLLs before importing/initializing sherpa-onnx.

    Shared ONNX Runtime provider DLLs themselves are intentionally NOT loaded here:
    ONNX Runtime must load its own provider library when the CUDA provider is added.
    """
    global _CUDA_PRELOADED
    value = str(provider or "cpu").lower()
    if value != "cuda" or os.name != "nt":
        return {}

    configure_cuda_dll_search()
    _load_system_msvc()
    directories = _nvidia_component_directories()

    required_components = {"cuda_runtime", "cublas", "cufft", "cudnn"}
    missing_components = sorted(required_components - directories.keys())
    if missing_components:
        raise ComputeBackendError(
            "Runtime CUDA incomplet dans le .venv : " + ", ".join(missing_components)
        )

    loaded: dict[str, Path] = {}
    for component, name in _required_preload_sequence():
        directory = directories.get(component)
        # JIT/curand are supplementary: if the package is absent we let the strict
        # package validator report it. Core ORT/cuDNN entries are mandatory.
        optional_component = component in {"nvjitlink", "cuda_nvrtc", "curand"}
        if directory is None:
            if optional_component:
                continue
            raise ComputeBackendError(f"Runtime CUDA incomplet : composant {component} absent.")
        path = directory / name
        if not path.is_file():
            if optional_component:
                continue
            raise ComputeBackendError(f"Runtime CUDA incomplet : {name} est absent de {directory}.")
        try:
            _DLL_HANDLES.append(ctypes.CDLL(str(path)))
        except OSError as exc:
            raise ComputeBackendError(
                f"La DLL {name} existe mais son chargement Windows échoue. Détail : {exc}"
            ) from exc
        loaded[name] = path

    _CUDA_PRELOADED = True
    return loaded




def _sherpa_internal_lib_directory() -> Path:
    purelib = Path(sysconfig.get_paths()["purelib"])
    directory = (purelib / "sherpa_onnx" / "lib").resolve()
    if not directory.is_dir():
        raise ComputeBackendError(
            f"Le dossier natif sherpa-onnx est introuvable : {directory}"
        )
    return directory


def _sherpa_onnxruntime_path() -> Path:
    """Return the ORT DLL shipped by the *CUDA sherpa wheel* on Windows.

    sherpa-onnx's Windows wheel installs onnxruntime.dll as a data_file in the
    environment Scripts directory, not in site-packages/sherpa_onnx/lib.  The
    CUDA 12/cuDNN 9 wheel for sherpa-onnx 1.13.7 is built against ORT 1.24.4.
    """
    if os.name != "nt":
        raise ComputeBackendError("La résolution du runtime ONNX sherpa est spécifique à Windows.")

    candidates: list[Path] = []
    scripts = Path(sysconfig.get_paths()["scripts"]).resolve()
    candidates.append(scripts / "onnxruntime.dll")

    # In a venv, sys.executable is normally <venv>/Scripts/python.exe. Keep this
    # second candidate because some Python/uv layouts report a different scripts path.
    import sys

    candidates.append(Path(sys.executable).resolve().parent / "onnxruntime.dll")

    # Compatibility fallback for any future wheel layout that places it beside
    # the native sherpa libraries. This is not the layout of 1.13.7 on Windows.
    try:
        candidates.append(_sherpa_internal_lib_directory() / "onnxruntime.dll")
    except ComputeBackendError:
        pass

    seen: set[str] = set()
    for candidate in candidates:
        key = os.path.normcase(os.path.abspath(str(candidate)))
        if key in seen:
            continue
        seen.add(key)
        if candidate.is_file():
            return candidate.resolve()

    pretty = "\n - ".join(str(item) for item in candidates)
    raise ComputeBackendError(
        "Le onnxruntime.dll fourni par le wheel sherpa-onnx CUDA est absent. "
        "Le wheel doit être réinstallé. Emplacements vérifiés :\n - " + pretty
    )


def _module_path_from_handle(handle: int) -> Path | None:
    if os.name != "nt" or not handle:
        return None
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.GetModuleFileNameW.argtypes = [ctypes.c_void_p, ctypes.c_wchar_p, ctypes.c_uint32]
    kernel32.GetModuleFileNameW.restype = ctypes.c_uint32
    buffer = ctypes.create_unicode_buffer(32768)
    length = kernel32.GetModuleFileNameW(ctypes.c_void_p(handle), buffer, len(buffer))
    if not length:
        return None
    return Path(buffer.value).resolve()


def _loaded_module_path(name: str) -> Path | None:
    if os.name != "nt":
        return None
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.GetModuleHandleW.argtypes = [ctypes.c_wchar_p]
    kernel32.GetModuleHandleW.restype = ctypes.c_void_p
    handle = kernel32.GetModuleHandleW(name)
    return _module_path_from_handle(int(handle or 0))


def _ort_api_info(handle: Any) -> tuple[str, bool]:
    """Read the ORT version string and confirm API v24 from an already loaded DLL."""
    if os.name != "nt":
        return "", False

    class OrtApiBase(ctypes.Structure):
        _fields_ = [("GetApi", ctypes.c_void_p), ("GetVersionString", ctypes.c_void_p)]

    try:
        handle.OrtGetApiBase.argtypes = []
        handle.OrtGetApiBase.restype = ctypes.POINTER(OrtApiBase)
        base = handle.OrtGetApiBase()
        if not base:
            raise RuntimeError("OrtGetApiBase a renvoyé NULL")
        get_version = ctypes.WINFUNCTYPE(ctypes.c_char_p)(base.contents.GetVersionString)
        raw = get_version()
        version_text = raw.decode("ascii", errors="replace") if raw else ""
        get_api = ctypes.WINFUNCTYPE(ctypes.c_void_p, ctypes.c_uint32)(base.contents.GetApi)
        api24 = bool(get_api(24))
        return version_text, api24
    except Exception as exc:
        raise ComputeBackendError(
            f"Impossible d'interroger l'API du onnxruntime.dll sherpa chargé. Détail : {exc}"
        ) from exc


def prepare_sherpa_windows_runtime(provider: str = "cuda") -> Path | None:
    """Preload sherpa's exact ORT 1.24.4 DLL before importing sherpa_onnx.

    The sherpa-onnx 1.13.7 Windows CUDA12/cuDNN9 wheel is built against ONNX
    Runtime 1.24.4 (ORT API 24) and installs its onnxruntime.dll in the venv
    Scripts directory.  Some Windows machines also contain stale ORT 1.17.1 in
    C:\\Windows\\System32; if that DLL is resolved first, sherpa fails with
    "requested API version 24 ... Current ORT Version is 1.17.1".

    This function is called only in the isolated diarization worker, before the
    first import of sherpa_onnx. It loads the exact wheel DLL by full path and
    verifies both its version and API level before sherpa can initialize.
    """
    value = str(provider or "cpu").lower()
    if value != "cuda" or os.name != "nt":
        return None

    preload_sherpa_cuda_dependencies("cuda")
    lib = _sherpa_internal_lib_directory()
    bundled = _sherpa_onnxruntime_path()
    scripts = bundled.parent

    # Keep both the wheel's Scripts directory (base ORT DLL) and sherpa's private
    # lib directory (provider DLLs) visible to the Windows loader and child code.
    for directory in (scripts, lib):
        if hasattr(os, "add_dll_directory"):
            try:
                _DLL_DIR_HANDLES.append(os.add_dll_directory(str(directory)))
            except OSError:
                pass

    current = [item for item in os.environ.get("PATH", "").split(os.pathsep) if item]
    priority = [str(scripts), str(lib)]
    keys = {os.path.normcase(os.path.abspath(item)) for item in priority}
    current = [
        item for item in current
        if os.path.normcase(os.path.abspath(item)) not in keys
    ]
    os.environ["PATH"] = os.pathsep.join(priority + current)

    already = _loaded_module_path("onnxruntime.dll")
    if already is not None and os.path.normcase(str(already)) != os.path.normcase(str(bundled)):
        raise ComputeBackendError(
            "Un ONNX Runtime incompatible est déjà chargé dans le worker avant sherpa-onnx : "
            f"{already}. Le worker doit démarrer sans ORT puis charger {bundled}."
        )

    if already is None:
        try:
            handle = ctypes.WinDLL(str(bundled))
            _DLL_HANDLES.append(handle)
        except OSError as exc:
            raise ComputeBackendError(
                f"Le ONNX Runtime 1.24.4 du wheel sherpa CUDA ne peut pas être chargé : "
                f"{bundled}. Détail : {exc}"
            ) from exc
    else:
        # Retrieve a ctypes handle for API/version interrogation without changing
        # which DLL is selected; the exact path has already been verified above.
        handle = ctypes.WinDLL(str(bundled))
        _DLL_HANDLES.append(handle)

    resolved = _loaded_module_path("onnxruntime.dll")
    if resolved is None:
        raise ComputeBackendError("onnxruntime.dll est chargé mais Windows ne retourne pas son chemin.")
    if os.path.normcase(str(resolved)) != os.path.normcase(str(bundled)):
        raise ComputeBackendError(
            "Windows a résolu le mauvais onnxruntime.dll : "
            f"{resolved} (attendu {bundled})."
        )

    ort_version, api24 = _ort_api_info(handle)
    if ort_version != "1.24.4":
        raise ComputeBackendError(
            "Le wheel sherpa CUDA n'expose pas le runtime ONNX attendu : "
            f"version chargée {ort_version or 'inconnue'}, attendu 1.24.4, fichier {bundled}."
        )
    if not api24:
        raise ComputeBackendError(
            f"Le runtime {ort_version} chargé depuis {bundled} ne fournit pas ORT API 24."
        )
    return resolved

def verify_cuda_runtime_dlls() -> dict[str, Path]:
    return preload_sherpa_cuda_dependencies("cuda")


def _load_ctranslate2() -> Any:
    preload_sherpa_cuda_dependencies("cuda")
    import ctranslate2
    return ctranslate2


def _load_sherpa_onnx() -> Any:
    preload_sherpa_cuda_dependencies("cuda")
    import sherpa_onnx
    return sherpa_onnx


def _module_version(module: Any) -> str:
    value = getattr(module, "__version__", None)
    if value is None:
        value = getattr(module, "version", "")
    if callable(value):
        try:
            value = value()
        except TypeError:
            pass
    return str(value or "")


def _choose_cuda_compute_type(ctranslate2: Any) -> str:
    try:
        supported = set(ctranslate2.get_supported_compute_types("cuda"))
    except Exception as exc:
        raise ComputeBackendError(
            "Le GPU NVIDIA est détecté mais CTranslate2 ne peut pas interroger ses capacités CUDA. "
            f"Détail : {exc}"
        ) from exc
    for candidate in ("float16", "int8_float16", "int8", "float32"):
        if candidate in supported:
            return candidate
    raise ComputeBackendError("Aucun type de calcul CUDA compatible n'est disponible pour Whisper.")


def validate_compute_backend(device: str, *, require_diarization: bool = False) -> ComputeBackendInfo:
    value = validate_compute_device(device)
    if value == "cpu":
        return ComputeBackendInfo("cpu", "int8", "cpu")

    preload_sherpa_cuda_dependencies("cuda")
    try:
        ctranslate2 = _load_ctranslate2()
        count = int(ctranslate2.get_cuda_device_count())
    except ComputeBackendError:
        raise
    except Exception as exc:
        raise ComputeBackendError(f"CUDA n'est pas utilisable par CTranslate2. Détail : {exc}") from exc
    if count < 1:
        raise ComputeBackendError("Aucun GPU NVIDIA CUDA utilisable n'est détecté.")
    compute_type = _choose_cuda_compute_type(ctranslate2)

    sherpa_version = ""
    if require_diarization:
        try:
            sherpa_version = version("sherpa-onnx")
        except PackageNotFoundError as exc:
            raise ComputeBackendError("sherpa-onnx n'est pas installe.") from exc
        if "+cuda" not in sherpa_version.lower():
            raise ComputeBackendError(
                "La diarisation est installée en version CPU. Appliquez le correctif GPU v1.0.7."
            )
        # Do not import sherpa_onnx in the main process. The isolated worker
        # forces sherpa's own onnxruntime.dll before importing sherpa and then
        # validates the CUDA provider by real inference.

    return ComputeBackendInfo(
        "cuda",
        compute_type,
        "cuda",
        cuda_device_count=count,
        sherpa_version=sherpa_version,
    )
