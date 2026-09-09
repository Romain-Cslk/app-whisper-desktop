# -*- mode: python ; coding: utf-8 -*-
import ast
from pathlib import Path

from PyInstaller.utils.hooks import collect_data_files, collect_dynamic_libs, collect_submodules
from PyInstaller.utils.win32.versioninfo import FixedFileInfo, StringFileInfo, StringStruct, StringTable, VarFileInfo, VarStruct, VSVersionInfo

root = Path(SPECPATH).parent
tree = ast.parse((root / "src/transcripteur_whisper/__init__.py").read_text(encoding="utf-8"))
version = next(ast.literal_eval(node.value) for node in tree.body if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "__version__" for t in node.targets))
version_tuple = tuple(map(int, version.split("."))) + (0,)
datas = [(str(root / "src/transcripteur_whisper/assets"), "transcripteur_whisper/assets")]
binaries = []
hiddenimports = [
    "PySide6.QtMultimedia",
    "soundcard.mediafoundation",
    "_cffi_backend",
    "onnxruntime",
    "tokenizers",
    "backports",
    "backports.tarfile",
    "sherpa_onnx",
]
for package in (
    "faster_whisper",
    "ctranslate2",
    "av",
    "soundcard",
    "_sounddevice_data",
    "_soundfile_data",
    "onnxruntime",
    "sherpa_onnx",
):
    datas += collect_data_files(package)
    binaries += collect_dynamic_libs(package)
hiddenimports += collect_submodules("faster_whisper")
hiddenimports += collect_submodules("sherpa_onnx")
info = VSVersionInfo(
    ffi=FixedFileInfo(filevers=version_tuple, prodvers=version_tuple, mask=0x3f, flags=0, OS=0x40004, fileType=1, subtype=0, date=(0, 0)),
    kids=[StringFileInfo([StringTable("040C04B0", [
        StringStruct("FileDescription", "Transcripteur Whisper"),
        StringStruct("FileVersion", version),
        StringStruct("ProductName", "Transcripteur Whisper"),
        StringStruct("ProductVersion", version),
        StringStruct("OriginalFilename", "TranscripteurWhisper.exe"),
    ])]), VarFileInfo([VarStruct("Translation", [1036, 1200])])],
)
a = Analysis(
    [str(root / "packaging/launcher.py")],
    pathex=[str(root / "src")], binaries=binaries, datas=datas, hiddenimports=hiddenimports,
    hookspath=[], runtime_hooks=[],
    excludes=["tkinter", "PyQt5", "PyQt6", "PySide2", "PySide6.QtWebEngineCore", "PySide6.QtWebEngineWidgets", "PySide6.QtWebEngineQuick", "fastapi", "starlette", "uvicorn", "jinja2", "webview", "torch", "tensorflow", "matplotlib", "pandas", "scipy", "pytest", "_pytest", "pytestqt"],
    noarchive=False,
)
pyz = PYZ(a.pure)
exe = EXE(pyz, a.scripts, [], exclude_binaries=True, name="TranscripteurWhisper", debug=False,
          bootloader_ignore_signals=False, strip=False, upx=False, console=False,
          icon=str(root / "src/transcripteur_whisper/assets/icon.ico"), version=info)
coll = COLLECT(exe, a.binaries, a.datas, strip=False, upx=False, name="TranscripteurWhisper")
