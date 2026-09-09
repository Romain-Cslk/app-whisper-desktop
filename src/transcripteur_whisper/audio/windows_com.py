"""Initialize COM on every thread using SoundCard's Windows backend."""

import ctypes
import sys
from contextlib import contextmanager


@contextmanager
def com_apartment():
    """Balance COM on this thread, accepting a pre-existing STA apartment."""
    if sys.platform != "win32":
        yield
        return
    ole32 = ctypes.OleDLL("ole32")
    # OleDLL raises for failed HRESULTs, including an existing STA apartment.
    initialized = False
    try:
        ole32.CoInitializeEx(None, 0)
        initialized = True
    except OSError as exc:
        if (exc.winerror or 0) & 0xFFFFFFFF != 0x80010106:
            raise
    try:
        yield
    finally:
        if initialized:
            ole32.CoUninitialize()
