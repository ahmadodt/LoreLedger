from __future__ import annotations

import ctypes
import sys
from collections.abc import Iterator
from contextlib import contextmanager


ES_CONTINUOUS = 0x80000000
ES_SYSTEM_REQUIRED = 0x00000001


@contextmanager
def prevent_system_sleep() -> Iterator[None]:
    if sys.platform != "win32":
        yield
        return

    _set_thread_execution_state(ES_CONTINUOUS | ES_SYSTEM_REQUIRED)
    try:
        yield
    finally:
        _set_thread_execution_state(ES_CONTINUOUS)


def _set_thread_execution_state(flags: int) -> int:
    return int(ctypes.windll.kernel32.SetThreadExecutionState(flags))
