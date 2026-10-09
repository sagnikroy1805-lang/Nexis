"""Keep Windows from idle-sleeping while a given process is alive.

Uses SetThreadExecutionState, the per-process request media players use. It
changes no power settings and is released automatically when this script
exits (the request belongs to this process). A closed lid or a manual Sleep
still sleeps the machine.

Usage:
    python scripts/keep_awake.py <pid>
"""

from __future__ import annotations

import ctypes
import sys
import time

ES_CONTINUOUS = 0x80000000
ES_SYSTEM_REQUIRED = 0x00000001
SYNCHRONIZE = 0x00100000
WAIT_TIMEOUT = 0x102


def main() -> None:
    pid = int(sys.argv[1])
    kernel32 = ctypes.windll.kernel32
    handle = kernel32.OpenProcess(SYNCHRONIZE, False, pid)
    if not handle:
        sys.exit(f"process {pid} not found")
    kernel32.SetThreadExecutionState(ES_CONTINUOUS | ES_SYSTEM_REQUIRED)
    print(f"keeping the system awake while process {pid} runs", flush=True)
    try:
        while kernel32.WaitForSingleObject(handle, 60_000) == WAIT_TIMEOUT:
            time.sleep(0)
    finally:
        kernel32.SetThreadExecutionState(ES_CONTINUOUS)
        kernel32.CloseHandle(handle)
    print(f"process {pid} finished; sleep allowed again", flush=True)


if __name__ == "__main__":
    main()
