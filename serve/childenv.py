# SPDX-FileCopyrightText: 2026 Niko1221 and the Strata contributors
# SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
# SPDX-License-Identifier: LGPL-3.0-or-later AND MIT
"""The server never starts a debugger on a user's machine (upstream 38ad0bcbd, strata_control/childenv.py there).

When ggml aborts (a failed GGML_ASSERT, an uncaught C++ exception in a program that links ggml) it prints a backtrace
first: on Linux it forks `gdb --batch -ex attach <pid>` (then lldb, then backtrace_symbols), on Windows nothing.  A
debugger that attaches to a process the server is about to end writes its own errors into the log ("Couldn't get
registers: No such process") and, because it inherits the picture reader's stdout, into the pipe the server reads the
reader's READY line from.  `GGML_NO_BACKTRACE` (any value) switches it all off.  Every child the server starts (the
engine, strata-vision, strata-image) gets it, unless a developer asks for the backtrace with STRATA_BACKTRACE=1 (then
nothing is set here, and the environment is left as the developer made it).
"""
from __future__ import annotations

import os
import sys

NO_BACKTRACE = "GGML_NO_BACKTRACE"
DEVELOPER_SWITCH = "STRATA_BACKTRACE"


def developer_wants_backtrace(env=None) -> bool:
    """STRATA_BACKTRACE is set to anything but empty / 0 / no / off / false, in `env` or in this process's
    environment."""
    for source in (env, os.environ):
        value = str((source or {}).get(DEVELOPER_SWITCH, "")).strip().lower()
        if value and value not in ("0", "no", "off", "false"):
            return True
    return False


def quiet_crash_env(env: dict | None = None):
    """`env` (this process's own environment when None) with GGML_NO_BACKTRACE=1 set unless it is already there or the
    developer switch is on.  Changes the mapping it is given and returns it, so it fits where an environment is built:
    `env = quiet_crash_env(dict(os.environ))`."""
    target = os.environ if env is None else env
    if NO_BACKTRACE not in target and not developer_wants_backtrace(target):
        target[NO_BACKTRACE] = "1"
    return target


# upstream aeecf0f5d: glibc gives every thread that allocates its own memory arena, up to eight per core, and a server
# that starts a thread per request keeps a little of each.  This fork caps the server's own arenas only: the engine is
# not given MALLOC_ARENA_MAX (its CPU expert threads allocate side by side), so its environment is left as it is.
ARENA_VAR = "MALLOC_ARENA_MAX"
ARENA_VALUE = 2
M_ARENA_MAX = -8                                    # glibc's mallopt(3) parameter number


def limit_own_arenas() -> bool:
    """mallopt(M_ARENA_MAX, 2) for this process (glibc takes it while the process runs).  False, and nothing done, when
    MALLOC_ARENA_MAX was set (the person's value wins), off Linux, or when the C library has no such call (musl).  It
    sets no environment variable, so the children the server starts keep glibc's default."""
    if not sys.platform.startswith("linux") or ARENA_VAR in os.environ:
        return False
    try:
        import ctypes
        return bool(ctypes.CDLL(None).mallopt(M_ARENA_MAX, ARENA_VALUE))
    except (OSError, AttributeError, ValueError):
        return False
