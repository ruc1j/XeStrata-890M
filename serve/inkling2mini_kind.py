# SPDX-FileCopyrightText: 2026 Niko1221 and the Strata contributors
# SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
# SPDX-License-Identifier: LGPL-3.0-or-later AND MIT
"""The Inkling-2-Mini engine as serve sees it: a program of its own (`strata-inkling-2-mini`) beside `strata`, with the
same `--serve` protocol, its DSpark drafter, the image reader inside the engine and 1 to 64 request slots
(`--parallel`).

From upstream Strata's serve/inkling_kind.py, with the model's name written out.  XeStrata has no `user_args` and no
second-GPU expert split for this engine, so engine_args takes the config's args as they are.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

ARCH = "inkling2_loop"
EXE_PREFIX = "strata-inkling-2-mini"
END_TEXTS = ("<|content_model_end_sampling|>", "<|endoftext|>")
IMAGE_PAD_TOKEN = "<|unused_200054|>"
REFUSED = {
    "logprobs": "logprobs: not supported by the Inkling-2-Mini engine",
    "logit_bias": "logit_bias: not supported by the Inkling-2-Mini engine",
}


def _base(exe) -> str:
    return str(exe or "").replace("\\", "/").rsplit("/", 1)[-1].lower()


def is_inkling2mini(cfg: dict | None) -> bool:
    if not isinstance(cfg, dict):
        return False
    if cfg.get("arch") == ARCH:
        return True
    return _base(cfg.get("exe")).startswith(EXE_PREFIX)


def is_inkling2mini_exe(exe) -> bool:
    return _base(exe).startswith(EXE_PREFIX)


NO_DRAFTER_NOTE = "Inkling-2-Mini: this engine has no DSpark drafter yet, so it starts without it"
NO_IMAGES_NOTE = "Inkling-2-Mini: this engine does not read images yet, so it starts without the image reader"
NO_IMAGES_REPLY = "this Inkling-2-Mini engine does not read images yet; update Strata or attach the Vision shard"
_PROBES: dict = {}


def wants_drafter(cfg: dict) -> bool:
    from serve import inkling2mini_family
    choice = str(cfg.get("drafting") or inkling2mini_family.DEFAULT_DRAFTER).lower()
    return choice != "off"


def parse_info(text: str) -> dict:
    for line in str(text or "").splitlines():
        if line.startswith("arch="):
            return dict(p.split("=", 1) for p in line.split() if "=" in p)
    return {}


def probe(exe: str | None, args: list, cfg: dict | None = None, timeout: float = 30.0) -> dict | None:
    import subprocess
    model = next((args[i + 1] for i, a in enumerate(args) if a in ("--native", "--model") and i + 1 < len(args)), None)
    if not exe or not model or not Path(str(exe)).is_file():
        return None
    try:
        st = Path(str(exe)).stat()
        key = (str(exe), st.st_mtime_ns, st.st_size, str(model))
    except OSError:
        return None
    if key in _PROBES:
        return _PROBES[key]
    from serve.childenv import quiet_crash_env
    env = quiet_crash_env(dict(os.environ))
    dirs = [d for d in ((cfg or {}).get("lib_dirs") or []) if Path(d).is_dir()]
    if dirs:
        var = "PATH" if os.name == "nt" else ("DYLD_LIBRARY_PATH" if sys.platform == "darwin" else "LD_LIBRARY_PATH")
        env[var] = os.pathsep.join(dirs + ([env[var]] if env.get(var) else []))
    try:
        done = subprocess.run([str(exe), "--info", str(model)], capture_output=True, text=True, errors="replace",
                              timeout=timeout, env=env, cwd=str(Path(str(exe)).parent))
        info = parse_info(done.stdout) if done.returncode == 0 else None
    except (OSError, subprocess.SubprocessError):
        info = None
    _PROBES[key] = info or None
    return _PROBES[key]


def drafter_args(cfg: dict, info: dict | None = None) -> list[str]:
    return ["--draft", "dspark"] if wants_drafter(cfg) and (info or {}).get("drafter") == "dspark" else []


def engine_args(cfg: dict, exe: str | None = None, *, announce: bool = True) -> list[str]:
    """The start arguments of `strata-inkling-2-mini`: the config's, `--parallel N` from the config's "parallel" (2 to
    64) where the args do not say it, `--draft dspark` where the engine has the drafter and the config did not turn it
    off, and without `--vision` where the engine does not read images."""
    from serve import inkling2mini_family
    args = list(cfg["args"])
    par = cfg.get("parallel")
    if (isinstance(par, int) and not isinstance(par, bool) and 1 < par <= inkling2mini_family.MAX_PARALLEL
            and "--parallel" not in args):
        args += ["--parallel", str(par)]
    wants_images = "--vision" in args
    need_draft = "--draft" not in args and wants_drafter(cfg)
    info = probe(exe, args, cfg) if (exe and (wants_images or need_draft)) else None
    if need_draft:
        add = drafter_args(cfg, info)
        args += add
        if not add and announce:
            print(f"[strata] {NO_DRAFTER_NOTE}", flush=True)
    if wants_images and (info or {}).get("images") != "1":
        at = args.index("--vision")
        del args[at:at + 2]
        if announce:
            print(f"[strata] {NO_IMAGES_NOTE}", flush=True)
    return args


def encode_path(path) -> str:
    return str(path).replace("%", "%25").replace(" ", "%20")


def decode_path(text: str) -> str:
    return str(text).replace("%20", " ").replace("%25", "%")


def write_images(sources, load, normalize, cancel=None, folder=None) -> list[str]:
    """Write normalized request images under unique per-request names (`strata-inkling-2-mini-images`)."""
    import hashlib
    import secrets
    import tempfile
    base = Path(folder) if folder else Path(tempfile.gettempdir()) / "strata-inkling-2-mini-images"
    base.mkdir(parents=True, exist_ok=True)
    nonce = secrets.token_hex(6)
    out: list[str] = []
    try:
        for idx, src in enumerate(sources):
            if cancel is not None and cancel.is_set():
                raise ValueError("the request was cancelled")
            data = normalize(src if isinstance(src, bytes) else load(src))
            ext = (".png" if data[:8] == b"\x89PNG\r\n\x1a\n" else ".jpg" if data[:3] == b"\xff\xd8\xff"
                   else ".bmp" if data[:2] == b"BM" else ".gif" if data[:3] == b"GIF" else ".img")
            p = base / f"{hashlib.sha256(data).hexdigest()[:24]}-{nonce}-{idx}{ext}"
            p.write_bytes(data)
            out.append(str(p))
    except Exception:
        for written in out:
            Path(written).unlink(missing_ok=True)
        raise
    return out


def refuse(engine, req: dict) -> None:
    from serve.server import top_logprobs
    if getattr(engine, "inkling2mini_engine", False) is True and top_logprobs(req) is not None:
        raise ValueError(REFUSED["logprobs"])


def literal_tags(tokenizer) -> list[str]:
    types = getattr(tokenizer, "token_types", None) or []
    toks = getattr(tokenizer, "tokens", None) or []
    return [toks[i] for i, ty in enumerate(types) if ty == 4 and i < len(toks)]


def end_ids(tokenizer) -> set[int]:
    ids = set()
    for text in END_TEXTS:
        got = tokenizer.encode(text, parse_special=True)
        if len(got) == 1:
            ids.add(int(got[0]))
    return ids


def model_path(cfg: dict) -> Path | None:
    args = cfg.get("args") or []
    for key in ("--native", "--model"):
        if key in args and args.index(key) + 1 < len(args):
            return Path(args[args.index(key) + 1])
    return None
