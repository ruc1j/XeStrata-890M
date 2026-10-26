#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Niko1221 and the Strata contributors
# SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
# SPDX-License-Identifier: LGPL-3.0-or-later AND MIT
"""A tiny terminal chat for a running Strata server (start it with run-<model>.sh first).

    python chat.py [--port 8095] [--think none|low|medium|high]

Type a message and press Enter.  /image <path> attaches a picture to your next message (when the server was set up
with images), /think <none|low|medium|high> sets how long the model thinks first, /reset starts a new conversation,
/quit leaves.  Standard library only.
"""
from __future__ import annotations

import argparse
import base64
import json
import mimetypes
import os
import sys
import time
import urllib.error
import urllib.request


class ServerError(Exception):
    pass


def http_error_text(e):
    """HTTP status and the server's own message (the JSON body's error.message, else the raw body)."""
    try:
        text = e.read().decode("utf-8", "replace").strip()
    except OSError:
        text = ""
    try:
        err = json.loads(text).get("error")
        text = (err.get("message") if isinstance(err, dict) else err) or text
    except (ValueError, AttributeError):
        pass
    return f"HTTP {e.code}: {text or e.reason}"


def stream(url, messages, think, max_tokens, api_key=None):
    body = {"model": "strata", "messages": messages, "stream": True, "max_tokens": max_tokens,
            "reasoning_effort": think}
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = "Bearer " + api_key
    req = urllib.request.Request(url, data=json.dumps(body).encode(), headers=headers)
    with urllib.request.urlopen(req, timeout=3600) as r:
        for raw in r:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data: ") or line == "data: [DONE]":
                continue
            try:
                chunk = json.loads(line[6:])
            except ValueError:
                continue
            if not isinstance(chunk, dict):
                continue
            if chunk.get("error"):                      # sent after the stream began (engine stopped, bad request)
                err = chunk["error"]
                raise ServerError(err.get("message") if isinstance(err, dict) else err)
            choices = chunk.get("choices")
            if not choices:
                continue
            d = choices[0].get("delta") or {}
            yield d.get("reasoning_content") or "", d.get("content") or ""


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port", type=int, default=8095)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--think", default="medium", choices=["none", "low", "medium", "high"],
                    help="how long the model thinks before answering (none = answer directly)")
    ap.add_argument("--no-think", action="store_true", help="same as --think none")
    ap.add_argument("--api-key", default=os.environ.get("STRATA_API_KEY"),
                    help="the server's API key, sent as a Bearer token (default: $STRATA_API_KEY)")
    ap.add_argument("--max-tokens", type=int, default=4096)
    a = ap.parse_args()
    url = f"http://{a.host}:{a.port}/v1/chat/completions"
    gray, reset = ("\033[90m", "\033[0m") if sys.stdout.isatty() else ("", "")
    messages, pending = [], []
    think = "none" if a.no_think else a.think
    print(f"Strata chat ({url}).  /image <path> = attach a picture, /think none|low|medium|high (now: {think}), "
          f"/reset = new conversation, /quit = leave.")
    while True:
        try:
            user = input("\nyou> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return 0
        if user in ("/quit", "/exit"):
            return 0
        if user == "/reset":
            messages, pending = [], []
            print("(new conversation)")
            continue
        if user.startswith("/think"):
            level = user[6:].strip().lower()
            if level in ("none", "low", "medium", "high"):
                think = level
                print(f"(thinking: {think})")
            else:
                print("(usage: /think none|low|medium|high)")
            continue
        if user.startswith("/image"):
            path = user[6:].strip().strip('"').strip("'")
            if not os.path.isfile(path):
                print(f"(no such file: {path or '?'} - usage: /image <path to a picture>)")
                continue
            mime = mimetypes.guess_type(path)[0] or "image/jpeg"
            data = base64.b64encode(open(path, "rb").read()).decode()
            pending.append({"type": "image_url", "image_url": {"url": f"data:{mime};base64,{data}"}})
            print(f"(picture attached: {os.path.basename(path)} - now type your question)")
            continue
        if not user:
            continue
        if pending:
            messages.append({"role": "user", "content": [*pending, {"type": "text", "text": user}]})
            pending = []
        else:
            messages.append({"role": "user", "content": user})
        answer, n, t0 = [], 0, time.time()
        print("model> ", end="", flush=True)
        failed = False
        try:
            in_think = False
            for reasoning, content in stream(url, messages, think, a.max_tokens, a.api_key):
                if reasoning:
                    if not in_think:
                        print(gray, end="")
                        in_think = True
                    print(reasoning, end="", flush=True)
                if content:
                    if in_think:
                        print(reset + "\n", end="")
                        in_think = False
                    print(content, end="", flush=True)
                    answer.append(content)
                n += 1
            if in_think:
                print(reset, end="")
        except KeyboardInterrupt:
            print(f"{reset}\n(interrupted)")
            failed = True
        except ServerError as e:
            print(f"{reset}\n(the server reports an error: {e})")
            failed = True
        except urllib.error.HTTPError as e:
            print(f"{reset}\n(the server answered {http_error_text(e)})")
            failed = True
        except OSError as e:
            print(f"{reset}\n(could not reach the server at {url}: {e})")
            failed = True
        if failed:
            messages.pop()                              # the unanswered question leaves the history
            continue
        dt = time.time() - t0
        print(f"\n{gray}[{n} chunks in {dt:.1f} s]{reset}")
        if answer:
            messages.append({"role": "assistant", "content": "".join(answer)})
        else:
            messages.pop()                              # only thinking came back: no empty assistant message
            print("(no answer, only thinking: ask again, or raise --max-tokens)")


if __name__ == "__main__":
    sys.exit(main())
