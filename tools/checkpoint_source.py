# SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
# SPDX-License-Identifier: LGPL-3.0-or-later
"""tools/checkpoint_source.py - a Hugging Face checkpoint's files, read from a local directory or straight from the
repository with HTTP range requests (tools/ple_fp8_pack.py, tools/embd_bf16_pack.py: they need a few tensors out of
files of tens of GB)."""
import json
import struct
import sys
import time
import urllib.request

ENDPOINT = "https://huggingface.co"


class Source:
    """The checkpoint's files: a local directory, or a Hugging Face repository read with HTTP range requests."""

    def __init__(self, model=None, repo=None):
        self.model, self.repo = model, repo
        self.name = model.name if model else repo

    def read(self, name, off, n):
        if self.model is not None:
            with open(self.model / name, "rb") as f:
                f.seek(off)
                return f.read(n)
        url = "%s/%s/resolve/main/%s" % (ENDPOINT, self.repo, name)
        for attempt in range(8):
            try:
                req = urllib.request.Request(url, headers={"Range": "bytes=%d-%d" % (off, off + n - 1)})
                with urllib.request.urlopen(req, timeout=120) as r:
                    # a range read is only the range asked for when the server says so (206 and its Content-Range);
                    # a mirror or proxy that ignores Range answers 200 with the whole file, which is not read
                    want = "bytes %d-%d/" % (off, off + n - 1)
                    if r.status != 206 or not (r.headers.get("Content-Range") or "").startswith(want):
                        raise OSError("the server ignored the Range header (HTTP %d)" % r.status)
                    b = r.read(n)
                if len(b) == n:
                    return b
            except OSError as e:
                print("  retry %d: %s" % (attempt + 1, e), flush=True)
            time.sleep(2 * (attempt + 1))
        sys.exit("cannot read %d bytes at %d of %s" % (n, off, url))

    def text(self, name):
        if self.model is not None:
            return (self.model / name).read_text(encoding="utf-8")
        url = "%s/%s/resolve/main/%s" % (ENDPOINT, self.repo, name)
        with urllib.request.urlopen(url, timeout=120) as r:
            return r.read().decode("utf-8")


def safetensors_header(src, name):
    n = struct.unpack("<Q", src.read(name, 0, 8))[0]
    return json.loads(src.read(name, 8, n)), 8 + n
