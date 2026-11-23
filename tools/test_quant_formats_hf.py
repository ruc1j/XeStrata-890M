# SPDX-FileCopyrightText: 2026 Niko1221 and the Strata contributors
# SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
# SPDX-License-Identifier: LGPL-3.0-or-later AND MIT
"""quant_formats.py --hf reads each shard's header over HTTP range requests with gguf_reader's bounds checks (#5075):
the header knows the file's size from the Hugging Face tree, so a complete header parses and a cut one is refused with
a ValueError that names the file.  Over a minimal GGUF v3 header written here (no network, no model).

    python -m unittest tools.test_quant_formats_hf
"""
from __future__ import annotations

import io
import json
import re
import struct
import sys
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent))
import quant_formats as Q  # noqa: E402


def gstr(s: bytes) -> bytes:
    return struct.pack("<Q", len(s)) + s


def shard() -> bytes:
    """One Q8_0 [32, 4] tensor `blk.0.ffn_down_exps.weight` (type 8), data at offset 0 after a 32-byte alignment."""
    head = struct.pack("<IIQQ", 0x46554747, 3, 1, 0) + gstr(b"blk.0.ffn_down_exps.weight") + struct.pack("<I", 2)
    head += struct.pack("<QQ", 32, 4) + struct.pack("<IQ", 8, 0)
    head += bytes(-len(head) % 32)
    return head + bytes(4 * 34)


class Response(io.BytesIO):
    def __init__(self, data: bytes, status: int = 200, headers: dict | None = None):
        super().__init__(data)
        self.status, self.headers = status, headers or {}

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def serve(files: dict[str, bytes], ranges: bool = True, reads: list | None = None):
    """A fake Hugging Face: the model's file list, and its files by range requests (`ranges` False: a mirror that
    ignores Range and answers 200 with the whole file).  `reads` collects the file answers it gave."""
    reads = [] if reads is None else reads

    def urlopen(req, timeout=None):
        url = req if isinstance(req, str) else req.full_url
        if "/api/models/" in url:
            tree = [{"type": "file", "path": p, "size": len(b)} for p, b in files.items()]
            return Response(json.dumps(tree).encode())
        name = url.rsplit("/", 1)[1]
        lo, hi = map(int, re.match(r"bytes=([0-9]+)-([0-9]+)", req.get_header("Range")).groups())
        data = files[name]
        if not ranges:
            resp = Response(data)
        else:
            hi = min(hi, len(data) - 1)
            resp = Response(data[lo:hi + 1], 206, {"Content-Range": f"bytes {lo}-{hi}/{len(data)}"})
        reads.append(resp)
        return resp
    return mock.patch.object(Q.urllib.request, "urlopen", urlopen)


class HfHeaders(unittest.TestCase):
    def test_a_shard_header_is_read_over_range_requests(self):
        with serve({"m-00001-of-00001.gguf": shard()}):
            heads = Q.hf_headers("org/repo", "rev", "dir")
        self.assertEqual([t.name for t in heads[0].tensors], ["blk.0.ffn_down_exps.weight"])

    def test_a_cut_header_is_refused_with_the_file_name(self):
        with serve({"m-00001-of-00001.gguf": shard()[:40]}):
            with self.assertRaises(ValueError) as cm:
                Q.hf_headers("org/repo", "rev", "dir")
        self.assertIn("m-00001-of-00001.gguf", str(cm.exception))

    def test_a_huge_name_length_is_refused_not_allocated(self):
        head = struct.pack("<IIQQ", 0x46554747, 3, 1, 0) + struct.pack("<Q", 2 ** 64 - 16) + bytes(32)
        with serve({"m.gguf": head}):
            with self.assertRaises(ValueError) as cm:
                Q.hf_headers("org/repo", "rev", "dir")
        self.assertIn("m.gguf", str(cm.exception))

    def test_a_server_that_ignores_range_is_refused_unread(self):
        # Issue #81: a 200 with the whole shard is an error, and its body is not read into memory
        reads: list = []
        with serve({"m.gguf": shard() + bytes(1 << 20)}, ranges=False, reads=reads):
            with self.assertRaises(OSError):
                Q.RangeFile("https://huggingface.co/org/repo/resolve/rev/m.gguf").read(16)
        self.assertTrue(reads)
        self.assertEqual([r.tell() for r in reads], [0] * len(reads))


if __name__ == "__main__":
    unittest.main()
