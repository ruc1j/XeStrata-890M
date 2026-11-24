# SPDX-FileCopyrightText: 2026 Niko1221 and the Strata contributors
# SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
# SPDX-License-Identifier: LGPL-3.0-or-later AND MIT
"""Tests for Inkling-2-Mini o200k tokenizer conversion and round-trip encode/decode."""
from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import inkling2mini_tokenizer  # noqa: E402
import strata_tokenizer  # noqa: E402


def _minimal_o200k_hf_json() -> dict:
    # Build a small but valid o200k-style byte-level BPE tokenizer.json
    b2u = strata_tokenizer.bytes_to_unicode()
    vocab = {b2u[b]: b for b in range(256)}
    next_id = 256
    merges = []
    # Add a few merges so multi-byte tokens merge cleanly
    for pair in [("h", "e"), ("l", "l"), ("he", "ll"), ("hell", "o"), ("Ġ", "w"), ("Ġw", "o"), ("Ġwo", "r"),
                 ("Ġwor", "l"), ("Ġworl", "d")]:
        merged = pair[0] + pair[1]
        if merged not in vocab:
            vocab[merged] = next_id
            next_id += 1
            merges.append(f"{pair[0]} {pair[1]}")
    added = [
        {"id": 199999, "content": "<|endoftext|>", "special": True},
        {"id": 200001, "content": "<|message_system|>", "special": True},
        {"id": 200002, "content": "<|message_user|>", "special": True},
        {"id": 200003, "content": "<|message_model|>", "special": True},
        {"id": 200004, "content": "<|message_tool|>", "special": True},
        {"id": 200005, "content": "<|end_message|>", "special": True},
        {"id": 200006, "content": "<|content_model_end_sampling|>", "special": True},
        {"id": 200010, "content": "<|content_text|>", "special": True},
        {"id": 200011, "content": "<|content_thinking|>", "special": True},
        {"id": 200012, "content": "<|content_invoke_tool_json|>", "special": True},
        {"id": 200013, "content": "<|content_image|>", "special": True},
        {"id": 200028, "content": "<|begin_of_text|>", "special": True},
        {"id": 200054, "content": "<|unused_200054|>", "special": True},
    ]
    return {
        "version": "1.0",
        "added_tokens": added,
        "pre_tokenizer": {
            "type": "Sequence",
            "pretokenizers": [
                {
                    "type": "Split",
                    "pattern": {"Regex": strata_tokenizer.O200K_PATTERN},   # the o200k Split rule
                    "behavior": "Isolated",
                    "invert": False,
                },
                {
                    "type": "ByteLevel",
                    "add_prefix_space": False,
                    "trim_offsets": False,
                    "use_regex": False,
                },
            ],
        },
        "model": {
            "type": "BPE",
            "vocab": vocab,
            "merges": merges,
        },
    }


class TestInkling2MiniTokenizer(unittest.TestCase):
    def test_convert_and_roundtrip_o200k(self):
        hf = _minimal_o200k_hf_json()
        with tempfile.TemporaryDirectory() as td:
            cfg = inkling2mini_tokenizer.convert(hf, "{{ messages }}", td)
            self.assertEqual(cfg["pre"], "o200k")
            self.assertEqual(cfg["vocab_size"], 200055)
            self.assertEqual(cfg["special_ids"]["eos"], 200006)
            self.assertEqual(cfg["special_ids"]["bos"], 200028)
            self.assertEqual(cfg["special_ids"]["pad"], 199999)

            p = Path(td)
            vocab = json.loads((p / "vocab.json").read_text(encoding="utf-8"))
            tokens = [""] * len(vocab)
            for t, i in vocab.items():
                tokens[i] = t
            merges = [m for m in (p / "merges.txt").read_text(encoding="utf-8").split("\n") if m]
            types = json.loads((p / "token_type.json").read_text(encoding="utf-8"))
            tok = strata_tokenizer.Tokenizer(tokens, merges, types, cfg["pre"], cfg["special_ids"])
            samples = [
                "hello world! 1234567890",
                "<|message_user|><|content_text|>Check x < y && a > b & \"quoted\" 'single'</tool_call><|end_message|>",
                "<|message_user|><|content_image|><|unused_200054|><|end_message|>",
                "def foo(x: int) -> bool:\n    return x < 10 and x > 0\n",
                "日本語テスト 🚀 café naïve",
            ]
            for s in samples:
                ids = tok.encode(s, parse_special=True)
                self.assertEqual(tok.decode(ids), s, f"round-trip mismatch for {s!r}")

            # Verify special tokens map to exact single IDs
            self.assertEqual(tok.encode("<|content_model_end_sampling|>", parse_special=True), [200006])
            self.assertEqual(tok.encode("<|unused_200054|>", parse_special=True), [200054])
            self.assertEqual(tok.encode("<|content_thinking|>", parse_special=True), [200011])

    def test_rejects_non_o200k_pretokenizer(self):
        bad = _minimal_o200k_hf_json()
        bad["pre_tokenizer"] = {"type": "Whitespace"}
        with tempfile.TemporaryDirectory() as td:
            with self.assertRaises(inkling2mini_tokenizer.NotInkling2MiniTokenizer):
                inkling2mini_tokenizer.convert(bad, "{{ messages }}", td)

    def test_serve_makes_a_missing_tokenizer_again_from_the_files_beside_the_model(self):
        # upstream 231df084a: serve/server.py rebuilds vocab.json instead of stopping with "tokenizer is missing"
        sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
        from serve import server
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            model = root / "models" / "Inkling-2-Mini-SQ-20.gguf"
            model.parent.mkdir()
            model.write_bytes(b"")
            (model.parent / "tokenizer.json").write_text(json.dumps(_minimal_o200k_hf_json()), encoding="utf-8")
            (model.parent / "chat_template.jinja").write_text("{{ messages }}", encoding="utf-8")
            cfg = {"arch": "inkling2_loop", "cwd": str(root), "args": ["--native", "models/" + model.name]}
            tpath = root / "tokenizer"
            self.assertTrue(server._ensure_inkling2mini_tokenizer(cfg, tpath))
            self.assertTrue((tpath / "vocab.json").is_file())
            # a model of another kind is left to the plain stop
            other = root / "other"
            self.assertFalse(server._ensure_inkling2mini_tokenizer({"args": ["--native", "qwen.strata"]}, other))
            self.assertFalse((other / "vocab.json").exists())


if __name__ == "__main__":
    unittest.main()
