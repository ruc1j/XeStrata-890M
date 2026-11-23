# SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
# SPDX-License-Identifier: LGPL-3.0-or-later
"""setup's Inkling-2-Mini family: its sizes and pins (serve/inkling2mini_family.py), the rule that recommends a size
(serve/inkling2mini_fit.py) and the download of one file (inkling2mini_fetch).  No network, no model file."""
import hashlib
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import setup  # noqa: E402
from serve import inkling2mini_family as family  # noqa: E402
from serve import inkling2mini_fit as fit  # noqa: E402


class Family(unittest.TestCase):
    def test_the_sizes_setup_lists_are_the_pinned_files(self):
        fam = setup.FAMILIES[setup.INKLING2MINI]
        self.assertTrue(fam["plain"] and fam["inkling2mini"])
        self.assertEqual(list(fam["models"]), ["SQ-10", "SQ-15", "SQ-20"])
        for build, d in fam["models"].items():
            size, sha = family.FILES[d["file"]]
            self.assertEqual((d["size"], d["sha256"]), (size, sha))
            self.assertEqual(len(sha), 64)
        self.assertEqual(fam["models"]["SQ-10"]["size"], 10_471_526_400)
        self.assertNotIn(setup.INKLING2MINI, setup.PLAIN_FAMILIES)        # its own flow, not setup_plain's

    def test_the_files_of_a_size(self):
        self.assertEqual(family.files_of("SQ-10"), ["Inkling-2-Mini-SQ-10.strata", "Inkling-2-Mini-Vision.strata"])
        self.assertEqual(family.files_of("SQ-20", vision=False), ["Inkling-2-Mini-SQ-20.strata"])
        self.assertEqual(family.model_key("SQ-15"), "inkling-2-mini-sq-15")
        self.assertEqual(family.EXE, "strata-inkling-2-mini")

    def test_the_sources(self):
        shipped = family.sources(env={})
        self.assertEqual(len(shipped), 1)
        self.assertTrue(shipped[0]["base"].startswith("https://huggingface.co/"))
        first = family.sources(env={family.SOURCE_ENV: "https://example.org/x/"})[0]
        self.assertEqual(family.urls_for("a.strata", first), ["https://example.org/x/a.strata"])
        self.assertEqual(family.sources(env={}, cfg={"inkling2mini_source": "ftp://no"})[0], shipped[0])


class Fit(unittest.TestCase):
    def build(self, *args, **kw):
        return fit.recommend(*args, **kw)["build"]

    def test_an_integrated_gpu_gets_the_smallest_size(self):
        got = fit.recommend(None, 62.0, True, igpu=True)
        self.assertEqual(got["build"], "SQ-10")
        self.assertTrue(got["unified"] and got["fits"])

    def test_cards_by_their_memory(self):
        self.assertEqual(self.build(24.0, 64.0), "SQ-20")
        self.assertEqual(self.build(16.0, 32.0), "SQ-15")
        self.assertEqual(self.build(12.0, 32.0), "SQ-15")
        self.assertEqual(self.build(12.0, 32.0, preference="speed"), "SQ-10")
        self.assertEqual(self.build(12.0, 16.0), "SQ-10")
        self.assertEqual(self.build(8.0, 32.0), "SQ-10")

    def test_other_unified_memory_by_its_size(self):
        self.assertEqual(self.build(None, 64.0, True), "SQ-20")
        self.assertEqual(self.build(None, 24.0, True), "SQ-15")
        self.assertEqual(self.build(None, 16.0, True), "SQ-10")

    def test_more_slots_take_more_memory(self):
        self.assertLess(fit.kv_cache_gb(32768, 1), fit.kv_cache_gb(32768, 8))
        self.assertEqual([r["build"] for r in fit.rows(8.0, 32.0)], list(fit.ORDER))


class Fetch(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.dir.cleanup)
        self.folder = Path(self.dir.name)
        self.data = b"strata"
        pins = {"x.strata": (len(self.data), hashlib.sha256(self.data).hexdigest())}
        for target, value in ((family, "FILES"), (family, "TOKENIZER_FILES")):
            patch = mock.patch.dict(getattr(target, value), pins if value == "FILES" else {"t.json": (6, None)})
            patch.start()
            self.addCleanup(patch.stop)

    def fetch(self, name, body):
        def fake(url, dst, what=None):
            self.urls.append(url)
            Path(dst).write_bytes(body)
        self.urls: list = []
        with mock.patch.object(setup, "download", fake), mock.patch.object(setup, "say", lambda *a, **k: None):
            return setup.inkling2mini_fetch(None, name, self.folder)

    def test_a_file_is_downloaded_and_checked(self):
        got = self.fetch("x.strata", self.data)
        self.assertEqual(got.read_bytes(), self.data)
        self.assertTrue(self.urls[0].endswith("/x.strata"))
        self.fetch("x.strata", b"other!")                    # there and whole: not fetched again
        self.assertEqual(self.urls, [])
        self.assertEqual(got.read_bytes(), self.data)

    def test_a_wrong_file_is_refused(self):
        with mock.patch.object(setup, "fail", side_effect=SystemExit(1)), self.assertRaises(SystemExit):
            self.fetch("x.strata", b"wrong.")                # the pinned size, another SHA-256
        with mock.patch.object(setup, "fail", side_effect=SystemExit(1)), self.assertRaises(SystemExit):
            self.fetch("t.json", b"too long")                # no SHA-256 pinned: the size is checked


if __name__ == "__main__":
    unittest.main()
