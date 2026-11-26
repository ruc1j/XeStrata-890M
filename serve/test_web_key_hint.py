# SPDX-FileCopyrightText: 2026 Niko1221 and the Strata contributors
# SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
# SPDX-License-Identifier: LGPL-3.0-or-later AND MIT
"""upstream #7063 (51644762f): the page's "needs an API key" messages say where the key comes from: the one the server
was started with (--api-key, STRATA_API_KEY or the run config's api_key).  The sentence is one catalog key, translated
in every language the page has.

    python -m unittest serve.test_web_key_hint
"""
from __future__ import annotations

import json
import re
import unittest
from pathlib import Path

WEB = Path(__file__).with_name("web")


class KeyHint(unittest.TestCase):
    def test_every_401_message_adds_the_sentence(self):
        js = (WEB / "app.js").read_text(encoding="utf-8")
        for key in ("toast.keyAbout", "toast.keyAbout2", "toast.keyNeeded"):
            m = re.search(r'tr\("' + re.escape(key) + r'", "[^"]*"\) \+ " " \+ keyFrom\(\)', js)
            self.assertIsNotNone(m, key)
        # one literal, one text for the three places
        self.assertEqual(len(re.findall(r'tr\("toast\.keyFrom"', js)), 1)

    def test_the_sentence_names_the_three_sources_in_every_language(self):
        files = sorted((WEB / "locales").glob("*.json"))
        self.assertGreaterEqual(len(files), 9)
        for f in files:
            text = json.loads(f.read_text(encoding="utf-8"))["toast.keyFrom"]
            for word in ("--api-key", "STRATA_API_KEY", "api_key"):
                self.assertIn(word, text, f.name)
            self.assertNotIn("sk-strata-", text, f.name)                      # XeStrata makes no such key


if __name__ == "__main__":
    unittest.main()
