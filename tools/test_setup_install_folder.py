# SPDX-FileCopyrightText: 2026 Niko1221 and the Strata contributors
# SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
# SPDX-License-Identifier: LGPL-3.0-or-later AND MIT
"""upstream #7012 (2fdbdb56d): a run config written into a folder that does not exist yet makes the folder first, so the
first config of a fresh install does not stop setup with FileNotFoundError on its .tmp file.

    python -m unittest tools.test_setup_install_folder
"""
from __future__ import annotations

import contextlib
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))
import setup  # noqa: E402


class MissingInstallFolder(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.folder = Path(self._tmp.name) / "share" / "xestrata" / "install"      # none of these exist
        self.assertFalse(self.folder.parent.exists())

    def test_write_config_makes_the_folder(self):
        cfg_path = self.folder / "xestrata-iq2_xs.json"
        setup.write_config(cfg_path, {"exe": "strata", "args": []})
        self.assertEqual(json.loads(cfg_path.read_text(encoding="utf-8"))["exe"], "strata")
        self.assertEqual([p.name for p in self.folder.iterdir()], [cfg_path.name], "no .tmp left behind")
        self.assertEqual(cfg_path.stat().st_mode & 0o777, 0o600)

    def test_write_config_takes_a_string_path(self):
        cfg_path = self.folder / "xestrata-a.json"
        setup.write_config(str(cfg_path), {"args": []})     # type: ignore[arg-type]
        self.assertTrue(cfg_path.is_file())

    def test_write_setup_config_first_install(self):
        cfg_path = self.folder / "xestrata-iq2_xs.json"
        with contextlib.redirect_stdout(io.StringIO()) as out:
            setup.write_setup_config(cfg_path, {"exe": "strata", "args": ["--port", "8080"]})
        self.assertEqual(json.loads(cfg_path.read_text(encoding="utf-8"))["args"], ["--port", "8080"])
        self.assertFalse(cfg_path.with_name(cfg_path.name + ".bak").exists(), out.getvalue())


if __name__ == "__main__":
    unittest.main()
