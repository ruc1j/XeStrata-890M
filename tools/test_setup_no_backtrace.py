# SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Setup starts no debugger (upstream 38ad0bcbd): its main() first sets GGML_NO_BACKTRACE=1 in its own environment
through the server's helper (serve/childenv.py), so every program setup starts inherits it, unless the variable is
already set or the developer asks for the backtrace (STRATA_BACKTRACE).

    python -m unittest tools.test_setup_no_backtrace
"""
from __future__ import annotations

import ast
import sys
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import setup  # noqa: E402
from serve import childenv  # noqa: E402


class SetupNoBacktrace(unittest.TestCase):
    def test_main_calls_it_first(self):
        tree = ast.parse((ROOT / "setup.py").read_text(encoding="utf-8"))
        main = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "main")
        self.assertEqual(ast.unparse(main.body[0]), "childenv.quiet_crash_env()")
        self.assertIs(setup.childenv, childenv)

    def test_setup_environment_gets_the_variable(self):
        with mock.patch.dict(setup.os.environ, {}, clear=True):
            setup.childenv.quiet_crash_env()
            self.assertEqual(setup.os.environ.get("GGML_NO_BACKTRACE"), "1")

    def test_a_value_already_there_and_the_developer_switch_are_kept(self):
        with mock.patch.dict(setup.os.environ, {"GGML_NO_BACKTRACE": "0"}, clear=True):
            setup.childenv.quiet_crash_env()
            self.assertEqual(setup.os.environ["GGML_NO_BACKTRACE"], "0")
        with mock.patch.dict(setup.os.environ, {"STRATA_BACKTRACE": "1"}, clear=True):
            setup.childenv.quiet_crash_env()
            self.assertNotIn("GGML_NO_BACKTRACE", setup.os.environ)


if __name__ == "__main__":
    unittest.main()
