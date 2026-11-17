# SPDX-FileCopyrightText: 2026 Niko1221 and the Strata contributors
# SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
# SPDX-License-Identifier: LGPL-3.0-or-later AND MIT
"""The server never starts a debugger on a user's machine (serve/childenv.py, upstream 38ad0bcbd).

ggml's abort path forks `gdb --batch -ex attach <pid>` (Linux; lldb and backtrace_symbols after it).  Every child the
server starts therefore gets GGML_NO_BACKTRACE=1, unless a developer sets STRATA_BACKTRACE=1.  These tests check the
helper and the environments the server builds for the engine, the picture reader and the image engine."""
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from serve.childenv import DEVELOPER_SWITCH, NO_BACKTRACE, quiet_crash_env  # noqa: E402


class Helper(unittest.TestCase):
    def test_sets_the_variable(self):
        with mock.patch.dict(os.environ, {}, clear=True):
            self.assertEqual(quiet_crash_env({})[NO_BACKTRACE], "1")

    def test_without_an_argument_it_changes_this_process(self):
        with mock.patch.dict(os.environ, {}, clear=True):
            quiet_crash_env()
            self.assertEqual(os.environ[NO_BACKTRACE], "1")

    def test_a_value_that_is_already_there_is_kept(self):
        with mock.patch.dict(os.environ, {}, clear=True):
            # (ggml reads any value as "skip")
            self.assertEqual(quiet_crash_env({NO_BACKTRACE: "0"})[NO_BACKTRACE], "0")

    def test_the_developer_switch_leaves_the_backtrace_on(self):
        for how in ("1", "yes", "true", "anything"):
            with self.subTest(how=how), mock.patch.dict(os.environ, {DEVELOPER_SWITCH: how}, clear=True):
                self.assertNotIn(NO_BACKTRACE, quiet_crash_env(dict(os.environ)))
        with mock.patch.dict(os.environ, {}, clear=True):
            # named in the dict being built, not in this process
            self.assertNotIn(NO_BACKTRACE, quiet_crash_env({DEVELOPER_SWITCH: "1"}))

    def test_the_switch_set_to_nothing_or_zero_is_not_a_request(self):
        for how in ("", "0", "no", "off", "False"):
            with self.subTest(how=how), mock.patch.dict(os.environ, {DEVELOPER_SWITCH: how}, clear=True):
                self.assertEqual(quiet_crash_env(dict(os.environ))[NO_BACKTRACE], "1")

    def test_it_returns_the_dict_it_was_given(self):
        env = {"A": "1"}
        self.assertIs(quiet_crash_env(env), env)


class SpawnEnvironments(unittest.TestCase):
    """The environment each place that builds one hands to its child."""

    def setUp(self):
        patcher = mock.patch.dict(os.environ, {k: v for k, v in os.environ.items()
                                               if k not in (NO_BACKTRACE, DEVELOPER_SWITCH)}, clear=True)
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_the_engine_and_the_picture_reader(self):
        import serve.server as server
        self.assertEqual(server.child_env({})[NO_BACKTRACE], "1")       # (the picture reader is given the same)

    def test_the_developer_switch_reaches_the_engine_environment(self):
        import serve.server as server
        with mock.patch.dict(os.environ, {DEVELOPER_SWITCH: "1"}):
            env = server.child_env({})
        self.assertNotIn(NO_BACKTRACE, env)

    def test_the_config_may_name_it_itself(self):
        import serve.server as server
        self.assertEqual(server.child_env({"env": {NO_BACKTRACE: "0"}})[NO_BACKTRACE], "0")

    def test_the_image_engine(self):
        from serve import images
        with tempfile.TemporaryDirectory() as d:
            for env in (None, {"PATH": "x"}):
                with self.subTest(env=env):
                    m = images.ImageGen({"exe": "strata-image", "diffusion_model": "d.gguf", "vae": "v.gguf",
                                         "llm": "l.gguf", "out_dir": d}, env=env)
                    self.assertEqual(m.engine.env[NO_BACKTRACE], "1")


if __name__ == "__main__":
    unittest.main()
