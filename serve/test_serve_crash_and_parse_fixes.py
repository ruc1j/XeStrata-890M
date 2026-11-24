# SPDX-FileCopyrightText: 2026 Niko1221 and the Strata contributors
# SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
# SPDX-License-Identifier: LGPL-3.0-or-later AND MIT
"""upstream f1f91d97b: the engine's exit status and last words when it dies, a declared parameter called `function`,
UTF-8 in streamed tool arguments, Codex's compaction_trigger item and the picture reader's log descriptor.

    python -m unittest serve.test_serve_crash_and_parse_fixes -v
"""
from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import types
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from serve import frontend as F, responses, server  # noqa: E402


class ExitCodeAndCrashReason(unittest.TestCase):
    """An engine's death never reads "no exit status yet" while the OS still reaps it, and names its last words."""

    def make(self, proc, log_path=None, last_err=None):
        e = object.__new__(server.StrataEngine)
        e.proc = proc
        e._last_exit_code = None
        e.last_err = last_err
        e.log_path = str(log_path) if log_path else None
        e.log_start = 0
        e.silent_note = None
        return e

    def test_exit_code_waits_briefly_when_poll_is_none_as_the_pipes_close(self):
        class ReapingProc:
            def __init__(self):
                self.returncode = None
                self.waited = []

            def poll(self):
                return self.returncode

            def wait(self, timeout=None):
                self.waited.append(timeout)
                self.returncode = -11
                return self.returncode

        with tempfile.TemporaryDirectory() as d:
            log = Path(d) / "engine.log"
            log.write_text("[strata] engine started\nmemory access fault by GPU node-1\n", encoding="utf-8")
            proc = ReapingProc()
            e = self.make(proc, log_path=log)
            err = e._engine_died()
            self.assertEqual(proc.waited, [0.25])
            self.assertEqual(e.exit_code(), -11)
            self.assertIn("SIGSEGV", str(err))
            self.assertIn("memory access fault", str(err))
            self.assertNotIn("[strata] engine started", str(err))
            e.proc = None                               # the status survives the process object
            self.assertEqual(e.exit_code(), -11)

    def test_an_unreaped_process_says_so(self):
        class StuckProc:
            returncode = None

            def poll(self):
                return None

            def wait(self, timeout=None):
                raise subprocess.TimeoutExpired("strata", timeout)

        with tempfile.TemporaryDirectory() as d:
            log = Path(d) / "engine.log"
            log.write_text("hipErrorOutOfMemory allocating 2048 MiB\n", encoding="utf-8")
            err = self.make(StuckProc(), log_path=log)._engine_died()
            self.assertIn("status unavailable", str(err))
            self.assertIn("hipErrorOutOfMemory", str(err))

    def test_death_note_uses_the_reaped_exit_code(self):
        class ReapingProc:
            returncode = None

            def poll(self):
                return self.returncode

            def wait(self, timeout=None):
                self.returncode = 1
                return 1

        note = self.make(ReapingProc(), last_err="kv allocation failed").death_note()
        self.assertIn("code 1", note)
        self.assertIn("kv allocation failed", note)


class DeclaredFunctionParameter(unittest.TestCase):
    """upstream #6461: a parameter literally named `function` is kept when the tool declares it (Playwright MCP's
    browser_evaluate)."""

    SCHEMA = {"name": "browser_evaluate",
              "parameters": {"type": "object",
                             "properties": {"function": {"type": "string"}, "element": {"type": "string"}},
                             "required": ["function"]}}

    def test_parse_tool_call_keeps_it(self):
        body = "<function=browser_evaluate>\n<parameter=function>\n() => document.title\n</parameter>\n</function>"
        for recover in (False, True):
            with self.subTest(recover=recover):
                call = F.parse_tool_call(body, self.SCHEMA, recover=recover)
                self.assertEqual(call.name, "browser_evaluate")
                self.assertEqual(call.arguments, {"function": "() => document.title"})

    def test_the_streaming_parser_keeps_it(self):
        raw = ("<tool_call>\n<function=browser_evaluate>\n<parameter=function>\n"
               "() => document.title\n</parameter>\n</function>\n</tool_call>")
        for width in (1, 3, len(raw)):
            with self.subTest(width=width):
                p = F.OutputParser(thinking=False, tools=[self.SCHEMA], stream_tools=True, recover=True)
                evs = []
                for i in range(0, len(raw), width):
                    evs.extend(p.feed(raw[i:i + width]))
                evs.extend(p.finish())
                self.assertEqual(json.loads("".join(e.text for e in evs if e.kind == "tool_args")),
                                 {"function": "() => document.title"})
                calls = [e.call for e in evs if e.kind == "tool_call"]
                self.assertEqual([c.arguments for c in calls], [{"function": "() => document.title"}])

    def test_an_undeclared_one_is_still_dropped(self):
        schema = {"name": "read_file", "parameters": {"type": "object", "properties": {"path": {"type": "string"}}}}
        body = ("<function=read_file>\n<parameter=function>\njunk\n</parameter>\n<parameter=path>\na.py\n</parameter>\n"
                "</function>")
        self.assertEqual(F.parse_tool_call(body, schema, recover=True).arguments, {"path": "a.py"})


class Utf8ToolArguments(unittest.TestCase):
    """upstream #6456: streamed string arguments keep their characters, not \\uXXXX escapes."""

    def test_streaming_string_params_keep_utf8(self):
        tools = [{"name": "write_file",
                  "parameters": {"type": "object",
                                 "properties": {"path": {"type": "string"}, "note": {"type": "string"}}}}]
        raw = ("<tool_call>\n<function=write_file>\n<parameter=path>\nПапка/Отчёт.md\n</parameter>\n"
               "<parameter=note>\n中文 🚀\n</parameter>\n</function>\n</tool_call>")
        for width in (1, 4, len(raw)):
            with self.subTest(width=width):
                p = F.OutputParser(thinking=False, tools=tools, stream_tools=True)
                evs = []
                for i in range(0, len(raw), width):
                    evs.extend(p.feed(raw[i:i + width]))
                evs.extend(p.finish())
                arg_text = "".join(e.text for e in evs if e.kind == "tool_args")
                self.assertNotIn("\\u041f", arg_text)
                self.assertIn("Папка/Отчёт.md", arg_text)
                self.assertEqual(json.loads(arg_text), {"path": "Папка/Отчёт.md", "note": "中文 🚀"})


class CodexCompactionTrigger(unittest.TestCase):
    """upstream #6459: /v1/responses takes Codex's compaction_trigger item instead of answering 400."""

    def test_a_bare_trigger_is_taken(self):
        msgs = responses.input_messages({"instructions": "You are Codex.",
                                         "input": [{"type": "message", "role": "user", "content": "Read the files."},
                                                   {"type": "compaction_trigger"}]})
        self.assertEqual([m["role"] for m in msgs], ["system", "user"])

    def test_its_prompt_is_kept(self):
        msgs = responses.input_messages({"input": [{"type": "message", "role": "user", "content": "Work done."},
                                                   {"type": "compaction_trigger", "prompt": "Summarize progress."}]})
        self.assertEqual(msgs[-1], {"role": "user", "content": "Summarize progress."})


class VisionLogDescriptor(unittest.TestCase):
    """upstream #6467: a log object whose name is a descriptor (os.fdopen) is never opened by number: open(int) would
    close the server's log descriptor on leaving the with block."""

    def test_an_integer_name_is_ignored(self):
        self.assertEqual(server.vision_log_tail(types.SimpleNamespace(name=999999)), "")

    def test_a_path_name_is_read(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "vision.log"
            with open(p, "a", encoding="utf-8") as log:
                log.write("gpu reader failed: no kernel image\n")
                log.flush()
                self.assertIn("no kernel image", server.vision_log_tail(log))


if __name__ == "__main__":
    unittest.main()
