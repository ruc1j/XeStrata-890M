# SPDX-FileCopyrightText: 2026 Niko1221 and the Strata contributors
# SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
# SPDX-License-Identifier: LGPL-3.0-or-later AND MIT
"""upstream 3cf22f5ef (#6055): a server that stops refuses new generation requests while the engine closes.

Before, a chat request that arrived during the stop (a kept-alive connection's next request, one accepted just before
the port closed) was taken, the engine closed under it, and it came back as 200 with 0 tokens and finish_reason
"length", which a client cannot tell from a real reply.  Now the stop closes admission first (Service.begin_drain):
such a request gets 503 server_draining, to retry against the next instance.
"""

import contextlib
import http.client
import io
import json
import os
import sys
import threading
import time as real_time
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from serve import server  # noqa: E402
from serve.frontend import ChatTemplate  # noqa: E402
from serve.server import ByteTokenizer, MockEngine, Service, serve  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


class DrainRefusal(unittest.TestCase):
    def setUp(self):
        tok = ByteTokenizer()
        self.svc = Service(MockEngine(tok, "</think>\n\nok", max_context=4096), tok,
                           ChatTemplate(ROOT / "serve/chat_template.jinja"))
        self.httpd = serve(self.svc, port=0)
        self.addCleanup(self.httpd.server_close)
        self.addCleanup(self.httpd.shutdown)

    def post(self, path, body):
        c = http.client.HTTPConnection("127.0.0.1", self.httpd.server_address[1], timeout=30)
        self.addCleanup(c.close)
        c.request("POST", path, json.dumps(body), {"Content-Type": "application/json"})
        r = c.getresponse()
        return r.status, dict(r.getheaders()), json.loads(r.read() or b"{}")

    def chat(self):
        return self.post("/v1/chat/completions", {"model": "m", "max_tokens": 8,
                                                  "messages": [{"role": "user", "content": "hi"}]})

    def test_before_the_stop_a_request_is_answered(self):
        self.assertEqual(self.chat()[0], 200)

    def test_after_begin_drain_a_chat_request_gets_503_server_draining(self):
        self.svc.begin_drain()
        status, headers, body = self.chat()
        self.assertEqual(status, 503)
        self.assertEqual(body["error"]["code"], "server_draining")
        self.assertEqual(headers.get("Retry-After"), "1")

    def test_messages_and_responses_are_refused_too(self):
        self.svc.begin_drain()
        status, _, body = self.post("/v1/messages", {"model": "m", "max_tokens": 8,
                                                     "messages": [{"role": "user", "content": "hi"}]})
        self.assertEqual(status, 503)                    # (in Anthropic's error shape, which has no code)
        self.assertIn("stopping", body["error"]["message"])
        status, _, body = self.post("/v1/responses", {"model": "m", "input": "hi"})
        self.assertEqual((status, body["error"]["code"]), (503, "server_draining"))


class _Clock:
    """serve.server's `time`, except that the main thread's idle `time.sleep(1)` is a Ctrl+C (the stop)."""

    def __getattr__(self, name):
        return getattr(real_time, name)

    @staticmethod
    def sleep(seconds):
        if threading.current_thread() is threading.main_thread() and seconds == 1:
            raise KeyboardInterrupt
        real_time.sleep(seconds)


class StopClosesAdmission(unittest.TestCase):
    def test_the_stop_path_closes_admission_before_the_engine_closes(self):
        drains, refused = [], []
        original = Service.begin_drain

        def begin_drain(svc):
            drains.append(svc)
            original(svc)
            refused.append(svc.draining)

        env = {k: v for k, v in os.environ.items() if k not in ("STRATA_API_KEY",)}
        env["STRATA_BLOCKING_CONSOLE"] = "1"
        with mock.patch.dict(os.environ, env, clear=True), \
                mock.patch.object(sys, "argv", ["server", "--engine", "mock", "--port", "0"]), \
                mock.patch.object(server, "time", _Clock()), \
                mock.patch.object(Service, "begin_drain", begin_drain), \
                mock.patch.object(server.release_check, "enabled", lambda: False), \
                contextlib.redirect_stdout(io.StringIO()) as out, contextlib.redirect_stderr(io.StringIO()):
            code = server.main()
        self.assertIn("[xestrata] stopped", out.getvalue())
        self.assertEqual(len(drains), 1)
        self.assertEqual(refused, [True])
        self.assertEqual(code, 0)


if __name__ == "__main__":
    unittest.main()
