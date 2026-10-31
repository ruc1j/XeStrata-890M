# SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
# SPDX-License-Identifier: LGPL-3.0-or-later
"""tools/checkpoint_source.py: a range read takes only a 206 with the range asked for.  A server that ignores Range
(a mirror, a proxy) answers 200 with the whole file: that answer is not read, so the retries cost no transfer."""
import http.server
import sys
import threading
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent))
import checkpoint_source  # noqa: E402

DATA = bytes(range(256)) * 400     # 102,400 bytes


class Server:
    """A file server that honours Range (206) or ignores it (200 with the whole file), counting the bytes it sent."""

    def __init__(self, ranges: bool):
        self.ranges, self.sent, self.requests = ranges, 0, 0
        outer = self

        class H(http.server.BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def do_GET(self):
                outer.requests += 1
                rng = self.headers.get("Range")
                if rng and outer.ranges:
                    a, b = rng.split("=")[1].split("-")
                    body = DATA[int(a):int(b) + 1]
                    self.send_response(206)
                    self.send_header("Content-Range", "bytes %s-%s/%d" % (a, b, len(DATA)))
                else:
                    body = DATA
                    self.send_response(200)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                try:
                    self.wfile.write(body)
                    outer.sent += len(body)
                except OSError:
                    pass

        self.httpd = http.server.ThreadingHTTPServer(("127.0.0.1", 0), H)
        self.port = self.httpd.server_address[1]
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()

    def close(self):
        self.httpd.shutdown()
        self.httpd.server_close()


class RangeReads(unittest.TestCase):
    def read(self, srv, off, n):
        with mock.patch.object(checkpoint_source, "ENDPOINT", "http://127.0.0.1:%d" % srv.port), \
                mock.patch.object(checkpoint_source.time, "sleep", lambda s: None):
            return checkpoint_source.Source(repo="x/y").read("model.safetensors", off, n)

    def test_a_206_gives_the_range(self):
        srv = Server(ranges=True)
        try:
            self.assertEqual(self.read(srv, 1000, 8), DATA[1000:1008])
            self.assertEqual(srv.requests, 1)
        finally:
            srv.close()

    def test_a_server_that_ignores_range_is_not_read_whole(self):
        srv = Server(ranges=False)
        try:
            with mock.patch("builtins.print"), self.assertRaises(SystemExit):
                self.read(srv, 0, 8)
            self.assertEqual(srv.requests, 8)                 # every attempt was refused by its status, not its length
        finally:
            srv.close()


if __name__ == "__main__":
    unittest.main()
