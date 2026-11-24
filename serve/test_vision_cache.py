# SPDX-FileCopyrightText: 2026 Niko1221 and the Strata contributors
# SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
# SPDX-License-Identifier: LGPL-3.0-or-later AND MIT
"""#1072 (upstream 91fb63284): a request's encoded pictures live until prepare() has copied them into its combined
file, then the cache returns to its 64 entries; a picture the reader's decoder refuses is sent once more as a PNG.

The real Vision cache and Service run against an in-memory ENC pipe; no GPU, model or network is needed.
    python -m unittest serve.test_vision_cache -v
"""
import contextlib
import hashlib
import queue
import tempfile
import threading
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from serve import server
from serve.frontend import ChatTemplate

ROOT = Path(__file__).resolve().parents[1]


def picture(i):
    # the PNG signature is enough for normalize()'s pass-through; no image decoder is under test
    return b"\x89PNG\r\n\x1a\n" + i.to_bytes(4, "big")


def record(data):
    return hashlib.sha256(data).digest()


class EncoderPipe:
    """The reader's stdin: an ENC line writes the output file at once and queues its answer."""

    def __init__(self, root, lines):
        self.root, self.lines, self.calls, self.fail_at = root, lines, 0, None
        self.text = ""

    def write(self, text):
        self.text = text

    def flush(self):
        _, image, out = self.text.split()
        self.calls += 1
        if self.calls == self.fail_at:
            self.lines.put("ERR injected encoding error\n")
            return
        (self.root / out).write_bytes(record((self.root / image).read_bytes()))
        self.lines.put("OK 1 1 1 0\n")


def make_vision(root):
    v = server.Vision.__new__(server.Vision)
    v.dir, v.lock, v.cache = root, threading.Lock(), {}
    v.stopped, v.dead, v.elastic = False, None, False
    v.lines = queue.Queue()
    pipe = EncoderPipe(root, v.lines)
    v.proc = SimpleNamespace(stdin=pipe, stdout=None, poll=lambda: None)
    return v, pipe


class VisionCacheLifetime(unittest.TestCase):
    @contextlib.contextmanager
    def service(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            v, pipe = make_vision(root)
            tok = server.ByteTokenizer()
            svc = server.Service(server.MockEngine(tok, "ok", max_context=131072), tok,
                                 ChatTemplate(ROOT / "serve/chat_template.jinja"), vision=v)
            number = 0

            def combined(*args):
                nonlocal number
                number += 1
                return root / f"req-{number}.sve"

            with mock.patch.object(server, "combined_embeddings_path", side_effect=combined), \
                    mock.patch.object(server.Vision, "load", side_effect=lambda src: picture(int(src))):
                try:
                    yield svc, v, pipe
                finally:
                    svc.drop_embeddings()

    @staticmethod
    def prepare(svc, values, **kwargs):
        messages = [{"role": "user", "content": [{"type": "image", "source": str(i)} for i in values]}]
        return svc.prepare(messages, None, {"enable_thinking": False}, **kwargs)

    def assert_idle(self, svc, v):
        self.assertEqual(v.cache_batches, 0)
        self.assertLessEqual(len(v.cache), 64)
        self.assertEqual(len(list(v.dir.glob("*.sve"))), len(v.cache))
        self.assertEqual(list(v.dir.glob("*.img")), [])
        free = threading.Event()

        def take():                     # another thread takes both locks after every exit, a BaseException's too
            with svc.fifo, v.lock:
                free.set()

        thread = threading.Thread(target=take, daemon=True)
        thread.start()
        self.assertTrue(free.wait(2), "FIFO/vision lock leaked")
        thread.join(2)

    def test_request_larger_than_cache_keeps_all_records_in_order(self):
        for size in (64, 65, 70, 130):
            with self.subTest(size=size), self.service() as (svc, v, pipe):
                self.prepare(svc, range(size))
                self.assertEqual(svc.embeddings.path.read_bytes(), b"".join(record(picture(i)) for i in range(size)))
                self.assertEqual(len(v.cache), 64)
                self.assertEqual(pipe.calls, size)
                svc.drop_embeddings()
                self.assert_idle(svc, v)

    def test_repeated_images_encode_once_and_preserve_order(self):
        with self.service() as (svc, v, pipe):
            self.prepare(svc, [0] * 70)
            self.assertEqual(pipe.calls, 1)
            self.assertEqual(svc.embeddings.path.read_bytes(), record(picture(0)) * 70)
            svc.drop_embeddings()
            values = list(range(70)) + [0, 4, 0, 69]
            self.prepare(svc, values)
            self.assertEqual(pipe.calls, 70)
            self.assertEqual(svc.embeddings.path.read_bytes(), b"".join(record(picture(i)) for i in values))
            svc.drop_embeddings()
            self.assert_idle(svc, v)

    def test_another_requests_encodes_cannot_evict_before_the_copy(self):
        # A has encoded its picture and given up the FIFO's turn; B encodes 64 others before A copies its file
        with self.service() as (svc, v, pipe):
            self.prepare(svc, range(100, 164))
            svc.drop_embeddings()
            paused, resume = threading.Event(), threading.Event()
            results, errors = {}, []
            original = server.write_temporary

            def gated(*args, **kwargs):
                if threading.current_thread().name == "image-A":
                    paused.set()
                    if not resume.wait(3):
                        raise AssertionError("the test's release did not arrive")
                return original(*args, **kwargs)

            def request_a():
                try:
                    self.prepare(svc, [0])
                    results["A"] = svc.embeddings.path.read_bytes()
                except BaseException as exc:            # noqa: BLE001 - the test reports it
                    errors.append(exc)
                finally:
                    svc.drop_embeddings()               # the combined file is this thread's

            with mock.patch.object(server, "write_temporary", side_effect=gated):
                a = threading.Thread(target=request_a, name="image-A", daemon=True)
                a.start()
                try:
                    self.assertTrue(paused.wait(2))
                    for i in range(64):
                        v.encode(str(200 + i))
                    self.assertGreater(len(v.cache), 64)    # nothing evicted while A's batch is open
                finally:
                    resume.set()
                    a.join(3)
            self.assertFalse(a.is_alive())
            self.assertEqual(errors, [])
            self.assertEqual(results, {"A": record(picture(0))})
            self.assert_idle(svc, v)

    def test_all_prepare_failures_trim_and_release(self) -> None:
        class Abort(BaseException):
            pass

        for case in ("encode", "markers", "starting", "no_room", "max_new", "write", "abort"):
            with self.subTest(case=case), self.service() as (svc, v, pipe), contextlib.ExitStack() as stack:
                error: type = ValueError
                kwargs = {}
                if case == "encode":
                    pipe.fail_at = 66
                elif case == "markers":
                    stack.enter_context(mock.patch.object(svc, "encode_prompt", return_value=[42]))
                elif case == "starting":
                    svc.engine.max_context, svc.engine.starting = 0, True
                    error = server.EngineStarting
                elif case == "no_room":
                    svc.engine.max_context = 64
                elif case == "max_new":
                    svc.fit_max_tokens = False
                    kwargs["max_new"] = 131072
                else:
                    error = OSError if case == "write" else Abort

                    def fail(path, parts, error=error):
                        path.write_bytes(b"partial")
                        raise error("injected copy failure")
                    stack.enter_context(mock.patch.object(server, "write_temporary", side_effect=fail))
                try:
                    with self.assertRaises(error):
                        self.prepare(svc, range(70), **kwargs)
                finally:
                    svc.drop_embeddings()
                self.assertEqual(len(v.cache), 64)
                self.assert_idle(svc, v)

    def test_nested_batch_trims_only_on_outer_exit(self):
        with self.service() as (svc, v, pipe):
            with v.batch():
                v.encode("0")
                with v.batch():
                    for i in range(1, 70):
                        v.encode(str(i))
                self.assertEqual(len(v.cache), 70)
                self.assertTrue(all(path.exists() for path, _ in v.cache.values()))
            self.assertEqual(len(v.cache), 64)
            self.assert_idle(svc, v)

    def test_no_run_drop_and_direct_encode_remain_bounded(self):
        with self.service() as (svc, v, pipe):
            for start in range(0, 210, 70):
                self.prepare(svc, range(start, start + 70))
                svc.drop_embeddings()
                svc.drop_embeddings()
                self.assert_idle(svc, v)
            for i in range(210, 300):
                v.encode(str(i))
                self.assertLessEqual(len(v.cache), 64)
            self.assert_idle(svc, v)


class VisionDecode(unittest.TestCase):
    """A picture the reader's decoder refuses ("ERR cannot read the image") is converted to PNG and sent once more."""

    def make(self, temp, replies):
        v = server.Vision.__new__(server.Vision)
        v.dir, v.lock, v.cache = Path(temp), threading.Lock(), {}
        v.stopped, v.dead = False, None
        answers = iter(replies)
        v._line = lambda timeout: next(answers, None)
        stdin = SimpleNamespace(lines=[], flush=lambda: None)
        stdin.write = stdin.lines.append
        v.proc = SimpleNamespace(stdin=stdin, stdout=None, poll=lambda: None)
        return v, stdin

    def test_a_refused_picture_is_converted_once(self):
        for replies, expected_calls in ((["ERR cannot read the image crop\n", "OK 450\n"], 2),
                                        (["ERR the vision encoder failed\n"], 1)):
            with self.subTest(replies=replies), tempfile.TemporaryDirectory() as temp:
                v, stdin = self.make(temp, replies)
                with mock.patch.object(server.Vision, "load", return_value=b"original"), \
                        mock.patch.object(server.Vision, "normalize",
                                          side_effect=lambda data, force=False: b"converted" if force else data) \
                        as normalize:
                    if expected_calls == 2:
                        encoded, tokens = v.encode("crop")
                        self.assertEqual(tokens, 450)
                        encoded.write_bytes(b"the encoder's output")
                        self.assertEqual(v.encode("crop")[1], 450)       # from the cache
                        self.assertEqual(sum(x.startswith("ENC ") for x in stdin.lines), 2)
                        self.assertEqual(normalize.call_args_list[1].kwargs, {"force": True})
                    else:
                        with self.assertRaisesRegex(ValueError, "vision encoder failed"):
                            v.encode("crop")
                        self.assertEqual(normalize.call_count, 1)
                self.assertFalse(list(Path(temp).glob("*.img")))

    def test_a_failed_conversion_removes_the_temporary_image(self):
        with tempfile.TemporaryDirectory() as temp:
            v, _stdin = self.make(temp, ["ERR cannot read the image crop\n"])
            with mock.patch.object(server.Vision, "load", return_value=b"original"), \
                    mock.patch.object(server.Vision, "normalize", side_effect=[b"original", ValueError("bad image")]):
                with self.assertRaisesRegex(ValueError, "bad image"):
                    v.encode("crop")
            self.assertFalse(list(Path(temp).glob("*.img")))


if __name__ == "__main__":
    unittest.main()
