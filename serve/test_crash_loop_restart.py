# SPDX-FileCopyrightText: 2026 Niko1221 and the Strata contributors
# SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
# SPDX-License-Identifier: LGPL-3.0-or-later AND MIT
"""serve/test_crash_loop_restart.py: upstream #6896 #6830 #6849.  An engine that dies at its start or on its first
request was started again by every request that found it dead, for ever (375 failed starts in five hours, no
message).  Three deaths within ten minutes stop the restarts, and the 503 says what the engine said last and where its
log is.  CPU only, no engine: Service.ensure_loaded runs on a stand-in whose engine dies each time it is started."""
import sys
import tempfile
import types
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from serve import server as S  # noqa: E402


class DyingEngine:
    """Every start 'works' and the process is dead again at the next request (a first-request fault), or the start
    itself fails."""

    def __init__(self, last_err, log_path=None, fail_start=False):
        self.last_err, self.log_path, self.fail_start = last_err, log_path, fail_start
        self.proc = types.SimpleNamespace(pid=100)
        self.restarts = 0
        self.unloaded = False

    def exit_code(self):
        return 3

    def restart(self):
        self.restarts += 1
        self.proc = types.SimpleNamespace(pid=100 + self.restarts)      # a new process, dead again at the next request
        if self.fail_start:
            raise RuntimeError("the engine did not start")

    def _crash_tail(self):
        return self.last_err or ""


def service(engine):
    svc = types.SimpleNamespace(engine=engine, before_load=None, min_free_vram_mib=0, restore_after_load=None,
                                vision=None, vram_reserve=None, loaded=lambda: False, _vision_down=lambda: False)
    for name in ("FATAL_GPU_MAX", "FATAL_GPU_WINDOW_S", "CRASH_LOOP_MAX", "CRASH_LOOP_WINDOW_S"):
        setattr(svc, name, getattr(S.Service, name))
    for name in ("_note_fatal_gpu_exit", "_note_crash_loop", "_engine_last_words", "_crash_loop_text",
                 "ensure_loaded"):
        setattr(svc, name, getattr(S.Service, name).__get__(svc))
    return svc


class CrashLoop(unittest.TestCase):
    def ask(self, svc, n):
        """n requests, each finding the engine dead: the refusals."""
        refused = []
        for _ in range(n):
            try:
                svc.ensure_loaded()
            except S.EngineDied as e:
                refused.append(str(e))
            except RuntimeError:
                pass                                   # a start that failed: the request fails, the next finds it dead
        return refused

    def test_an_engine_that_dies_every_time_is_started_twice_then_refused_with_its_last_words(self):
        with tempfile.TemporaryDirectory() as d:
            log = str(Path(d) / "engine.log")
            eng = DyingEngine("verify: something else broke", log_path=log)
            refused = self.ask(service(eng), 50)
            self.assertEqual(eng.restarts, 2, "restarted for the first two deaths, never again")
            self.assertEqual(len(refused), 48)
            text = refused[0]
            self.assertIn("The engine stopped 3 times in 10 minutes", text)
            self.assertIn("verify: something else broke", text)
            self.assertIn(log, text)
            self.assertEqual(set(refused), {text}, "the same plain message every time")

    def test_a_start_that_itself_fails_is_limited_the_same_way(self):
        eng = DyingEngine("out of memory", fail_start=True)
        refused = self.ask(service(eng), 50)
        self.assertEqual(eng.restarts, 2)
        self.assertEqual(len(refused), 48)

    def test_the_last_line_of_several_is_named(self):
        refused = self.ask(service(DyingEngine("strata: loading\nstrata: hipErrorOutOfMemory")), 5)
        self.assertIn("hipErrorOutOfMemory", refused[0])
        self.assertNotIn("strata: loading", refused[0])

    def test_after_the_window_it_tries_again(self):
        eng = DyingEngine("boom")
        svc = service(eng)
        self.ask(svc, 5)
        self.assertEqual(eng.restarts, 2)
        svc._crash_times[:] = [t - S.Service.CRASH_LOOP_WINDOW_S - 1 for t in svc._crash_times]
        self.ask(svc, 1)
        self.assertEqual(eng.restarts, 3, "the old deaths aged out: this request starts it once more")

    def test_deaths_far_apart_never_stop_a_working_engine(self):
        eng = DyingEngine("boom")
        svc = service(eng)
        for _ in range(10):
            self.ask(svc, 1)
            # ten minutes pass between deaths
            svc._crash_times[:] = [t - S.Service.CRASH_LOOP_WINDOW_S - 1 for t in svc._crash_times]
        self.assertEqual(eng.restarts, 10)

    def test_an_engine_program_that_cannot_be_run_keeps_its_own_plain_error(self):
        """No process ever existed (a missing program): every request gets "the engine could not be started: ..." at
        once, never the loop message (serve.test_ready_timeout.SpawnFailure asks four requests in a row)."""
        eng = DyingEngine("boom")
        eng.proc = None

        def restart():
            eng.restarts += 1
            raise OSError(2, "No such file or directory")
        eng.restart = restart
        svc = service(eng)
        for _ in range(10):
            with self.assertRaises(S.EngineDied) as cm:
                svc.ensure_loaded()
            self.assertIn("the engine could not be started", str(cm.exception))
        self.assertEqual(eng.restarts, 10)

    def test_an_unloaded_model_is_not_a_crash(self):
        eng = DyingEngine("boom")
        eng.unloaded = True
        svc = service(eng)
        self.ask(svc, 10)
        self.assertEqual(eng.restarts, 10, "loading it again after an unload is not counted")

    def test_a_refusal_stops_the_vision_encoder_it_started(self):
        """The encoder starts before the engine (it takes its VRAM first).  When the engine is then refused, the encoder
        does not stay loaded on the shared memory for the ten minutes of refusals."""
        eng = DyingEngine("boom")
        vision = types.SimpleNamespace(starts=0, unloads=0)
        vision.restart = lambda: setattr(vision, "starts", vision.starts + 1)
        vision.unload = lambda: setattr(vision, "unloads", vision.unloads + 1)
        svc = service(eng)
        svc.vision = vision
        svc._vision_down = lambda: True                 # an on-demand encoder: each request finds it stopped
        refused = self.ask(svc, 5)
        self.assertEqual(len(refused), 3)
        self.assertEqual(vision.starts, 5)
        self.assertEqual(vision.unloads, 3, "each refusal stops the encoder it had started")


class CpuReader(unittest.TestCase):
    """An elastic picture reader on the CPU (no "gpu" in the vision section) uses no VRAM: its start does not ask the
    expert cache to shrink."""

    def test_a_cpu_reader_starts_without_asking_the_cache(self):
        asked = []
        vision = types.SimpleNamespace(elastic=True, cpu_only=True, elastic_mib=2048, starts=0, alive=lambda: False)
        vision.restart = lambda: setattr(vision, "starts", vision.starts + 1)
        engine = types.SimpleNamespace(vram=asked.append)
        svc = types.SimpleNamespace(vision=vision, engine=engine, loaded=lambda: True, vision_held=False)
        S.Service._vision_up(svc)
        self.assertEqual(vision.starts, 1)
        self.assertEqual(asked, [])


if __name__ == "__main__":
    unittest.main()
