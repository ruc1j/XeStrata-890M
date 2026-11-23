# SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
# SPDX-License-Identifier: LGPL-3.0-or-later
"""serve/test_inkling2mini_serve.py - the server with the Inkling-2-Mini engine (serve/inkling2mini_kind.py): what it
sends the engine and how it reads the engine's channels, against a scripted engine (no GPU, no model file).

    python -m unittest serve.test_inkling2mini_serve -v
"""
from __future__ import annotations

import json
import sys
import tempfile
import unittest
import urllib.error
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from serve import inkling2mini_kind  # noqa: E402
from serve.frontend import ChatTemplate  # noqa: E402
from serve.server import ByteTokenizer, HOLD_KEY, Service, StrataEngine, engine_args, serve  # noqa: E402
from serve.test_tool_markup_retry import Scripted  # noqa: E402

# the shape of the model's chat template: channels, tools declared as XML, a call written as JSON
TEMPLATE = (
    "<|message_system|><|content_text|>Thinking effort level: "
    "{% if reasoning_effort == 'none' %}0{% else %}{{ reasoning_effort }}{% endif %}<|end_message|>"
    "{%- if tools -%}<|message_system|>tool_declare<|content_xml|>{{ tools | tojson }}<|end_message|>{%- endif -%}"
    "{%- for m in messages -%}"
    "{%- if m.role == 'user' -%}<|message_user|><|content_text|>{{ m.content }}<|end_message|>"
    "{%- elif m.role == 'assistant' -%}"
    "{%- if m.tool_calls -%}{%- for tc in m.tool_calls -%}<|message_model|>{{ tc.function.name }}"
    "<|content_invoke_tool_json|>{{ tc.function.arguments | tojson }}<|end_message|>{%- endfor -%}{%- endif -%}"
    "{%- if m.content -%}<|message_model|><|content_text|>{{ m.content }}<|end_message|>{%- endif -%}"
    "{%- elif m.role == 'tool' -%}<|message_tool|><|content_text|>{{ m.content }}<|end_message|>{%- endif -%}"
    "{%- endfor -%}"
)
TOOLS = [{"type": "function", "function": {"name": "get_weather", "parameters": {
    "type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}}}]


class ChannelEngine(Scripted):
    """The scripted engine, marked as the Inkling-2-Mini one; the sampling of each pass is kept."""
    inkling2mini_engine = True

    def __init__(self, tok, texts):
        super().__init__(tok, texts)
        self.samplings: list = []
        self.info = {"images": 1}

    def generate(self, ids, max_new, sampling, cancel, embeddings=None):
        self.samplings.append(dict(sampling or {}))
        yield from super().generate(ids, max_new, sampling, cancel, embeddings)


class SamplingKeys(unittest.TestCase):
    def test_only_the_keys_the_engine_takes(self):
        sampling = {"temperature": 0.7, "top_p": 0.9, "seed": 3, "strata_checkpoint": False, HOLD_KEY: (64, 5),
                    "strata_tune": {"spec_min_p": 0.5}, "logprobs": True, "top_logprobs": 2,
                    "strata_inkling2mini_images": ["/tmp/a b.png", "/tmp/c%d.jpg"]}
        keys = StrataEngine.sampling_keys(sampling, inkling2mini=True).split()
        self.assertEqual(keys, ["temperature=0.7", "top_p=0.9", "seed=3", "image=/tmp/a%20b.png",
                                "image=/tmp/c%25d.jpg"])
        # the engine reads the path back as it was
        self.assertEqual(inkling2mini_kind.decode_path("/tmp/a%20b.png"), "/tmp/a b.png")
        self.assertEqual(inkling2mini_kind.decode_path("/tmp/c%25d.jpg"), "/tmp/c%d.jpg")
        # the Flash-Next engine's keys are as before
        other = StrataEngine.sampling_keys(sampling)
        self.assertIn("ckpt=0", other)
        self.assertNotIn("image=", other)

    def test_the_engines_own_arguments(self):
        cfg = {"exe": "/opt/x/strata-inkling-2-mini", "parallel": 4, "vision": {"gpu": True}, "coupled_draft": True,
               "args": ["--native", "m.strata", "--max-context", "32768"]}
        self.assertEqual(engine_args(cfg), ["--native", "m.strata", "--max-context", "32768", "--parallel", "4"])
        # another engine's config keeps the Flash-Next handling
        self.assertIn("--coupled-draft", engine_args({"exe": "/opt/x/strata", "coupled_draft": True, "args": []}))


class Chat(unittest.TestCase):
    def start(self, texts):
        self.dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.dir.cleanup)
        tpl = Path(self.dir.name) / "chat_template.jinja"
        tpl.write_text(TEMPLATE, encoding="utf-8")
        self.tok = ByteTokenizer()
        self.engine = ChannelEngine(self.tok, texts)
        self.svc = Service(self.engine, self.tok, ChatTemplate(tpl), model_name="inkling-2-mini-sq-10")
        self.httpd = serve(self.svc, port=0)
        self.base = f"http://127.0.0.1:{self.httpd.server_address[1]}"
        self.addCleanup(self.httpd.server_close)
        self.addCleanup(self.httpd.shutdown)

    def post(self, body):
        req = urllib.request.Request(self.base + "/v1/chat/completions", data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=30) as r:
            raw = r.read().decode()
            return raw if body.get("stream") else json.loads(raw)

    def prompt(self, i=-1) -> str:
        return bytes(self.engine.prompts[i]).decode("utf-8", "replace")

    def test_text_in_its_channel(self):
        self.start(["<|message_model|><|content_text|>The capital is Tokyo.<|end_message|>"
                    "<|content_model_end_sampling|>"])
        self.assertTrue(self.svc.inkling2mini)
        b = self.post({"model": "m", "max_tokens": 200, "messages": [{"role": "user", "content": "capital?"}]})
        msg = b["choices"][0]["message"]
        self.assertEqual(msg["content"], "The capital is Tokyo.")
        self.assertFalse(msg.get("reasoning_content"))
        self.assertEqual(b["choices"][0]["finish_reason"], "stop")
        # the prompt: the model's own template, thinking off whatever the request asks
        self.assertIn("Thinking effort level: 0<|end_message|>", self.prompt())
        self.post({"model": "m", "max_tokens": 200, "reasoning_effort": "high",
                   "messages": [{"role": "user", "content": "capital?"}]})
        self.assertIn("Thinking effort level: 0<|end_message|>", self.prompt())

    def test_the_models_own_thinking_is_not_shown(self):
        self.start(["<|message_model|><|content_thinking|>The user asks a capital.<|end_message|>"
                    "<|message_model|><|content_text|>Tokyo.<|end_message|><|content_model_end_sampling|>"])
        b = self.post({"model": "m", "max_tokens": 200, "messages": [{"role": "user", "content": "capital?"}]})
        msg = b["choices"][0]["message"]
        self.assertEqual(msg["content"], "Tokyo.")
        self.assertFalse(msg.get("reasoning_content"))

    def test_a_tool_call_in_its_channel(self):
        self.start(['<|message_model|>get_weather<|content_invoke_tool_json|>{"name":"get_weather","args":'
                    '{"city":"Paris </tool_call> & <b>"}}<|end_message|><|content_model_end_sampling|>'])
        b = self.post({"model": "m", "max_tokens": 200, "tools": TOOLS,
                       "messages": [{"role": "user", "content": "weather in Paris?"}]})
        c = b["choices"][0]
        self.assertEqual(c["finish_reason"], "tool_calls")
        calls = c["message"]["tool_calls"]
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0]["function"]["name"], "get_weather")
        self.assertEqual(json.loads(calls[0]["function"]["arguments"]), {"city": "Paris </tool_call> & <b>"})
        self.assertIn("tool_declare<|content_xml|>", self.prompt())
        # streamed: the same call, and no channel tag in the text
        raw = self.post({"model": "m", "max_tokens": 200, "tools": TOOLS, "stream": True,
                         "messages": [{"role": "user", "content": "weather in Paris?"}]})
        chunks = [json.loads(x[6:]) for x in raw.splitlines() if x.startswith("data: {")]
        text = "".join(ch["choices"][0]["delta"].get("content") or "" for ch in chunks if ch.get("choices"))
        self.assertNotIn("<|", text)
        args = "".join(tc["function"].get("arguments") or "" for ch in chunks if ch.get("choices")
                       for tc in ch["choices"][0]["delta"].get("tool_calls") or [])
        self.assertEqual(json.loads(args), {"city": "Paris </tool_call> & <b>"})

    def test_a_history_call_is_rendered_with_its_arguments_as_an_object(self):
        self.start(["<|message_model|><|content_text|>It is sunny.<|end_message|><|content_model_end_sampling|>"])
        self.post({"model": "m", "max_tokens": 200, "tools": TOOLS, "messages": [
            {"role": "user", "content": "weather in Paris?"},
            {"role": "assistant", "content": None, "tool_calls": [{"id": "c1", "type": "function", "function": {
                "name": "get_weather", "arguments": json.dumps({"city": "Paris"})}}]},
            {"role": "tool", "tool_call_id": "c1", "content": "sunny"}]})
        self.assertIn('<|message_model|>get_weather<|content_invoke_tool_json|>{"city": "Paris"}<|end_message|>',
                      self.prompt())

    def test_logprobs_are_refused(self):
        self.start(["<|message_model|><|content_text|>x<|end_message|>"])
        with self.assertRaises(urllib.error.HTTPError) as e:
            self.post({"model": "m", "max_tokens": 4, "logprobs": True, "top_logprobs": 2,
                       "messages": [{"role": "user", "content": "hi"}]})
        self.assertEqual(e.exception.code, 400)
        self.assertIn("Inkling-2-Mini", e.exception.read().decode())
        e.exception.close()

    def test_the_flash_next_keys_do_not_reach_the_engine(self):
        self.start(["<|message_model|><|content_text|>ok<|end_message|><|content_model_end_sampling|>"])
        self.post({"model": "m", "max_tokens": 50, "temperature": 0.5, "messages": [{"role": "user", "content": "hi"}]})
        keys = StrataEngine.sampling_keys(self.engine.samplings[-1], inkling2mini=True)
        for word in keys.split():
            self.assertIn(word.split("=")[0], ("temperature", "top_p", "top_k", "min_p", "penalty_repeat",
                                               "penalty_freq", "penalty_present", "penalty_last_n", "seed", "image"))


if __name__ == "__main__":
    unittest.main()
