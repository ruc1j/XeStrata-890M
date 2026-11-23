# SPDX-FileCopyrightText: 2026 Niko1221 and the Strata contributors
# SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
# SPDX-License-Identifier: LGPL-3.0-or-later AND MIT
"""serve/frontend.py - plan v0.3 P8: the request/response layer between API clients and the engine.

Everything here is text: no model, no GPU. The engine consumes token ids and produces text deltas; this module

  * normalizes OpenAI Chat Completions and Anthropic Messages requests into the chat template's message form,
  * renders the model's own Jinja chat template (pack `tokenizer/chat_template.jinja`) exactly as Hugging Face
    does (checked against the pack's `chat_golden.json`, 10 cases incl. thinking and tools),
  * parses the streamed output incrementally into reasoning (`<think>...</think>`), content and tool calls in
    the template's XML form (`<tool_call><function=NAME><parameter=P>VALUE</parameter>...</function></tool_call>`),
    never emitting a partial tag to the client.

Design note: plan v0.3 describes a single C++ process. The frontend is Python first because the template is Jinja
and the formats change often; the engine boundary is token ids in, text deltas out, which keeps a later C++ port
(or a pybind wrapper) a local change.
"""
from __future__ import annotations

import base64
import datetime
import json
import logging
import math
import mimetypes
import os
import re
import threading
import uuid
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import unquote

import jinja2
import jinja2.ext
from jinja2.sandbox import ImmutableSandboxedEnvironment

LOGGER = logging.getLogger(__name__)


# ------------------------------------------------------------------------------------------------ template
class _GenerationExtension(jinja2.ext.Extension):
    """Hugging Face's `{% generation %}...{% endgeneration %}` tag (Inkling-2-Mini's chat template has it): the block
    is rendered as it is, the tag itself writes nothing.  From upstream Strata's serve/frontend.py."""
    tags = {"generation"}

    def parse(self, parser):
        next(parser.stream)
        return parser.parse_statements(("name:endgeneration",), drop_needle=True)


class TemplateRequestError(jinja2.exceptions.TemplateError, ValueError):
    """The template refused the request's messages (e.g. "No user query found in messages."): a ValueError, so the
    client gets a 400 with the template's message instead of a dropped connection (#365)."""


class ChatTemplate:
    """The model's chat template, rendered with the same Jinja settings as transformers' apply_chat_template."""

    def __init__(self, path: str | Path):
        def raise_exception(message):
            raise TemplateRequestError(message)

        def tojson(x, ensure_ascii=False, indent=None, separators=None, sort_keys=False):
            return json.dumps(x, ensure_ascii=ensure_ascii, indent=indent, separators=separators, sort_keys=sort_keys)

        def strftime_now(fmt):   # transformers' global for templates that write today's date (llm-jp-4.1's)
            return datetime.datetime.now().strftime(fmt)

        env = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True,
                                            extensions=["jinja2.ext.loopcontrols", _GenerationExtension])
        env.filters["tojson"] = tojson
        env.globals["raise_exception"] = raise_exception
        env.globals["strftime_now"] = strftime_now
        self.source = Path(path).read_text(encoding="utf-8")
        self.template = env.from_string(self.source)
        # the Harmony format's templates (llm-jp-4.1, GPT-OSS): its replies are channels, read by HarmonyParser
        self.harmony = "<|channel|>" in self.source
        # the tools as OpenAI sends them ({"type": "function", "function": {...}}) where the template unwraps them
        # itself; otherwise their inner objects, as Qwen's templates read them
        self.wrapped_tools = re.search(r"tool\.function\b|tool\[['\"]function['\"]\]", self.source) is not None
        # GLM's templates (GLM-5.3-Flash): a call is `<tool_call>NAME<arg_key>K</arg_key><arg_value>V</arg_value>...
        # </tool_call>`, read through GlmCallRewriter; the thinking always opened, its effort low, high or max
        self.call_form = "glm" if "<arg_key>" in self.source else "qwen"
        # Inkling-2-Mini's template (serve/inkling2mini_dialect.py): tools declared under
        # `tool_declare<|content_xml|>`, a call written in a channel as JSON; the tool calls' arguments rendered as
        # objects, thinking always off
        self.inkling2mini = ("<|content_invoke_tool_json|>" in self.source
                             or "tool_declare<|content_xml|>" in self.source)
        self.caps = self._detect_caps()
        self.thinking = self._thinking_levels()

    def _thinking_levels(self) -> dict:
        """The thinking levels this template takes, by their own names, and the one a request that names none gets
        (/health's "thinking", the web page's buttons): GLM-5.3-Flash's low, high and max; Harmony's (llm-jp-4.1's)
        low, medium and high, medium its default; Qwen3.8-Flash-Next's low, medium and high (xhigh); a template that
        only turns its thinking on and off, on; none at all, no levels."""
        if self.inkling2mini:                          # no thinking mode
            return {"levels": [], "default": None}
        if self.call_form == "glm":
            return {"levels": ["none", "low", "high", "max"], "default": "max"}
        if self.harmony:
            return {"levels": ["none", "low", "medium", "high"], "default": "medium"}
        if "reasoning_effort" in self.source:
            return {"levels": ["none", "low", "medium", "high"], "default": "high"}
        if "enable_thinking" in self.source:
            return {"levels": ["none", "on"], "default": "on"}
        return {"levels": [], "default": None}

    def render(self, messages: list[dict], tools: list[dict] | None = None, add_generation_prompt: bool = True,
               **kwargs) -> str:
        # Do not use completed empty assistant turns as examples for the next reply. Keep the final message.
        # (#843.  STRATA_KEEP_EMPTY_TURNS=1 renders them as before.)
        keep_empty = os.environ.get("STRATA_KEEP_EMPTY_TURNS") == "1"
        messages = [m for i, m in enumerate(messages)
                    if i == len(messages) - 1 or keep_empty or not (isinstance(m, dict) and m.get("role") == "assistant"
                                                                    and not _text_of(m.get("content")).strip()
                                                                    and not _has_image(m.get("content"))
                                                                    and not m.get("tool_calls"))]
        if self.inkling2mini:
            from serve import inkling2mini_dialect
            messages = inkling2mini_dialect.prepare_messages(messages)
            kwargs = inkling2mini_dialect.template_kwargs(kwargs)
        if self.harmony:   # as llama.cpp's LLM-jp-4.1 path: the reasoning in the field the template reads, `thinking`,
            # and no content beside the calls of a turn that made them
            adjusted = []
            for m in messages:
                if isinstance(m, dict) and isinstance(m.get("reasoning_content"), str):
                    m = {**m, "thinking": m["reasoning_content"]}
                    if m.get("tool_calls"):
                        m.pop("content", None)
                adjusted.append(m)
            messages = adjusted
        if tools and self.wrapped_tools:   # the template reads OpenAI's form, `tool.function` (llm-jp-4.1's)
            tools = [t if "function" in t else {"type": "function", "function": t} for t in tools]
        if self.call_form == "glm":        # its efforts: low, high and max (the default), taken by name
            effort = kwargs.get("reasoning_effort")
            kwargs = {**kwargs, "reasoning_effort": {"low": "low", "medium": "high", "high": "high"}.get(str(effort))}
        elif self.harmony and kwargs.get("reasoning_effort") == "xhigh":   # Harmony's efforts: low, medium and high
            kwargs = {**kwargs, "reasoning_effort": "high"}
        text = self.template.render(messages=messages, tools=tools, add_generation_prompt=add_generation_prompt,
                                    **kwargs)
        if (self.call_form == "glm" and add_generation_prompt and kwargs.get("enable_thinking", True) is False
                and text.endswith("<think>")):   # GLM's template always opens the thinking: closed at once
            text += THINK_END
        if (self.harmony and add_generation_prompt and kwargs.get("enable_thinking", True) is False
                and text.endswith(HARMONY_TURN)):
            # Harmony has no switch for thinking: an empty analysis message, so the model goes on with the final
            # answer or a call (a forced final channel would leave it no call)
            text += HARMONY_NO_THINKING + HARMONY_TURN
        return text

    def _detect_caps(self) -> dict[str, bool]:
        """llama.cpp's capability names, checked at load time against this template and Strata's tool-call format.
        These are rendering hints, not a guarantee that the model will follow a request."""
        def render(messages, tools=None):
            try:
                return self.render(messages, tools=tools)
            except Exception as exc:                # noqa: BLE001 - a hint only: never stop the server for a probe
                # A template may reject a role or feature (a custom one may fail in any way): the feature is off,
                # discovery still answers for the others, and the start-up goes on.
                LOGGER.debug("chat template capability probe failed: %s", exc, exc_info=True)
                return ""

        user = {"role": "user", "content": "strata_caps_user"}
        tools = [{"name": f"strata_caps_call_{i}", "description": "strata_caps_description",
                  "parameters": {"type": "object", "properties": {"arg": {"type": "string"}}}}
                 for i in range(2)]
        tool_prompt = render([user], tools)

        def calls_supported(count):
            calls = [{"name": f"strata_caps_call_{i}", "arguments": {"arg": f"strata_caps_arg_{i}"}}
                     for i in range(count)]
            messages = [user, {"role": "assistant", "content": "",
                               "tool_calls": [{"function": call} for call in calls]}]
            replies = [f"strata_caps_result_{i}" for i in range(count)]
            messages += [{"role": "tool", "content": reply} for reply in replies] + [user]
            prompt = render(messages, tools)
            # Instructions include example XML and literal tag names, which are not model output. Parse only
            # complete calls to our probe functions, using the same body parser as OutputParser.
            if self.call_form == "glm":
                found = re.findall(r"<tool_call>(strata_caps_call_\d+.*?)</tool_call>", prompt, re.S)
                bodies = [glm_call_body(b) for b in found]
            else:
                bodies = re.findall(r"<tool_call>\s*(<function=strata_caps_call_\d+>.*?</function>)\s*</tool_call>",
                                    prompt, re.S)
            try:
                parsed = [parse_tool_call(body) for body in bodies]
            except ValueError:
                return False
            return [{"name": call.name, "arguments": call.arguments} for call in parsed] == calls \
                and all(reply in prompt for reply in replies)

        if self.inkling2mini:   # its own spelling of the same capabilities (upstream's probe for this template)
            def channel_calls(count):
                calls = [{"name": f"strata_caps_call_{i}", "arguments": {"arg": f"strata_caps_arg_{i}"}}
                         for i in range(count)]
                messages = [user, {"role": "assistant", "content": "",
                                   "tool_calls": [{"function": call} for call in calls]}]
                messages += [{"role": "tool", "name": f"strata_caps_call_{i}", "content": f"strata_caps_result_{i}"}
                             for i in range(count)] + [user]
                prompt = render(messages, tools)
                return all(f"<|message_model|>strata_caps_call_{i}<|content_invoke_tool_json|>" in prompt
                           and f"strata_caps_result_{i}" in prompt for i in range(count))
            return {"supports_tools": all(s in tool_prompt for s in ("strata_caps_call_0", "strata_caps_description",
                                                                     "tool_declare<|content_xml|>")),
                    "supports_tool_calls": channel_calls(1),
                    "supports_system_role": "strata_caps_system" in render(
                        [{"role": "system", "content": "strata_caps_system"}, user]),
                    "supports_parallel_tool_calls": channel_calls(2),
                    "supports_preserve_reasoning": False}
        history = [user, {"role": "assistant", "content": "strata_caps_answer",
                          "reasoning_content": "strata_caps_reasoning"}, user]
        form_tag = "<arg_key>" if self.call_form == "glm" else "<function="
        return {"supports_tools": all(s in tool_prompt for s in
                                      ("strata_caps_call_0", "strata_caps_description", "<tool_call>", form_tag)),
                "supports_tool_calls": calls_supported(1),
                "supports_system_role": "strata_caps_system" in render(
                    [{"role": "system", "content": "strata_caps_system"}, user]),
                "supports_parallel_tool_calls": calls_supported(2),
                "supports_preserve_reasoning": "strata_caps_reasoning" in render(history)}


# ------------------------------------------------------------------------------------------------ requests
# The content-part types the readers take as text (a part with no type that has a "text" field reads as text too)
TEXT_PARTS = ("text", "input_text", None)


def _text_of(content) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    return "".join(part.get("text", "") for part in content if isinstance(part, dict) and part.get("type") in
                   TEXT_PARTS)


IMAGE_PARTS = ("image_url", "input_image", "image")

# Thinking levels.  Qwen3.8-Flash-Next's template knows low, medium and xhigh (its default; "high" means xhigh), and
# enable_thinking=false for none.  Clients spell these many ways; everything maps onto none, low, medium, high and
# xhigh, high kept apart from xhigh for the templates whose high is below their top (GLM-5.3-Flash's max).
EFFORT = {"none": None, "off": None, "minimal": None, "disabled": None, "false": None,
          "low": "low", "medium": "medium", "high": "high", "xhigh": "xhigh", "max": "xhigh", "maximum": "xhigh"}


def effort_kwargs(value) -> dict:
    """A reasoning effort as given by a client -> the template's kwargs.  Unknown values are a 400, not a crash."""
    if value is None or value == "":
        return {}
    if value is False:
        return {"enable_thinking": False}
    key = str(value).strip().lower()
    if key not in EFFORT:
        raise ValueError(f"unknown reasoning effort {value!r}: use none, low, medium or high")
    level = EFFORT[key]
    return {"enable_thinking": False} if level is None else {"reasoning_effort": level}


def budget_effort(tokens) -> dict:
    """Anthropic's thinking budget (budget_tokens) -> a level: under 2K low, under 8K medium, else high."""
    try:
        n = int(tokens)
    except (TypeError, ValueError):
        return {}
    return {"reasoning_effort": "low" if n < 2048 else "medium" if n < 8192 else "xhigh"}


# A file a request sends (Chat's "file", Responses' "input_file", an Anthropic PDF document as converted by
# _anthropic_documents): a "file" item of the template's content until Service.expand_files reads it into text and
# page images
FILE_PARTS = ("file", "input_file")


def _has_image(content) -> bool:
    """Content that must stay a list of parts: a picture in it (or a file, which becomes text and pictures)."""
    return isinstance(content, list) and any(isinstance(p, dict) and p.get("type") in IMAGE_PARTS + FILE_PARTS
                                             for p in content)


def file_item(part: dict, path: str) -> dict:
    """A file part -> {"type": "file", "data": a data: URL or None, "url": an http(s) URL or None, "name", "mime"}.
    A file_id (OpenAI's and Anthropic's Files APIs) is a ValueError: this server keeps no files."""
    inner = part.get("file")
    src: dict = inner if part.get("type") == "file" and isinstance(inner, dict) else part
    if src.get("file_id"):
        raise ValueError(f"{path}: a file_id needs a Files API, which this server does not have: send the file "
                         "itself (file_data)")
    data, url, name = src.get("file_data"), src.get("file_url"), str(src.get("filename") or "")
    if not (isinstance(data, str) and data) and not (isinstance(url, str) and url):
        raise ValueError(f"{path}: a file needs file_data (a data: URL or base64) or file_url")
    if isinstance(data, str) and data and not data.startswith("data:"):
        guess = mimetypes.guess_type(name)[0] or "application/octet-stream"   # bare base64: the name's type
        data = f"data:{guess};base64,{data}"
    mime = data[5:data.find(";")] if isinstance(data, str) and data and ";" in data else None
    return {"type": "file", "data": data or None, "url": None if data else url, "name": name, "mime": mime}


def _image_source(part: dict) -> str:
    """An image part's source as one string: a data: URL, an http(s) URL or a local file path.
    OpenAI: {"type": "image_url", "image_url": {"url": ...}} (or "image_url": "..."), Responses-style
    {"type": "input_image", "image_url": ...}; Anthropic: {"type": "image", "source": {"type": "base64",
    "media_type": ..., "data": ...}} or {"source": {"type": "url", "url": ...}}."""
    if part.get("type") == "image":
        src = part.get("source") or {}
        if src.get("type") == "base64":
            return f"data:{src.get('media_type', 'image/png')};base64,{src.get('data', '')}"
        return src.get("url") or src.get("path") or ""
    url = part.get("image_url")
    if isinstance(url, dict):
        url = url.get("url")
    return url or ""


def _parts_of(content):
    """Message content for the template: a string when there is no image (unchanged behaviour), otherwise the
    template's list form - text items and image items, in order - whose image items carry their source."""
    if not _has_image(content):
        return _text_of(content)
    items = []
    for part in content:
        if not isinstance(part, dict):
            continue
        if part.get("type") in FILE_PARTS:
            items.append(file_item(part, "content"))
        elif part.get("type") in IMAGE_PARTS:
            item = {"type": "image", "source": _image_source(part)}
            url = part.get("image_url")
            if (url.get("detail") if isinstance(url, dict) else part.get("detail")) == "low":
                item["detail"] = "low"               # OpenAI's low detail: read at a smaller size (Service.prepare)
            items.append(item)
        elif part.get("type") in TEXT_PARTS and "text" in part:
            items.append({"type": "text", "text": part.get("text", "")})
    return items


# The blocks a messages request may carry: text, the image parts, thinking, tool_use and tool_result.
# redacted_thinking is allowed only to be dropped without an error: another server's encrypted reasoning, which
# carries nothing this server can read - the same reason responses.py's _reasoning_text gives for another server's
# encrypted_content.
ANTHROPIC_BLOCKS = ("text",) + IMAGE_PARTS + FILE_PARTS + ("thinking", "tool_use", "tool_result", "redacted_thinking",
                                                           "tool_addition")


def _check_parts(content, path: str, kinds):
    """A request's content parts (on chat) or content blocks (on messages), looked at while the request is converted.
    A part whose type this server does not read (a file, audio, a document, Anthropic's tool_reference,
    search_result, server_tool_use / web_search_tool_result blocks, ...) is left out of the prompt exactly as it always
    was, with a debug line: real clients (Claude Code, Codex, opencode, Cline, Continue, Open WebUI) send such parts
    next to text the model can answer from, so refusing them would break requests that work.  A "refusal" part is read
    as its text, as /v1/responses does.  A part that is not an object inside the array passes through unchanged.
    Content itself must be a string or an array of parts, and a text part's "text" a string: another shape (a number,
    a boolean, an object, a non-string text) is a 400 naming the field, as messages (#460) and tools (#592) are,
    instead of a TypeError later in _text_of / the messages join, caught by the generic malformed-request handler."""
    if content is None or isinstance(content, str):
        return content
    if not isinstance(content, list):
        raise ValueError(f"{path} must be a string or an array of content parts")
    out = content
    for j, part in enumerate(content):
        if not isinstance(part, dict):
            continue
        kind = part.get("type")
        if kind in TEXT_PARTS and "text" in part and not isinstance(part["text"], str):
            raise ValueError(f"{path}[{j}].text must be a string")
        if kind == "refusal":
            if out is content:
                out = list(content)
            refusal = part.get("refusal") or ""
            if not isinstance(refusal, str):
                raise ValueError(f"{path}[{j}].refusal must be a string")
            out[j] = {"type": "text", "text": refusal}
        elif kind not in kinds:
            LOGGER.debug("%s[%d].type: content part of type %r is not read by this server; left out of the prompt",
                         path, j, kind)
    return out


def images_of(messages: list[dict]) -> list[str]:
    """The image sources of the rendered conversation, in prompt order (the template renders one
    <|vision_start|><|image_pad|><|vision_end|> per image item, message by message)."""
    return [item["source"] for m in messages if isinstance(m.get("content"), list)
            for item in m["content"] if item.get("type") == "image"]


# #537: a literal <think> / </think> inside a message's text is plain text, not the model's reasoning markers.  The
# tokenizer matches those two strings as their special tokens everywhere (GGUF token type 4, as llama.cpp does), so
# a user quoting "</think>" used to hand the model a real end-of-reasoning token.  Before the template is rendered
# they are swapped for these private-use characters, and the server encodes the spans they mark as ordinary text.
THINK_TAGS = {"<think>": "\U000F0E01", "</think>": "\U000F0E02"}
# #554: the vision markers the same way.  They are special tokens only where the template writes them for an image
# item; the same strings in a message's text (an agent reading chat_template.jinja, a tool result quoting it) became
# the same special ids, so text with <|vision_start|><|image_pad|> before a picture took that picture's embeddings
VISION_TAGS = {"<|vision_start|>": "\U000F0E03", "<|image_pad|>": "\U000F0E04", "<|vision_end|>": "\U000F0E05",
               "<|video_pad|>": "\U000F0E06"}
# #2026: `</tool_call>` the same way.  A user asking for the string verbatim handed the model the real end-of-call token
# (type 4, matched everywhere); it copied that token into its answer and the reply came back empty.  An assistant
# message keeps its own: a client replaying the model's raw call text sends the tokens the model wrote.
# 0.1.44: `<tool_call>`, `<tool_response>` and `</tool_response>` are type 4 too (ids 248058, 248066, 248070), so a
# quoted one handed the model the real call / tool-result marker in the same way (found by 47Hunter47 in #2026).  The
# template writes the tool-result markers around a tool message itself; only the text inside a message is marked.
TOOL_LITERAL_TAGS = {"</tool_call>": "\U000F0E07", "<tool_call>": "\U000F0E08", "<tool_response>": "\U000F0E09",
                     "</tool_response>": "\U000F0E0A"}
LITERAL_TAGS = {**THINK_TAGS, **VISION_TAGS, **TOOL_LITERAL_TAGS}
THINK_MARKS = {v: k for k, v in LITERAL_TAGS.items()}
CONTROL_MARK0 = 0xF0E10      # the control tokens' marks start here, clear of the two sets above


def literal_tags(controls) -> dict[str, str]:
    """LITERAL_TAGS plus a mark for each control token's text (`controls`: the tokenizer's CONTROL literals,
    <|im_start|>, <|im_end|>, <|endoftext|>, ...).  The rendered prompt is encoded with those literals parsed, so one
    written inside a message - a file an agent reads, a pasted chat template - opened or ended a turn there.  Longest
    first, as the tokenizer matches them: a literal inside a longer one is not marked before it."""
    own = [c for c in controls if c not in LITERAL_TAGS]
    tags = {**LITERAL_TAGS, **{c: chr(CONTROL_MARK0 + k) for k, c in enumerate(own)}}
    return dict(sorted(tags.items(), key=lambda t: -len(t[0])))


def _mark(text: str, tags: dict[str, str] = LITERAL_TAGS) -> str:
    for tag, mark in tags.items():
        text = text.replace(tag, mark)
    return text


def _mark_deep(v, tags: dict[str, str] = LITERAL_TAGS):
    if isinstance(v, str):
        return _mark(v, tags)
    if isinstance(v, dict):
        return {k: _mark_deep(x, tags) for k, x in v.items()}
    if isinstance(v, list):
        return [_mark_deep(x, tags) for x in v]
    return v


def _has_tag(v, tags: dict[str, str] = LITERAL_TAGS) -> bool:
    if isinstance(v, str):
        return any(tag in v for tag in tags)
    if isinstance(v, dict):
        return any(_has_tag(x, tags) for x in v.values())
    if isinstance(v, list):
        return any(_has_tag(x, tags) for x in v)
    return False


_TOOL_RESULT_OPEN, _TOOL_RESULT_CLOSE = "<tool_response>", "</tool_response>"


def _keep_tool_result_wrap(content, tags: dict[str, str]):
    """A user message whose whole text is <tool_response>...</tool_response> (a client that formats a tool result the
    way the template does for a tool message) keeps that outer pair as the markers - the template tells such a turn from
    a real question by exactly that pair - and only the text inside is marked.  None: not such a message."""
    o, c = _TOOL_RESULT_OPEN, _TOOL_RESULT_CLOSE

    def wrap(text: str, first: bool, last: bool) -> str:
        i = text.index(o) + len(o) if first else 0
        j = text.rindex(c) if last else len(text)
        head = _mark(text[:i - len(o)], tags) + o if first else ""
        tail = c + _mark(text[j + len(c):], tags) if last else ""
        return head + _mark(text[i:j], tags) + tail

    if isinstance(content, str):
        body = content.strip()
        if len(body) >= len(o) + len(c) and body.startswith(o) and body.endswith(c):
            return wrap(content, True, True)
        return None
    if not (isinstance(content, list) and content and all(isinstance(x, dict) and "text" in x for x in content)):
        return None
    texts = [x["text"] for x in content]
    if not all(isinstance(t, str) for t in texts):
        return None
    body = "".join(texts).strip()
    if not (len(body) >= len(o) + len(c) and body.startswith(o) and body.endswith(c)):
        return None
    if not (texts[0].lstrip().startswith(o) and texts[-1].rstrip().endswith(c)):
        return None          # the pair is split across parts: mark it all, as before
    if len(content) == 1:
        return [{**content[0], "text": wrap(texts[0], True, True)}]
    return ([{**content[0], "text": wrap(texts[0], True, False)}] +
            [_mark_deep(x, tags) for x in content[1:-1]] +
            [{**content[-1], "text": wrap(texts[-1], False, True)}])


def mark_think_literals(messages: list[dict], tools: list[dict] | None, tags: dict[str, str] = LITERAL_TAGS):
    """#537: (messages, tools) with every literal <think> / </think> (#554: and vision marker) in their text swapped for
    LITERAL_TAGS' marks (or `tags`': literal_tags() adds the control tokens' texts), and
    whether there was one (None: no change, the same objects back - a prompt without them renders as it always did).
    An assistant message whose content opens with a whole <think>...</think> block (clients that send the reasoning
    inline) keeps that one block as the model's markers, as before."""
    if not _has_tag(messages, tags) and not _has_tag(tools, tags):
        return messages, tools, False
    out = []
    own = {t: v for t, v in tags.items() if t not in TOOL_LITERAL_TAGS}     # what an assistant message keeps as written
    for m in messages:
        m = dict(m)
        content = m.get("content")
        mine = own if m.get("role") == "assistant" else tags
        for k, v in m.items():
            if k != "role":
                m[k] = _mark_deep(v, mine)
        if m.get("role") == "user":
            kept = _keep_tool_result_wrap(content, tags)
            if kept is not None:
                m["content"] = kept
        if m.get("role") == "assistant" and isinstance(content, str) and content.lstrip().startswith("<think>") \
                and "</think>" in content:
            i, j = content.index("<think>") + len("<think>"), content.index("</think>")
            m["content"] = (content[:i] + _mark(content[i:j], own) + "</think>" +
                            _mark(content[j + len("</think>"):], own))
        out.append(m)
    return out, _mark_deep(tools, tags), True


_THINK_MARK_RE = re.compile("|".join(THINK_MARKS))


def unmark_think_literals(prompt: str, tags: dict[str, str] = LITERAL_TAGS) -> tuple[str, list[tuple[int, int]]]:
    """The rendered prompt with THINK_TAGS' marks turned back into the tags' text, and the (start, end) spans of those
    tags in it: the server encodes them as ordinary text (the tokenizer's encode_plain_spans)."""
    marks = THINK_MARKS if tags is LITERAL_TAGS else {v: k for k, v in tags.items()}
    mark_re = _THINK_MARK_RE if tags is LITERAL_TAGS else re.compile("|".join(map(re.escape, marks)))
    out, spans, pos, n = [], [], 0, 0
    for m in mark_re.finditer(prompt):
        out.append(prompt[pos:m.start()])
        n += m.start() - pos
        tag = marks[m.group(0)]
        out.append(tag)
        spans.append((n, n + len(tag)))
        n += len(tag)
        pos = m.end()
    out.append(prompt[pos:])
    return "".join(out), spans


# Config "env_to_user" (opt-in): opencode writes its <env> block (working directory, today's date) near the START of
# the system message, before ~40K characters of instructions and tool definitions that are the same in every session.
# The prefix cache only reuses a prompt's start, so a new folder or a new day read all of it again (~14K tokens,
# 8-11 s).  Moved to the front of the first user message, the system turn (and the tools the template writes into it)
# is the same for every session and the disk prefix cache restores it.  The model still sees the same lines, one turn
# later.
ENV_TO_USER = False
_ENV_BLOCK = re.compile(r"(?:[^\n]*environment you are running in:\n)?<env>\n.*?\n</env>\n?", re.S)


def _env_to_user(messages: list[dict]) -> list[dict]:
    if not ENV_TO_USER or not messages or messages[0].get("role") != "system":
        return messages
    sys_text = messages[0].get("content")
    if not isinstance(sys_text, str):
        return messages
    m = _ENV_BLOCK.search(sys_text)
    if m is None:
        return messages
    for i in range(1, len(messages)):
        if messages[i].get("role") != "user":
            continue
        block = m.group(0).rstrip("\n")
        c = messages[i].get("content")
        nc: str | list
        if isinstance(c, str):
            nc = block + "\n\n" + c
        elif isinstance(c, list):
            nc = [{"type": "text", "text": block + "\n\n"}] + c
        else:
            return messages
        out = list(messages)
        out[0] = dict(messages[0], content=sys_text[:m.start()] + sys_text[m.end():])
        out[i] = dict(messages[i], content=nc)
        return out
    return messages


def _late_system_to_user(messages: list[dict]) -> list[dict]:
    """The chat template takes a system message only at the start ("System message must be at the beginning").
    Clients also send them mid-conversation - Claude Code's hook context as {"role": "system"} after the first user
    turn, some OpenAI clients a late "developer" message (issue #56) - so those become user messages, in place:
    merging them into the first one would change the prompt's start and cost the conversation cache every turn."""
    return [dict(m, role="user") if m.get("role") == "system" and i > 0 else m for i, m in enumerate(messages)]


def _object_list(value, name: str) -> list[dict]:
    """#460: a request's "messages" (or a message's "tool_calls") as a list of objects.  Some clients send the array
    double-encoded, as a JSON string, which used to be iterated character by character and crashed on m.get: such a
    string is decoded.  Anything that is still not a list of objects is a ValueError, which the server answers with
    a 400 naming the field.  None (or no field) is an empty list, as a missing field always was."""
    if value is None:
        return []
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except ValueError:
            raise ValueError(f"{name} must be a list of objects (a string was sent that is not JSON)") from None
    if not isinstance(value, list) or not all(isinstance(m, dict) for m in value):
        raise ValueError(f"{name} must be a list of objects")
    return value


_TOOL_ORDERS: "OrderedDict[frozenset, tuple]" = OrderedDict()   # a set of tool names -> the order it first came in
_TOOL_ORDERS_LOCK = threading.Lock()
_TOOL_ORDERS_MAX = 64


def _stable_tool_order(tools: list[dict], name_of) -> list[dict]:
    """The same set of tools comes back in the order it was first seen.  Clients send their tools in the order their
    MCP servers finished registering, which can differ from one request (or one client restart) to the next, and the
    template renders the tool list before the system prompt: one swap moves the shared prefix back thousands of
    tokens and the whole conversation is read again (opencode issue 23571).  A request's first sight of a tool set
    is untouched (so its prompt is byte for byte what the client sent); later requests with the same names in
    another order are put back in that first order.  Duplicate names, or a single tool: left as sent."""
    if len(tools) < 2:
        return tools
    names = [name_of(t) for t in tools]
    if len(set(names)) != len(names):
        return tools
    key = frozenset(names)
    with _TOOL_ORDERS_LOCK:
        order = _TOOL_ORDERS.get(key)
        if order is None:
            _TOOL_ORDERS[key] = tuple(names)
            while len(_TOOL_ORDERS) > _TOOL_ORDERS_MAX:
                _TOOL_ORDERS.popitem(last=False)
            return tools
        _TOOL_ORDERS.move_to_end(key)
    pos = {n: k for k, n in enumerate(order)}
    return sorted(tools, key=lambda t: pos[name_of(t)])


def _tool_list(value, wrapper: str | None) -> list[dict]:
    """#592: a request's "tools" as a list of tool objects, each with a name - in the OpenAI shape
    {"type": "function", "function": {"name": ...}} (`wrapper` "function"; a bare {"name": ...} is still taken), or
    in the Anthropic shape {"name": ...} (`wrapper` None).  A value that is not (a string such as "auto", a list of
    names, an object without a name) is a ValueError - a 400 naming the field - where it used to take the request
    thread down with no reply at all.  No value (or an empty one) is no tools, as always.  A tool's schema -
    "parameters" in the OpenAI shape, "input_schema" in the Anthropic one - is an object when it is there, for the
    same reason: a string or a list passed and raised later instead, on the model's first call of that tool."""
    if not value:
        return []
    shape = ('{"type": "function", "function": {"name": ..., "parameters": {...}}}' if wrapper else
             '{"name": ..., "input_schema": {...}}')
    try:
        tools = _object_list(value, "tools")
    except ValueError:
        raise ValueError(f"tools must be a list of tool objects, each {shape}") from None
    key = "parameters" if wrapper else "input_schema"
    for i, t in enumerate(tools):
        fn = t.get(wrapper, t) if wrapper and t.get("type") == wrapper else t
        if not isinstance(fn, dict) or not isinstance(fn.get("name"), str) or not fn["name"]:
            raise ValueError(f"tools[{i}] has no name: each tool must be {shape}")
        schema = fn.get(key)
        if schema is not None and not isinstance(schema, dict):
            # it reached parse_tool_call and the stream parser as the tool's schema, whose .get("properties") raised
            # AttributeError in the request thread - after the 200 and whatever the model had said before the call
            raise ValueError(f'tools[{i}] ({fn["name"]}): "{key}" must be an object (the JSON schema of its '
                             f"parameters), not {type(schema).__name__}")
    return _stable_tool_order(tools,
                              lambda t: (t.get(wrapper, t) if wrapper and t.get("type") == wrapper else t)["name"])


def declared_parameters(tools, name):
    """The parameter names a request declares for the tool `name` (the OpenAI, Anthropic and Responses tool shapes, a
    Responses namespace included): a set, empty when the tool declares none; None when the request does not declare
    it."""
    for t in tools if isinstance(tools, list) else ():
        if not isinstance(t, dict):
            continue
        fn = t["function"] if isinstance(t.get("function"), dict) else t
        if isinstance(t.get("tools"), list):                          # a namespace of tools
            found = declared_parameters(t["tools"], name)
            if found is not None:
                return found
        own = fn.get("name")
        if isinstance(name, str) and name and (own == name or (isinstance(own, str) and name.endswith("." + own))):
            schema = fn.get("parameters") or fn.get("input_schema")
            props = schema.get("properties") if isinstance(schema, dict) else None
            return set(props) if isinstance(props, dict) else set()
    return None


def tool_arguments(raw, declared=None) -> dict:
    """A call's arguments from a client's history -> the mapping the template renders (all three APIs).  A JSON object
    (as text or already parsed) is used as it is; empty is {}.  Text that starts like an object but is not valid JSON, a
    call max_tokens cut short, gives the members it still holds (cut_arguments): the model sees the call it wrote, with
    its parameters, not a parameter called "arguments" it would then write itself (soak 11-17: after a dozen cut calls
    in a history the model wrote {"arguments": "<json>"} calls).  `declared` = the tool's parameter names
    (declared_parameters), None when unknown: a tool that really has a parameter "arguments" keeps the old form.
    Anything else (a list, a number, prose) is kept as text under "arguments" instead of refusing the whole conversation
    (#510)."""
    if raw is None or (isinstance(raw, str) and not raw.strip()):
        return {}
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str):
        try:
            value = json.loads(raw)
        except ValueError:
            if declared is None or "arguments" not in declared:
                cut = cut_arguments(raw)
                if cut is not None:
                    return cut
            return {"arguments": raw}
        return value if isinstance(value, dict) else {"arguments": raw}
    return {"arguments": json.dumps(raw, ensure_ascii=False)}


def require_messages(messages: list) -> None:
    """A chat request with no message at all is a 400 (OpenAI: "empty array. Expected an array with minimum length
    1"); the template would otherwise render just the assistant turn and the model would answer nothing."""
    if not messages:
        raise ValueError("messages must be a non-empty list: send at least one message")


def _order_chat_tool_results(messages: list[dict]) -> None:
    """The template pairs tool results with the calls by position, so a run of results the client returned in another
    order is put back in the order of the calls while their ids are still there (upstream 11237f831, #3395).  Only a
    complete, unambiguous run is reordered: the calls' and the results' ids non-empty unique strings, the same set.
    Anything else keeps the order received.  The ids are private to this conversion and never reach the template."""
    i = 0
    while i < len(messages):
        if messages[i]["role"] != "tool":
            i += 1
            continue
        j = i + 1
        while j < len(messages) and messages[j]["role"] == "tool":
            j += 1
        prev = messages[i - 1] if i else {}
        calls = [c.get("_call_id") for c in prev.get("tool_calls") or []] if prev.get("role") == "assistant" else []
        results = [m.get("_call_id") for m in messages[i:j]]
        if len(calls) == len(results) and all(isinstance(cid, str) and cid for cid in calls + results):
            if len(set(calls)) == len(calls) and len(set(results)) == len(results) and set(calls) == set(results):
                rank = {cid: n for n, cid in enumerate(calls)}
                messages[i:j] = sorted(messages[i:j], key=lambda m: rank[m["_call_id"]])
        i = j
    for message in messages:
        message.pop("_call_id", None)
        for call in message.get("tool_calls") or []:
            call.pop("_call_id", None)


def custom_tool_names(req: dict) -> set[str]:
    """The names of a Chat Completions request's custom tools ({"type": "custom", "custom": {"name": ...}}): their
    calls are answered as {"type": "custom", "custom": {"name", "input"}}."""
    tools = req.get("tools")
    return {t["custom"]["name"] for t in tools if isinstance(t, dict) and t.get("type") == "custom"
            and isinstance(t.get("custom"), dict) and isinstance(t["custom"].get("name"), str)} \
        if isinstance(tools, list) else set()


def _chat_tool(t):
    """A Chat Completions tool -> the template's.  A custom tool takes free text, maybe in a grammar the model reads
    (it is not enforced: the text is checked by nothing), as /v1/responses' custom tools."""
    if not isinstance(t, dict) or t.get("type") != "custom" or not isinstance(t.get("custom"), dict):
        return t.get("function", t) if isinstance(t, dict) and t.get("type") == "function" else t
    c = t["custom"]
    description = c.get("description") or ""
    fmt = c.get("format") if isinstance(c.get("format"), dict) else {}
    grammar = fmt.get("grammar") if isinstance(fmt.get("grammar"), dict) else fmt
    if fmt.get("type") == "grammar" and grammar.get("definition"):
        description += f"\n\nThe input must follow this {grammar.get('syntax', '')} grammar:\n{grammar['definition']}"
    return {"name": c.get("name"), "description": description,
            "parameters": {"type": "object", "required": ["input"],
                           "properties": {"input": {"type": "string", "description": "the raw input"}}}}


def openai_to_messages(req: dict) -> tuple[list[dict], list[dict] | None, dict]:
    """OpenAI Chat Completions -> (template messages, template tools, template kwargs)."""
    messages = []
    for i, m in enumerate(_object_list(req.get("messages"), "messages")):
        role = m.get("role")
        if role == "developer":
            role = "system"
        if role == "function":                       # the old shape of a tool's result: the call it answers by name
            role = "tool"
        raw_content = m.get("content")
        for j, part in enumerate(raw_content if isinstance(raw_content, list) else []):
            if isinstance(part, dict) and part.get("type") == "input_audio":
                raise ValueError(f"messages[{i}].content[{j}]: input_audio is not supported: this model does not "
                                 "read audio")
            if isinstance(part, dict) and part.get("type") == "file":
                file_item(part, f"messages[{i}].content[{j}]")   # a file_id or no data is a 400 here
        # an image (or a file) is read in a user, tool or assistant message; the system prompt takes text only, as
        # _text_of reads
        images = role in ("user", "tool", "assistant")
        content = _check_parts(m.get("content"), f"messages[{i}].content",
                               TEXT_PARTS + IMAGE_PARTS + FILE_PARTS if images else TEXT_PARTS)
        if role == "assistant" and content in (None, "") and isinstance(m.get("refusal"), str):
            content = m["refusal"]                   # a refusal the client sends back is what the model said
        out = {"role": role, "content": _parts_of(content) if images else _text_of(content)}
        name = m.get("name")
        if role in ("user", "system") and isinstance(name, str) and name:
            # the speaker's name, which the template has no place for: written before what they said
            said = out["content"]
            out["content"] = f"{name}: {said}" if isinstance(said, str) else \
                [{"type": "text", "text": f"{name}: "}] + list(said or [])
        if role == "tool":
            out["_call_id"] = m.get("tool_call_id")
        if m.get("reasoning_content"):
            out["reasoning_content"] = m["reasoning_content"]
        raw = m.get("tool_calls")
        if not raw and isinstance(m.get("function_call"), dict):   # the old shape of one call
            raw = [{"function": m["function_call"]}]
        if raw:
            calls = []
            for c in _object_list(raw, "tool_calls"):
                if c.get("type") == "custom" and isinstance(c.get("custom"), dict):
                    calls.append({"function": {"name": c["custom"].get("name"),
                                               "arguments": {"input": str(c["custom"].get("input") or "")}},
                                  "_call_id": c.get("id")})
                    continue
                fn = c.get("function", c)
                if not isinstance(fn, dict):
                    raise ValueError("tool_calls must be a list of objects (each with a \"function\" object)")
                params = declared_parameters(req.get("tools"), fn.get("name"))
                calls.append({"function": {"name": fn.get("name"),
                                           "arguments": tool_arguments(fn.get("arguments"), params)},
                              "_call_id": c.get("id")})
            out["tool_calls"] = calls
        messages.append(out)
    _order_chat_tool_results(messages)
    given = req.get("tools")
    plain = [t for t in given if not (isinstance(t, dict) and t.get("type") == "custom")] \
        if isinstance(given, list) else given
    listed = [_chat_tool(t) for t in _tool_list(plain, "function")]
    if isinstance(given, list):                      # the custom tools, after the function tools
        listed += [_chat_tool(t) for t in given if isinstance(t, dict) and t.get("type") == "custom"
                   and isinstance(t.get("custom"), dict) and isinstance(t["custom"].get("name"), str)]
    tools = listed or None
    kwargs = {}
    # OpenAI Chat Completions: "reasoning_effort"; Responses style: "reasoning": {"effort": ...}
    reasoning = req["reasoning"] if isinstance(req.get("reasoning"), dict) else {}
    given = req.get("reasoning_effort")
    # `false` is a level (thinking off, effort_kwargs): an `or` between the two fields dropped it
    kwargs.update(effort_kwargs(given if given is not None and given != "" else reasoning.get("effort")))
    # the vLLM / llama.cpp convention: {"chat_template_kwargs": {"enable_thinking": false, "reasoning_effort": "low"}}
    # A truthy non-object ("x", 5, [1]) has no .items(): name the field as a 400, as messages (#460) and tools (#592)
    # do, instead of an AttributeError caught by the generic malformed-request handler. Falsy values stay ignored.
    ctk = req.get("chat_template_kwargs") or {}
    if not isinstance(ctk, dict):
        raise ValueError("chat_template_kwargs must be an object")
    for k, v in ctk.items():
        if k == "enable_thinking" and not v:
            kwargs = {"enable_thinking": False}
        elif k == "reasoning_effort" and "enable_thinking" not in kwargs:
            kwargs.update(effort_kwargs(v))
    return _env_to_user(_late_system_to_user(messages)), tools, kwargs


BILLING_HEADER = "x-anthropic-billing-header:"


def pin_billing_stamp(system: str) -> str:
    """Claude Code starts its system prompt with `x-anthropic-billing-header: cc_version=2.1.170.bf4;
    cc_entrypoint=sdk-cli; cch=b145e;`.  cch changes on EVERY request and the version's 4th part on every session, so
    the prompt changed ~22K tokens in (after the tool list) on every turn and the conversation cache could only reuse
    up to its last 16K checkpoint: half of every agent prompt was read again.  Both stamps are pinned to f's, as
    llama.cpp does (ggml-org/llama.cpp#21793); only a header at the very start of the system text is touched, and
    only inside its first 160 characters."""
    if not system.startswith(BILLING_HEADER):
        return system
    s = list(system)
    cch = system.find("cch=", len(BILLING_HEADER))
    if 0 <= cch <= 160:
        v, end = cch + 4, system.find(";", cch + 4)
        if end > v and end - v <= 16:
            s[v:end] = "f" * (end - v)
    cv = system.find("cc_version=")
    if 0 <= cv <= 160:
        v, end = cv + len("cc_version="), system.find(";", cv)
        if v < end and end - v <= 64:
            parts = system[v:end].split(".")
            if len(parts) > 3:
                tail = v + len(".".join(parts[:3])) + 1
                s[tail:end] = "f" * (end - tail)
    return "".join(s)


# Anthropic's thinking signature, and the data of a redacted_thinking block: the thinking's text in the form of
# responses.py's encrypted_content (base64, not encrypted: the client already holds the text), so a block this
# server wrote can be read back and checked against its text.  Another server's signature is opaque and ignored.
THINKING_PREFIX = "strata.r1:"


def thinking_signature(text: str) -> str:
    return THINKING_PREFIX + base64.b64encode(json.dumps({"text": text}, ensure_ascii=False).encode()).decode("ascii")


def _signed_thinking(signature, where: str) -> str | None:
    """The text a signature of this server's carries; None for another server's (or none)."""
    if not isinstance(signature, str) or not signature.startswith(THINKING_PREFIX):
        return None
    try:
        return str(json.loads(base64.b64decode(signature[len(THINKING_PREFIX):], validate=True).decode())["text"])
    except (ValueError, KeyError, TypeError, UnicodeError):
        raise ValueError(f"{where}: the thinking's signature is damaged") from None


def _anthropic_documents(content, path: str):
    """Anthropic's text documents and search results -> text blocks the template reads: a document's text
    (source "text", or "content" blocks) under its title and context, a "content" document's images after it as image
    blocks of their own, a search result's text under its title and source.  A PDF document is left as it is (not
    read yet).  citations.enabled: true is a 400: no citations are made."""
    if not isinstance(content, list):
        return content
    out = []
    for j, block in enumerate(content):
        kind = block.get("type") if isinstance(block, dict) else None
        if kind in ("document", "search_result"):
            cite = block.get("citations")
            if isinstance(cite, dict) and cite.get("enabled") is True:
                raise ValueError(f"{path}[{j}].citations: citations are not supported by this server")
        text, images = None, []
        if kind == "document":
            src = block.get("source") if isinstance(block.get("source"), dict) else {}
            if src.get("type") == "text":
                text = str(src.get("data") or "")
            elif src.get("type") == "content":
                inner = src.get("content")
                text = inner if isinstance(inner, str) else "\n\n".join(
                    b.get("text") or "" for b in inner or [] if isinstance(b, dict) and b.get("type") == "text")
                images = [] if isinstance(inner, str) else [
                    b for b in inner or [] if isinstance(b, dict) and b.get("type") == "image"]
            if text is not None:
                head = [x for x in (block.get("title"), block.get("context")) if isinstance(x, str) and x]
                text = "\n".join(["Document: " + head[0]] + head[1:] + ["", text]) if head else text
            elif src.get("type") in ("base64", "url", "file"):   # a PDF: read by Service.expand_files
                name = str(block.get("title") or "")
                part: dict = {"type": "input_file", "filename": name}
                if src.get("type") == "base64":
                    part["file_data"] = f"data:{src.get('media_type') or 'application/pdf'};base64,{src.get('data')}"
                elif src.get("type") == "url":
                    part["file_url"] = str(src.get("url") or "")
                else:
                    part["file_id"] = src.get("file_id") or "?"
                file_item(part, f"{path}[{j}]")     # a file_id is a 400 here
                out.append(part)
                continue
        elif kind == "search_result":
            body = "\n\n".join(b.get("text") or "" for b in block.get("content") or []
                               if isinstance(b, dict) and b.get("type") == "text")
            text = f"Search result: {block.get('title') or ''} ({block.get('source') or ''})\n\n{body}"
        out.append({"type": "text", "text": text} if text is not None else block)
        out += images
    return out


def _anthropic_tool(t: dict) -> dict:
    """An Anthropic tool -> the template's form of it (also what the output parser reads its schema from)."""
    return {"name": t["name"], "description": t.get("description", ""), "parameters": t.get("input_schema", {})}


def anthropic_tools(req: dict) -> tuple[list[dict] | None, bool]:
    """#4663, Anthropic's tool search: (every tool the request declared, the `defer_loading` ones included, whether any
    tool is deferred).  A deferred tool is callable although the prompt does not list it.  A client that defers tools
    (Claude Code) holds back more than it declares - it names the rest in the conversation and runs a call for any of
    them - so with one deferred tool a call for an undeclared name is a call too (OutputParser's `deferred`)."""
    tools = _anthropic_callables(req)
    return [_anthropic_tool(t) for t in tools] or None, any(t.get("defer_loading") for t in tools)


def _anthropic_callables(req: dict) -> list[dict]:
    """#5952: the request's callable tools, each name once where it comes from a tool_addition: the top-level `tools`
    first (as declared, `defer_loading` ones included), then each name a system message's tool_addition adds, its
    FIRST definition.  A name the top level declares keeps the top-level definition (a later addition of it adds
    nothing).  The prompt shows an added tool with this same definition (anthropic_to_messages), so what the model
    reads and what the parser takes a call against are one schema."""
    tools = list(_tool_list(req.get("tools"), None))      # a copy: the request's own list is not changed
    names = {t["name"] for t in tools}
    for d in _tool_additions(req):
        if d["name"] not in names:
            names.add(d["name"])
            tools.append(d)
    return tools


def _tool_addition(block) -> dict | None:
    """#5952: a `tool_addition` block's tool definition ({"name", "description", "input_schema"}), or None when the
    block does not carry one in that shape.  Claude Code sends some tools (MCP servers that connect after the session
    started) this way, in a system message inside `messages`, instead of in the top-level `tools`:
    {"type": "tool_addition", "tool": {"type": "tool_definition", "definition": {"name": ..., "input_schema": ...}}}.
    The outer `tool` must say "tool_definition"; the name a non-empty string; a description, when there, a string; a
    schema, when there, an object."""
    tool = block.get("tool") if isinstance(block, dict) else None
    if not isinstance(tool, dict) or tool.get("type") != "tool_definition":
        return None
    d = tool.get("definition")
    if (isinstance(d, dict) and isinstance(d.get("name"), str) and d["name"]
            and isinstance(d.get("input_schema", {}), dict) and isinstance(d.get("description", ""), str)):
        return d
    return None


def _tool_additions(req: dict) -> list[dict]:
    """#5952: every tool definition a SYSTEM message of the conversation adds with a `tool_addition` block, in order.
    The block is the client's (Claude Code puts it in a system message); one in a user or assistant turn - text a
    user pasted, a tool's output echoed back - adds no tool."""
    out = []
    messages = req.get("messages")
    for m in messages if isinstance(messages, list) else []:
        if not isinstance(m, dict) or m.get("role") != "system":
            continue
        content = m.get("content")
        for block in content if isinstance(content, list) else []:
            if isinstance(block, dict) and block.get("type") == "tool_addition":
                d = _tool_addition(block)
                if d is not None:
                    out.append(d)
    return out


def _tool_references(content, deferred: dict):
    """A tool_result's tool_reference blocks (a tool search's results) -> the referenced deferred tools' definitions,
    in the form the prompt lists tools: the model sees a tool where it was loaded, and the prompt's start (its tool
    list) stays the same, so the prompt cache does too.  Any other reference is left out as before (_check_parts): a
    listed tool is in the prompt already, and an old one (its MCP server gone) must not fail the rest of the session."""
    if not isinstance(content, list):
        return content
    return [{"type": "text",
             "text": "<tools>\n" + json.dumps(deferred[p.get("tool_name")], ensure_ascii=False) + "\n</tools>"}
            if isinstance(p, dict) and p.get("type") == "tool_reference" and isinstance(p.get("tool_name"), str)
            and p["tool_name"] in deferred else p
            for p in content]


def anthropic_to_messages(req: dict, think_unasked: bool = True) -> tuple[list[dict], list[dict] | None, dict]:
    """Anthropic Messages -> (template messages, template tools, template kwargs).  `think_unasked`: a request
    without "thinking", an effort or a budget gets the template's default (it thinks), as through 0.1.31 (the
    config's "anthropic_thinking": "model"); False renders it without thinking, Anthropic's rule (#278, the server's
    default "on_request")."""
    declared = _tool_list(req.get("tools"), None)
    # #4663: a `defer_loading` tool stays out of the prompt's tool list until a tool_reference loads it
    deferred = {t["name"]: _anthropic_tool(t) for t in declared if t.get("defer_loading")}
    # #5952: the names the prompt already shows the model (its tool list); a tool_addition of one of them adds nothing.
    # An added name is shown with its one callable definition (_anthropic_callables: a deferred top-level tool's own,
    # else the first addition's), where the first addition of it stands.
    shown = {t["name"] for t in declared if t["name"] not in deferred}
    callable_defs: dict = {}
    for t in _anthropic_callables(req):
        callable_defs.setdefault(t["name"], t)
    messages = []
    system = req.get("system")
    if system:
        messages.append({"role": "system",
                         "content": pin_billing_stamp(_text_of(_check_parts(system, "system", TEXT_PARTS)))})
    for i, m in enumerate(_object_list(req.get("messages"), "messages")):
        content = _check_parts(_anthropic_documents(m.get("content"), f"messages[{i}].content"),
                               f"messages[{i}].content", ANTHROPIC_BLOCKS)
        if isinstance(content, str):
            messages.append({"role": m["role"], "content": content})
            continue
        if m.get("role") == "user" and _has_image(content) and not any(
                isinstance(b, dict) and b.get("type") == "tool_result" for b in content or []):
            messages.append({"role": "user", "content": _parts_of(content)})
            continue
        text, reasoning, calls, parts = [], [], [], []
        for j, block in enumerate(content or []):
            # a non-object block is skipped leniently, as _text_of / _check_parts do on OpenAI; it used to raise
            # AttributeError here (a generic 400).  (Through a name, so the type checker keeps `block` as it was.)
            is_object = isinstance(block, dict)
            if not is_object:
                continue
            kind = block.get("type")
            if kind == "text":
                text.append(block.get("text", ""))
                parts.append(block)
            elif kind in IMAGE_PARTS + FILE_PARTS:
                parts.append(block)
            elif kind == "thinking":
                said = block.get("thinking", "")
                if not isinstance(said, str):     # a non-string value crashed "".join(reasoning) below; name the
                    raise ValueError(f"messages[{i}].content[{j}].thinking must be a string")   # field (#6098)
                signed = _signed_thinking(block.get("signature"), f"messages[{i}].content[{j}]")
                if signed is not None and signed != said:
                    raise ValueError(f"messages[{i}].content[{j}]: the thinking does not match its signature")
                reasoning.append(said)
            elif kind == "redacted_thinking":     # one of this server's is read; another server's carries nothing
                signed = _signed_thinking(block.get("data"), f"messages[{i}].content[{j}]")
                if signed is not None:
                    reasoning.append(signed)
            elif kind == "tool_addition":
                # #5952: the added tool's definition where the client added it, in the form a tool_reference loads
                # one (_tool_references): the prompt's start (its tool list) stays the same, and so does its cache
                d = _tool_addition(block)
                if d is None or m.get("role") != "system":
                    LOGGER.debug("messages[%d].content[%d]: tool_addition %s; left out of the prompt", i, j,
                                 "without a tool definition" if d is None else f"in a {m.get('role')} message")
                elif d["name"] not in shown:
                    shown.add(d["name"])
                    added = ("<tools>\n" + json.dumps(_anthropic_tool(callable_defs[d["name"]]), ensure_ascii=False) +
                             "\n</tools>")
                    text.append(added)
                    parts.append({"type": "text", "text": added})
            elif kind == "tool_use":
                params = declared_parameters(req.get("tools"), block.get("name"))
                calls.append({"function": {"name": block.get("name"),
                                           "arguments": tool_arguments(block.get("input"), params)},
                              "_call_id": block.get("id")})
            elif kind == "tool_result":
                # A tool's image (Claude Code's Read of a picture) reaches the encoder like a user's, as the OpenAI
                # path's tool messages already do; a text-only result is one string, as before.  Its tool_use_id
                # puts it beside its call (_order_chat_tool_results): parallel results may come in any order.
                where = f"messages[{i}].content[{j}].content"
                result = _parts_of(_check_parts(
                    _anthropic_documents(_tool_references(block.get("content"), deferred), where), where,
                    TEXT_PARTS + IMAGE_PARTS + FILE_PARTS))
                if block.get("is_error") is True:   # the tool failed: the model is told so, not only its text
                    result = "Error: " + result if isinstance(result, str) else \
                        [{"type": "text", "text": "Error: "}] + list(result or [])
                messages.append({"role": "tool", "content": result, "_call_id": block.get("tool_use_id")})
        if text or calls or reasoning or parts:
            # an image sent beside tool results stays in this turn instead of being dropped
            out = {"role": m["role"], "content": _parts_of(parts) if _has_image(parts) else "".join(text)}
            if reasoning:
                out["reasoning_content"] = "".join(reasoning)
            if calls:
                out["tool_calls"] = calls
            messages.append(out)
    _order_chat_tool_results(messages)
    tools = [_anthropic_tool(t) for t in declared if t["name"] not in deferred] or None
    kwargs = {}
    # Anthropic: "thinking": {"type": "disabled"} or {"type": "enabled", "budget_tokens": N};
    # "output_config": {"effort": "low" | "medium" | "high"}
    thinking = req.get("thinking")
    if thinking is not None and (not isinstance(thinking, dict)
                                 or thinking.get("type") not in ("enabled", "disabled", "adaptive")):
        raise ValueError('thinking must be {"type": "enabled", "budget_tokens": N}, {"type": "adaptive"} or '
                         '{"type": "disabled"}')
    # "adaptive": the model decides how much to think (the template's default, an effort if output_config has one)
    effort = (req.get("output_config") or {}).get("effort") if isinstance(req.get("output_config"), dict) else None
    # 0.1.45: Anthropic's budget_tokens is a real budget, not only a level: the thinking is closed (wrap-up, </think>,
    # the answer) after that many tokens.  A reasoning_budget_tokens of the request's own wins.
    bt = thinking.get("budget_tokens") if isinstance(thinking, dict) and thinking.get("type") != "disabled" else None
    if isinstance(bt, int) and not isinstance(bt, bool) and bt > 0 and req.get("reasoning_budget_tokens") is None:
        req["reasoning_budget_tokens"] = bt
    if isinstance(thinking, dict) and thinking.get("type") == "disabled":
        kwargs["enable_thinking"] = False
    elif effort:
        kwargs.update(effort_kwargs(effort))
    elif isinstance(thinking, dict) and thinking.get("budget_tokens"):
        kwargs.update(budget_effort(thinking["budget_tokens"]))
    elif thinking is None and not req.get("reasoning_budget_tokens") and not think_unasked:
        # The server's default (the config's "anthropic_thinking": "on_request"; "model" thinks as 0.1.31 did, for
        # a client that never asks).  Anthropic's thinking is opt-in. Claude Code's helper calls (a session title,
        # a topic check) ask for none
        # and allow a few dozen tokens, which the model otherwise spent thinking and answered with no text at all.
        # A config's reasoning_effort still applies: Service.with_shared sets output_config before this runs.  A
        # request that gives its own reasoning_budget_tokens (#123) asks for thinking, so it thinks as before.
        kwargs["enable_thinking"] = False
    return _env_to_user(_late_system_to_user(messages)), tools, kwargs


def anthropic_format(req: dict):
    """Anthropic's structured output, `output_config.format` (or the beta's name for it, `output_format`):
    {"type": "json_schema", "schema": {...}} -> the Chat Completions response_format prepare_format reads, or None."""
    config = req.get("output_config")
    fmt = config.get("format") if isinstance(config, dict) else None
    if fmt is not None and req.get("output_format") is not None:
        raise ValueError("give the format once: output_config.format or output_format, not both")
    fmt = req.get("output_format") if fmt is None else fmt
    if fmt is None:
        return None
    if not isinstance(fmt, dict) or fmt.get("type") != "json_schema":
        raise ValueError('output_config.format must be {"type": "json_schema", "schema": {...}}')
    if not isinstance(fmt.get("schema"), dict):
        raise ValueError("output_config.format needs a schema object")
    # Anthropic's structured output always follows its schema: strict, as OpenAI's strict: true
    return {"type": "json_schema", "json_schema": {"name": "output", "strict": True, "schema": fmt.get("schema")}}


# ------------------------------------------------------------------------------------------------ output parser
@dataclass
class ToolCall:
    name: str
    arguments: dict
    id: str = field(default_factory=lambda: "call_" + uuid.uuid4().hex[:24])


@dataclass
class Event:
    kind: str                     # "reasoning" | "content" | "tool_call"
    text: str = ""
    call: ToolCall | None = None


THINK_END = "</think>"
CALL_START = "<tool_call>"
CALL_END = "</tool_call>"
FUNCTION_CALLS_START = "<function_calls>"     # #6022: Claude's dialect's wrapper around its `<invoke>` calls (recover)
FUNCTION_CALLS_END = "</function_calls>"


PARAM_END = "</parameter>"
FUNC_START = "<function="
FUNC_END = "</function>"
# #3441: the opener a model writes when it drifts by a token inside the envelope the template documents:
# `<=NAME>` (the word `function` dropped), `<function:NAME>`, `<function =NAME>`, or the bare `<NAME>`
_LOOSE_OPEN = re.compile(r"<\s*(?:function\s*[=:]\s*|=\s*)?([A-Za-z0-9_.\-]+)\s*>")
# A live Pi turn omitted only the opening angle bracket: `function=write>`.
# Recovery accepts it only immediately inside an explicit wrapper and only for
# a declared tool; the ordinary content path never scans for this spelling.
_MISSING_FUNC_OPEN = re.compile(r"function\s*=\s*([A-Za-z0-9_.\-]+)\s*>")
# how far before a new delta a prefixed name or a misspelt opener can start (see OutputParser._prefixed)
_PREFIX_LOOKBACK = 256
# #5409: `<function=tools actio>` / `<function=tools>`; the prefixes are the ones of #4751
_PREFIX_SPACED = re.compile(r"(<function=\s*(?:tools|functions|default_api)[ \t]+)([^<>\s]+)\s*>")
_PREFIX_PART = re.compile(r"<function=\s*(?:tools|functions|default_api)[ \t]+[^<>\s]{0,80}\s*$")
_PREFIX_BARE = re.compile(r"(<tool_call>\s*)<function=\s*(tools|functions|default_api)\s*>(.*?)</function>", re.S)
_PREFIX_BARE_OPEN = re.compile(r"<tool_call>\s*<function=\s*(tools|functions|default_api)\s*>(?!.*</function>)", re.S)
# #5276: `<toolcall>` / `<tool-call>` as the opener (then whitespace, `<function=NAME>`), and such an opener still
# arriving
_MISSPELT_OPEN = re.compile(r"<tool-?call>(\s*)<function=(\s*[^<>\s]+\s*)>")
_MISSPELT_PART = re.compile(r"<tool-?call>\s*<function=[^<>]{0,80}$"
                            r"|<tool-?call>\s*(?:<(?:f(?:u(?:n(?:c(?:t(?:i(?:o(?:n)?)?)?)?)?)?)?)?)?$")
# upstream #5969 (recover): Claude's `<invoke name="NAME"><parameter name="K">V</parameter></invoke>` dialect inside the
# wrapper
_INVOKE_OPEN = re.compile(r'<invoke\s+name="([^"<>\n]+)"\s*>')
_INVOKE_PART = re.compile(r'<(?:i(?:n(?:v(?:o(?:k(?:e(?:\s+(?:n(?:a(?:m(?:e(?:=(?:"[^"<>\n]*"?)?)?)?)?'
                          r')?)?)?)?)?)?)?)?)?\s*$')
_INVOKE_PARAM = re.compile(r'((?:<invoke\s+name="[^"<>\n]+"\s*>|</parameter>)\s*)<parameter\s+name="([^"<>\n]+)"\s*>')
# the call's end: `</invoke>` right after a value's `</parameter>` or the opener (a value may hold the words)
_INVOKE_END = re.compile(r'(?:</parameter>|<invoke\s+name="[^"<>\n]+"\s*>)\s*</invoke>')


def invoke_to_template(raw: str) -> str:
    """#5969: an `<invoke ...>` call body (up to its `</invoke>`) in the template's form: `<function=T>`,
    `<parameter=K>`, `</function>`; the tags are rewritten only where the call's structure has them."""
    m = _INVOKE_OPEN.match(raw.lstrip())
    body = raw.lstrip()
    body = _INVOKE_PARAM.sub(lambda p: p.group(1) + PARAM_START + p.group(2) + ">", body)
    body = _INVOKE_PARAM.sub(lambda p: p.group(1) + PARAM_START + p.group(2) + ">", body)   # (overlapping neighbours)
    assert m is not None                    # (the caller saw the opener)
    body = FUNC_START + m.group(1) + ">" + body[m.end():]
    return body[:len(body) - len("</invoke>")] + FUNC_END if body.endswith("</invoke>") else body


_LOOSE_PART = re.compile(r"<\s*(?:function[ \t]*[=:]?\s*|=\s*)?[A-Za-z0-9_.\-]*\s*$")


# a quoted call's opener in a value: `<function=NAME>` and its first parameter or close; a bare FUNC_START is text
_QUOTED_OPEN = re.compile(r"<function=[^<>\s]+>\s*(?:<parameter=|</function>)")


#: #4701: how much text after a closing sequence is looked at before the value is taken as ended there (a call followed
#: by a long sentence)
CLOSE_LOOKAHEAD_CHARS = 1500


def _close_verdict(after: str, final: bool) -> str:
    """`after` starts with `</function>` and/or `</tool_call>` right after a `</parameter>`.  A file that documents the
    call format has those three tags on lines of its own inside a value (#4701), so the sequence only ends the value
    when what follows is the end of the turn, another call, or prose with no closing tag after it.  "end": the value
    ends here; "continue": it goes on to the next `</parameter>`; "hold": not decidable yet (streaming): wait for more
    text.  Bounded: the end of the turn (`final`), a new `<tool_call>`, or CLOSE_LOOKAHEAD_CHARS of text decide, so a
    call is never held longer than that."""
    m = re.match(r"(?:</function>)?\s*</tool_call>", after)
    if not m:
        tail = after[len(FUNC_END):].lstrip() if after.startswith(FUNC_END) else after
        if not final and CALL_END.startswith(tail):
            return "hold"                                  # #5040: `</tool_call>` is still arriving: a streamed view
            #                                                 must not end the value yet
        if not after.startswith(FUNC_END) or tail.startswith((CALL_START, FUNC_START)):
            return "end"                                   # the call is cut, or the next call follows without this
            #                                                 one's `</tool_call>`
        if not final and (CALL_START.startswith(tail) or FUNC_START.startswith(tail)):
            return "hold"                                  # that next opener is still arriving
        # A file in a value (an XML file, a test of a parser) can have `</parameter>` and then `</function>` on the
        # next line, and then its own next element: the wrapped call goes on, so the sequence is part of the value when
        # closing tags follow later (#4701's rule)
        if PARAM_END in tail or FUNC_END in tail or CALL_END in tail:
            return "continue"
        if final or len(tail) > CLOSE_LOOKAHEAD_CHARS:
            return "end"                                   # the call is cut, or its `</tool_call>` was left out
            #                                                 before prose: as before
        return "hold"
    rest = after[m.end():].lstrip()
    if not rest:
        return "end" if final else "hold"                  # the usual call: the turn ends right after it
    if rest.startswith(CALL_START):
        return "end"
    if not final and CALL_START.startswith(rest):
        return "hold"                                      # `<tool_c` of a second call is arriving
    if CALL_START in rest:
        return "end"                                       # prose, then another call: this one is complete
    if PARAM_END in rest or CALL_END in rest or FUNC_END in rest:
        # prose, then closing tags again: the text was part of the value
        return "continue"
    if final or len(rest) > CLOSE_LOOKAHEAD_CHARS:
        return "end"
    return "hold"


def param_end(text: str, final: bool = False, name: str | None = None, hold: bool = True,
              allow_reopen: bool = False, names=None, closer: str | None = None) -> int:
    """Where a parameter value in `text` ends: the first `</parameter>` followed (after whitespace) by the next
    `<parameter=` or `</function>`, or by the wrapper's `</tool_call>` when the call's `</function>` was left out
    (#4427) - the same text inside a value (a file that documents the call format, #210) is part of the value.
    #4078: a close inside a call the value itself quotes (more _QUOTED_OPEN than `</function>` before it), or one
    followed by the same parameter `name` again, is part of the value too - with `allow_reopen` only: a string value or
    one with no declared type (upstream #5493: a non-string parameter written twice is two writes, the later counts).
    upstream #6032 R2: `names` (recovery on): a damaged opener (opener_at) of one of these declared parameters not set
    yet ends the value as well.  upstream #6032 R4: `closer` (`</NAME>` of a value opened as the element `<NAME>`) ends
    it too, under the same follower rule; close_len() says which closer stands at the answer.  `text` starts at the
    value.  -1: none yet; -2: a candidate whose follower has not arrived (streaming; `final` accepts it)."""
    def nxt(start):
        a = text.find(PARAM_END, start)
        b = text.find(closer, start) if closer else -1
        return a if b < 0 or 0 <= a < b else b
    at = nxt(0)
    while at >= 0:
        if len(_QUOTED_OPEN.findall(text, 0, at)) <= text.count(FUNC_END, 0, at):
            if hold and text.startswith(PARAM_END, at) and _nested_close(text, at):
                # #5756: this `</parameter>` closes a `<parameter=X>` the value itself holds (a file that documents the
                # format) on the same line, so it is value text.  The model often skips the value's own closer after
                # such a pair: then the call's end (or the next declared parameter) follows the nested closer, and the
                # value ends AFTER it (close_len is 0 there).
                skip_to = at + len(PARAM_END)
                nafter = text[skip_to:].lstrip()
                if nafter.startswith((FUNC_END, CALL_END)):
                    verdict = _close_verdict(nafter, final)
                    if verdict == "end":
                        return skip_to
                    if verdict == "hold":
                        return -2
                elif nafter.startswith(PARAM_START):
                    gt = nafter.find(">")
                    if gt < 0:
                        return skip_to if final else -2
                    if not names or param_name(nafter[len(PARAM_START):gt]) in names:
                        return skip_to
                elif not final and (not nafter or FUNC_END.startswith(nafter) or CALL_END.startswith(nafter)
                                    or PARAM_START.startswith(nafter)):
                    return -2                                   # what follows has not arrived
                at = nxt(at + 1)
                continue
            after = text[at + close_len(text, at, closer):].lstrip()
            if after.startswith(PARAM_START):
                gt = after.find(">")
                if gt < 0:
                    return at if final else -2
                if not allow_reopen or name is None or after[len(PARAM_START):gt].strip() != name:
                    return at
            elif after.startswith((FUNC_END, CALL_END)):          # #4427: the model left out its `</function>`
                # hold=False: a call inside the reasoning keeps its old rule
                verdict = _close_verdict(after, final) if hold else "end"
                if verdict == "end":
                    return at
                if verdict == "hold":
                    return -2
            elif after.startswith(FOLLOWERS) or _follower_form(after):    # upstream #6032 R1: the damaged followers
                return at
            elif names and opener_at(after, names, final) not in (None, "pending"):
                # upstream #6032 R2: `<NAME>` / `</NAME>` of a declared parameter not set
                return at
            elif names is not None and _stub_follows(after):
                # upstream #6032 R5 (recover): the model's empty stub, or a repeated closer
                return at
            elif not after or any(tag.startswith(after) for tag in (PARAM_START, FUNC_END, CALL_END) + FOLLOWERS) \
                    or _follower_form(after) is None or (names and opener_at(after, names, final) == "pending") \
                    or (names is not None and _stub_follows(after) is None):
                return at if final else -2
        at = nxt(at + 1)
    return -1


def _nested_close(text: str, at: int) -> bool:
    """#5756: the `</parameter>` at `at` closes a `<parameter=` opener inside the value that is still open, on the same
    line (`<parameter=limit>25</parameter>`).  The value's own closer comes after a line break, so a closer on the
    opener's own line is the nested pair's."""
    opens = [m.start() for m in re.finditer(re.escape(PARAM_START), text[:at])]
    if len(opens) <= text.count(PARAM_END, 0, at):
        return False
    return "\n" not in text[opens[-1]:at]


def close_len(text: str, at: int, closer: str | None = None) -> int:
    """#6032 R4: the length of the value closer param_end found at `at`: `</parameter>`, or the element's `</NAME>`.
    0 where no closer stands (#5756: the value ended after a nested pair whose own closer the model left out)."""
    if text.startswith(PARAM_END, at):
        return len(PARAM_END)
    if closer and text.startswith(closer, at):
        return len(closer)
    return 0


# #6032 R1: what else ends a value after its `</parameter>` (Gyro / Coder-L recordings): the Anthropic XML form's
# `</invoke>` and `<parameter name=` written in place of the call format's own tags, and a garbled `</<`
FOLLOWERS = ("</<", "</invoke>", "<parameter name=")
_DAMAGED_OPENER = re.compile(r"<=[^<>\s]+>")          # `<=NAME>`: a damaged parameter opener
_STRAY_CLOSER = re.compile(r"</(?!parameter>|function>|tool_call>)[^<>\s=/]+>\s*")


def _follower_form(after: str):
    """#6032 R1: `after` (what follows a `</parameter>`, no leading whitespace) starts with a damaged opener `<=NAME>`,
    or with a stray closer `</NAME>` standing where `</function>` belongs (then `</tool_call>`): True; None while it
    may still become one (streaming); False otherwise (a tag of the value itself: the value goes on)."""
    if _DAMAGED_OPENER.match(after):
        return True
    if re.fullmatch(r"<=[^<>\s]*", after):
        return None
    m = _STRAY_CLOSER.match(after)
    if m:
        rest = after[m.end():]
        if rest.startswith(CALL_END):
            return True
        return None if not rest or CALL_END.startswith(rest) else False
    return None if re.fullmatch(r"</(?!parameter>|function>|tool_call>)[^<>\s=/]*", after) else False


# #6032 R5: the model's empty stub after a value (Gyro / Coder-L recordings): a lone `<` or `</` line, a line of one tag
# (`</head_after>`), either closed by a `</parameter>` or followed by the next opener; and a closer written again
_STUB = re.compile(r"(?:</?[ \t]*\n|<[/=]?[^<>\s=/]+>[ \t]*\n)\s*")
_STUB_NEXT = (PARAM_END, "<parameter=", "</function>", "</tool_call>") + ("</<", "</invoke>", "<parameter name=")


def _stub_follows(after: str):
    """#6032 R5: `after` (what follows a value's `</parameter>`, no leading whitespace) is a closer written again, or
    an empty stub line followed by a closer or an opener: True; None while it may still become one; False otherwise."""
    if after.startswith(PARAM_END):
        return True
    m = _STUB.match(after)
    if m:
        rest = after[m.end():]
        if rest.startswith(_STUB_NEXT):
            return True
        return None if not rest or any(t.startswith(rest) for t in _STUB_NEXT) else False
    return None if re.fullmatch(r"</?[ \t]*|<[/=]?[^<>\s=/]*>?[ \t]*", after) else False


NOT_PARAMS = ("function", "think", "parameter")


def param_name(opener: str) -> str | None:
    """#6032 R1: the parameter a `<parameter=` opener names (`opener`: what follows `=` up to the `>` found): the name
    cut at its first whitespace, `<` or `>` (the model ran a closer or `</think>` into it, or put a space before the
    `>`); None - the value is read and dropped - when that leaves nothing, a placeholder (null / none) or one of the
    call format's own tag names."""
    name = re.split(r"[\s<>]", opener.lstrip(), 1)[0]
    if not name or name in NOT_PARAMS or name.lower() in ("null", "none"):
        return None
    return name


ARGS_START = "<arguments>"


def json_args(s: str, props=None, notes: list | None = None, root=None):
    """#6032 R3 (recover): `s` starts with `<arguments>` + a JSON object (Gyro / Coder-L wrote the call's arguments so,
    instead of `<parameter=` tags), closed by `</arguments>`, by the stray `</parameter>` the model wrote, or by
    nothing.  -> (arguments, length read) with each string value of a declared non-string parameter read by its schema
    (param_value; `root` resolves its references); None while the object is not whole (a call cut inside it is never
    run with partial strings)."""
    t = s[len(ARGS_START):]
    body = t.lstrip()
    try:
        obj, end = json.JSONDecoder().raw_decode(body)
    except ValueError:
        return None
    if not isinstance(obj, dict):
        return None
    tail = re.match(r"\s*(?:</arguments>|" + re.escape(PARAM_END) + r")?", body[end:])
    args = {}
    for k, v in obj.items():
        schema = (props or {}).get(k)
        types = _schema_hints(schema, root)[0]
        if isinstance(v, str) and types is not None and "string" not in types:
            v = param_value(v, schema, root, notes)
            if v is _SKIP:
                continue
        args[k] = v
    assert tail is not None                 # (every part of the pattern is optional)
    return args, len(ARGS_START) + len(t) - len(body) + end + tail.end()


_COLON_OPENER = re.compile(r"<parameter=([^<>\s:=]+):[ \t]*([^\n<>]*)\n")
_BARE_OPENER = re.compile(r"<(?:/|=)?([^<>\s=/]+)>")
# #5969: the Anthropic XML form's parameter opener written inside the call format's `<function=NAME>` (a declared
# parameter)
_NAMED_OPENER = re.compile(r'<parameter\s+name\s*=\s*"?([^"<>\s]+)"?\s*>')


def opener_at(s: str, names=None, final: bool = False):
    """#6032 R2: the parameter opener `s` starts with (no leading whitespace), at a point where one may begin (after
    the function opener or a value's `</parameter>`).  -> (name, length, inline value): name None = read and dropped;
    inline value: the colon form's own value, the rest None.  "pending": it may still become one (streaming; `final`
    decides); None: no opener here.
      `<parameter=NAME>`: the call format's own (the name as param_name reads it);
      `<parameter=NAME: value` + line break (no `>` on the line): NAME with that value, quotes around it dropped,
          then its `</parameter>` (Gyro / Coder-L, recorded);
      `<NAME>`, `</NAME>`, `<=NAME>`: an opener of NAME when NAME is in `names` (the call's declared parameters not
          set yet); any other such tag is no opener (a tag of the text, or one R1 drops)."""
    if s.startswith(PARAM_START):
        line = s.split("\n", 1)[0]
        gt = line.find(">")
        if gt >= 0:
            return param_name(s[len(PARAM_START):gt]), gt + 1, None
        m = _COLON_OPENER.match(s)
        if m:
            value = m.group(2).strip()
            if value.startswith('"'):
                value = value[1:]
                value = value[:-1] if value.endswith('"') else value
            tail = re.match(r"\s*" + re.escape(PARAM_END), s[m.end():])
            if tail is None and not final and PARAM_END.startswith(s[m.end():].lstrip()):
                return "pending"
            return param_name(m.group(1)), m.end() + (tail.end() if tail else 0), value
        if "\n" not in s:
            # the `>` still arriving, or a long name: held, except that the call's own scan reads it as before when
            # the line ends (a `<parameter=` line with neither `>` nor a colon form is no opener)
            if s.find(">") < 0 and not final:
                return "pending"
            gt = s.find(">")
            return (param_name(s[len(PARAM_START):gt]), gt + 1, None) if gt >= 0 else None
        return None, len(line), None        # #6032 R5: an opener cut before its `>` (its line ends first): dropped
    if names:
        m = _NAMED_OPENER.match(s)
        if m:                                # #5969: `<parameter name="NAME">` inside a `<function=` call
            return (m.group(1), m.end(), None) if m.group(1) in names else None
        if not final and any(f'<parameter name="{n}">'.startswith(s) or f"<parameter name={n}>".startswith(s)
                             for n in names):
            return "pending"
        m = _BARE_OPENER.match(s)
        if m:
            return (m.group(1), m.end(), None) if m.group(1) in names else None
        if not final and re.fullmatch(r"<[/=]?[^<>\s=/]*", s) and any(
                n.startswith(s.lstrip("</=")) for n in names):
            return "pending"
    return None


def call_end(text: str, final: bool = False, hold: bool = True, names=None) -> int:
    """Where a tool call's body ends (#210): the `</tool_call>` after the call's own `</function>`, found by walking
    its parameters with param_end, so a value may contain either tag.  -1: not complete yet.  A body that is not in
    the call format ends at the first `</tool_call>`, as before.  #6032 R4 (recover): `names` - the declared
    parameters of the call's tool - lets the walk step over damaged openers and elements as parse_tool_call reads
    them."""
    seen: set = set()
    s = text.lstrip()
    pos = len(text) - len(s)
    if not s.startswith("<function="):
        return text.find(CALL_END) if not "<function=".startswith(s) else -1
    gt = text.find(">", pos)
    if gt < 0:
        return -1
    pos = gt + 1
    while True:
        rest = text[pos:]
        s = rest.lstrip()
        pos += len(rest) - len(s)
        # #6032 R2-R5 (recover, the tool's names known): every opener as parse_tool_call reads it - a damaged one, an
        # element, the colon form, one cut before its `>` - so the walk never runs one call's end into the next call
        got = opener_at(s, set(names) - seen, final) if names and not s.startswith(FUNC_END) else None
        if got == "pending":
            return -1
        if got is not None and got[2] is not None:            # the colon form: its value and closer are read
            seen.add(got[0])
            pos += got[1]
        elif s.startswith("<parameter=") and got is None:
            gt = text.find(">", pos)
            if gt < 0:
                return -1
            pname = param_name(text[pos:gt].split("=", 1)[1])
            end = param_end(text[gt + 1:], final, pname, hold, allow_reopen=True,
                            names=(set(names) - seen - {pname}) if names else None)
            if end < 0:
                return -1
            seen.add(pname)
            pos = gt + 1 + end + close_len(text[gt + 1:], end)
        elif got is not None:                                 # #6032 R2 / R4 / R5: any opener, an element too
            pname, length, _ = got
            closer = "</%s>" % pname if pname and s.startswith("<%s>" % pname) else None
            end = param_end(text[pos + length:], final, pname, hold, allow_reopen=True,
                            names=set(names) - seen - {pname}, closer=closer)
            if end < 0:
                return -1
            seen.add(pname)
            pos = pos + length + end + close_len(text[pos + length:], end, closer)
        elif s.startswith(FUNC_END):
            rest = text[pos + len(FUNC_END):]
            s = rest.lstrip()
            if s.startswith(CALL_END):
                return pos + len(FUNC_END) + len(rest) - len(s)
            return -1 if CALL_END.startswith(s) else text.find(CALL_END, pos)
        elif not s or "<parameter=".startswith(s) or FUNC_END.startswith(s):
            return -1
        else:
            return text.find(CALL_END, pos)


GLM_KEY, GLM_KEY_END, GLM_VALUE, GLM_VALUE_END = "<arg_key>", "</arg_key>", "<arg_value>", "</arg_value>"


def glm_call_body(body: str) -> str:
    """A GLM call's body (`NAME<arg_key>K</arg_key><arg_value>V</arg_value>...`, between <tool_call> and
    </tool_call>) in the XML form parse_tool_call reads (`<function=NAME>` with `<parameter=K>` values).  The values
    stay as written: a string as it is, other types as JSON, which parse_tool_call decodes by the tool's schema."""
    k = body.find(GLM_KEY)
    out = [f"<function={(body if k < 0 else body[:k]).strip()}>\n"]
    for key, value in re.findall(re.escape(GLM_KEY) + r"(.*?)" + re.escape(GLM_KEY_END) + r"\s*" + re.escape(GLM_VALUE)
                                 + r"(.*?)" + re.escape(GLM_VALUE_END), body, re.S):
        out.append(f"<parameter={key.strip()}>\n{value}\n{PARAM_END}\n")
    return "".join(out) + FUNC_END


class GlmCallRewriter:
    """GLM's calls, as they stream, in the XML form OutputParser reads: the text outside `<tool_call>...</tool_call>`
    passes as it comes (a partial `<tool_call>` held back), a call is held until its `</tool_call>` and then written
    out rewritten (glm_call_body).  A `<tool_call>` that a `</think>` follows before its end, or that runs past
    RCALL_MAX, was not a call: it goes on as it was written."""

    def __init__(self):
        self.buf, self.inside = "", False

    def feed(self, delta: str) -> str:
        self.buf += delta
        out = []
        while True:
            if not self.inside:
                i = self.buf.find(CALL_START)
                if i < 0:
                    keep = next((n for n in range(min(len(CALL_START) - 1, len(self.buf)), 0, -1)
                                 if CALL_START.startswith(self.buf[-n:])), 0)
                    out.append(self.buf[:len(self.buf) - keep])
                    self.buf = self.buf[len(self.buf) - keep:]
                    return "".join(out)
                out.append(self.buf[:i])
                self.buf, self.inside = self.buf[i + len(CALL_START):], True
            end, think = self.buf.find(CALL_END), self.buf.find(THINK_END)
            if end >= 0 and (think < 0 or think > end):
                out.append(CALL_START + "\n" + glm_call_body(self.buf[:end]) + "\n" + CALL_END)
                self.buf, self.inside = self.buf[end + len(CALL_END):], False
            elif think >= 0 or len(self.buf) > RCALL_MAX:
                out.append(CALL_START)
                self.inside = False
            else:
                return "".join(out)

    def flush(self) -> str:
        """The end of the output: what is held, as it was written."""
        text = (CALL_START if self.inside else "") + self.buf
        self.buf, self.inside = "", False
        return text


def tool_choice_of(tool_choice) -> tuple[str, str | None]:
    """A client's `tool_choice` in either API's shape -> ("auto" | "none" | "required" | "named", name).  OpenAI:
    "auto" / "none" / "required", {"type": "function", "function": {"name": N}} or the flat {"type": "function",
    "name": N}; Anthropic: {"type": "auto" | "any" | "none"} and {"type": "tool", "name": N}.  A value it does not
    know is ("unknown", None)."""
    if tool_choice is None:
        return "auto", None
    if isinstance(tool_choice, str):
        return (tool_choice, None) if tool_choice in ("auto", "none", "required") else ("unknown", None)
    if isinstance(tool_choice, dict):
        kind = tool_choice.get("type")
        if kind in ("auto", "none"):
            return kind, None
        if kind in ("any", "required"):
            return "required", None
        if kind in ("function", "tool", "custom"):
            fn = tool_choice.get("function") or tool_choice.get("custom")
            name = (fn.get("name") if isinstance(fn, dict) else None) or tool_choice.get("name")
            if isinstance(name, str) and name:
                return "named", name
    return "unknown", None


def allowed_tools(tool_choice, tools: list[dict] | None, more: dict | None = None):
    """OpenAI's {"type": "allowed_tools"} tool_choice (Chat: {"allowed_tools": {"mode", "tools"}}; Responses: "mode"
    and "tools" beside the type) -> (the tools the model is offered: those it lists, the tool_choice to apply:
    "auto" or "required").  Any other tool_choice comes back as it is, with all the tools.  `more`: a listed tool
    type's template names (Responses' {"type": "image_generation"}: the image tools).  A list naming a tool the
    request does not have, or an unknown mode, is a ValueError (a 400)."""
    if not (isinstance(tool_choice, dict) and tool_choice.get("type") == "allowed_tools"):
        return tools, tool_choice
    spec = tool_choice["allowed_tools"] if isinstance(tool_choice.get("allowed_tools"), dict) else tool_choice
    mode, listed = spec.get("mode", "auto"), spec.get("tools")
    if mode not in ("auto", "required"):
        raise ValueError('tool_choice.allowed_tools: mode must be "auto" or "required"')
    if not isinstance(listed, list) or not listed:
        raise ValueError("tool_choice.allowed_tools: tools must be a non-empty list of the request's tools")
    names: set = set()
    for t in listed:
        if not isinstance(t, dict):
            raise ValueError("tool_choice.allowed_tools: each tool is an object")
        if t.get("type") in (more or {}):
            names |= set((more or {})[t["type"]])
            continue
        fn = t.get("function") or t.get("custom") or t
        name = fn.get("name") if isinstance(fn, dict) else None
        if not isinstance(name, str) or not name:
            raise ValueError("tool_choice.allowed_tools: each tool names a tool of the request")
        names.add(f"{t['namespace']}.{name}" if t.get("namespace") else name)
    have = {t.get("name") for t in tools or [] if isinstance(t, dict)}
    unknown = sorted(names - have)
    if unknown:
        raise ValueError(f"tool_choice.allowed_tools names tools the request does not have: {', '.join(unknown)}")
    return [t for t in tools or [] if t.get("name") in names], mode


def legacy_functions(req: dict) -> dict | None:
    """OpenAI's deprecated `functions` and `function_call` (before tools) -> the request with `tools` and `tool_choice`
    in their place and one call at a time (the old shape answers with one), or None when it asks for neither (a null
    is no field, and function_call "none" without functions is the default).  Mixed with tools or tool_choice, or a
    function_call "auto" or name without functions, it is a 400."""
    fns, choice = req.get("functions"), req.get("function_call")
    if fns is None and choice is None:
        return None
    if req.get("tools") is not None or req.get("tool_choice") is not None:
        raise ValueError("functions and function_call cannot be used with tools or tool_choice")
    if fns is None and choice == "none":
        return None
    if not isinstance(fns, list) or not fns or not all(isinstance(f, dict) for f in fns):
        raise ValueError("functions must be a non-empty array of function objects")
    if isinstance(choice, dict) and isinstance(choice.get("name"), str):
        choice = {"type": "function", "function": {"name": choice["name"]}}
    elif choice not in (None, "none", "auto"):
        raise ValueError('function_call must be "none", "auto" or {"name": ...}')
    out = {k: v for k, v in req.items() if k not in ("functions", "function_call")}
    out.update(tools=[{"type": "function", "function": f} for f in fns], parallel_tool_calls=False)
    if choice is not None:
        out["tool_choice"] = choice
    return out


def legacy_function_call(choice: dict) -> dict:
    """A reply's choice (a message or a stream's delta) for a `functions` request: its call as the old function_call
    and finish_reason "function_call", as OpenAI answers one."""
    for key in ("message", "delta"):
        m = choice.get(key)
        if not isinstance(m, dict):
            continue
        calls = m.pop("tool_calls", None)
        if calls:
            fn = calls[0].get("function") or {}
            m["function_call"] = {k: fn[k] for k in ("name", "arguments") if k in fn}
    if choice.get("finish_reason") == "tool_calls":
        choice["finish_reason"] = "function_call"
    return choice


def forced_call(tool_choice, tools: list[dict] | None, form: str = "qwen",
                callable_tools: list[dict] | None = None) -> str | None:
    """`tool_choice` -> the text that opens the call the reply must make, or None (the model decides).
    There is no grammar here: the server writes this opening itself, so the model can only go on with a call.
    "required" / Anthropic "any": any of the tools; a named function: that one.  "none" is handled by the caller
    (no tools).  A value it cannot honour (an unknown shape, a name that is not one of the tools, "required" with no
    tools) is a ValueError - a 400, as OpenAI and Anthropic answer it - not a request that runs as "auto".
    `callable_tools` (Anthropic: the tools added in the conversation and the deferred ones as well, #5952): a name
    among them may be named too; "required" still picks from the prompt's own list."""
    kind, name = tool_choice_of(tool_choice)
    names = {t.get("name") for t in tools or [] if isinstance(t, dict)}
    if kind == "named" and name not in names and any(
            isinstance(t, dict) and t.get("name") == name for t in callable_tools or []):
        return CALL_START + str(name) if form == "glm" else CALL_START + f"\n<function={name}>\n"
    if kind in ("auto", "none"):
        return None
    if kind == "required" and len(names) == 1:     # one tool to call: name it, the model cannot invent another
        kind, name = "named", next(iter(names))
    # the model writes "<function=NAME>": from a written "<function=", Qwen3.8 went on with names no tool has
    if kind == "required" and names:
        return CALL_START if form == "glm" else CALL_START + "\n"
    if kind == "named" and name in names:
        return CALL_START + str(name) if form == "glm" else CALL_START + f"\n<function={name}>\n"
    if kind == "named":
        raise ValueError(f"tool_choice names {name!r}, which is not one of the request's tools")
    if kind == "required":
        raise ValueError("tool_choice asks for a tool call, but the request has no tools")
    raise ValueError(f"tool_choice {json.dumps(tool_choice)[:200]} is not a tool_choice this server takes")


PARAM_START = "<parameter="


def function_end(text: str) -> int:
    """Where a `<function=NAME>...</function>` at the start of `text` ends (just past `</function>`), walking its
    parameters with param_end like call_end does; -1: not complete (or not in the call format)."""
    s = text.lstrip()
    pos = len(text) - len(s)
    if not s.startswith(FUNC_START):
        return -1
    gt = text.find(">", pos)
    if gt < 0:
        return -1
    pos = gt + 1
    while True:
        rest = text[pos:]
        s = rest.lstrip()
        pos += len(rest) - len(s)
        if s.startswith(PARAM_START):
            gt = text.find(">", pos)
            if gt < 0:
                return -1
            # a bare or unwrapped call: the old rule
            end = param_end(text[gt + 1:], name=param_name(text[pos:gt].split("=", 1)[1]), hold=False,
                            allow_reopen=True)
            if end < 0:
                return -1
            pos = gt + 1 + end + len(PARAM_END)
        elif s.startswith(FUNC_END):
            return pos + len(FUNC_END)
        else:
            return -1


_JSON_FENCE = re.compile(r"```[ \t]*(?:json)?[ \t]*\r?\n\s*", re.I)


def json_tool_call(body: str, schemas: dict) -> ToolCall | None:
    """`{"name": N, "arguments": {...}}` (or "parameters", or the arguments as a JSON string) of a declared tool -> its
    call; anything else -> None.  #2037: the object may sit in a Markdown fence (```json ... ```)."""
    body = body.strip()
    m = _JSON_FENCE.match(body)
    if m:
        body = body[m.end():].rstrip()
        body = body[:-3] if body.endswith("```") else body
    try:
        obj = json.loads(body)
    except ValueError:
        return None
    if not isinstance(obj, dict) or not isinstance(obj.get("name"), str) or obj["name"] not in schemas:
        return None
    args = obj.get("arguments", obj.get("parameters", {}))
    if isinstance(args, str):
        try:
            args = json.loads(args)
        except ValueError:
            return None
    if args is None:
        args = {}
    return ToolCall(name=obj["name"], arguments=args) if isinstance(args, dict) else None


LENIENT_MAX = 1 << 20           # the longest value lenient_json will try (it is a python loop)


class _Lenient:
    """A JSON reader for the slips a model makes inside an array or object argument (#4427): a trailing or doubled
    comma, single quotes and Python's True/False/None, a bare string between two members of an object (`"status":
    "pending", "in_progress", "pending": false`), brackets left open at the end, extra closers at the end.  It is only
    asked about text json.loads has refused.  It never guesses at a value: a string cut short, an unknown word, a
    missing colon after a key, anything else raises ValueError."""

    _WORDS = {"true": True, "false": False, "null": None, "True": True, "False": False, "None": None}
    _ESC = {'"': '"', "'": "'", "\\": "\\", "/": "/", "b": "\b", "f": "\f", "n": "\n", "r": "\r", "t": "\t"}
    _TOKEN = re.compile(r"[^\s,\]}:]+")

    def __init__(self, text: str):
        self.s, self.i = text, 0
        # cut_arguments: a string the text ends inside is returned as far as it goes (and `cut` says so)
        self.cut_ok = False
        self.cut = False

    def peek(self) -> str:
        s, n = self.s, len(self.s)
        while self.i < n and s[self.i] in " \t\r\n":
            self.i += 1
        return s[self.i:self.i + 1]

    def string(self) -> str:
        s, q = self.s, self.s[self.i]
        self.i += 1
        out: list[str] = []
        while self.i < len(s):
            c = s[self.i]
            if c == q:
                self.i += 1
                return "".join(out)
            if c == "\\":
                self.i += 1
                e = s[self.i:self.i + 1]
                half_u = e == "u" and len(s) - self.i - 1 < 4 and re.fullmatch(r"[0-9a-fA-F]*", s[self.i + 1:])
                if self.cut_ok and (not e or half_u):
                    self.i = len(s)                 # cut inside an escape: the half escape is not text
                    self.cut = True
                    return "".join(out)
                if e == "u" and re.fullmatch(r"[0-9a-fA-F]{4}", s[self.i + 1:self.i + 5]):
                    out.append(chr(int(s[self.i + 1:self.i + 5], 16)))
                    self.i += 5
                    continue
                if e not in self._ESC:
                    raise ValueError("bad escape")
                out.append(self._ESC[e])
            else:
                out.append(c)
            self.i += 1
        if self.cut_ok:
            self.cut = True
            return "".join(out)
        raise ValueError("string not closed")

    def value(self, depth: int = 0):
        if depth > 64:
            raise ValueError("too deep")
        c = self.peek()
        if c == "{":
            return self.obj(depth + 1)
        if c == "[":
            return self.arr(depth + 1)
        if c in ('"', "'"):
            return self.string()
        m = self._TOKEN.match(self.s, self.i)
        if not m:
            raise ValueError("no value")
        self.i = m.end()
        if m.group() in self._WORDS:
            return self._WORDS[m.group()]
        return json.loads(m.group())            # a number, or ValueError

    def obj(self, depth: int):
        self.i += 1
        out: dict = {}
        while True:
            c = self.peek()
            if not c or c == "}":               # (no closer: the brackets were left open at the end)
                self.i += 1
                return out
            if c == ",":
                self.i += 1
                continue
            if c not in ('"', "'"):
                raise ValueError("object member")
            key = self.string()
            if self.peek() == ":":
                self.i += 1
                out[key] = self.value(depth)
            elif self.peek() not in (",", "}", ""):     # a bare string where a member belongs is dropped
                raise ValueError("object member")

    def arr(self, depth: int):
        self.i += 1
        out: list = []
        while True:
            c = self.peek()
            if not c or c == "]":
                self.i += 1
                return out
            if c == ",":
                self.i += 1
                continue
            out.append(self.value(depth))
            if self.peek() not in (",", "]", ""):
                raise ValueError("array element")


def lenient_json(text: str):
    """The JSON array or object `text` was meant to be, read past the slips of _Lenient; ValueError for anything else
    (and for text that is not an array or object).  Valid JSON is not its business: callers try json.loads first."""
    t = text.strip()
    m = _JSON_FENCE.match(t)
    if m:
        t = t[m.end():].rstrip()
        t = t[:-3].rstrip() if t.endswith("```") else t
    if not t.startswith(("[", "{")) or len(t) > LENIENT_MAX:
        raise ValueError("not an array or object")
    r = _Lenient(t)
    v = r.value()
    if t[r.i:].strip("]}) \t\r\n"):
        raise ValueError("text after the value")
    return v


def _json_type(value) -> str:
    # JSON Schema treats 1.0 as an integer too; Python's representation must not turn a valid enum number into text.
    return ("null" if value is None else "boolean" if isinstance(value, bool)
            else "integer" if isinstance(value, int) or isinstance(value, float) and value.is_integer()
            else "number" if isinstance(value, float) else "string" if isinstance(value, str)
            else "array" if isinstance(value, list) else "object")


def _schema_hints(schema, root=None, depth: int = 0, refs: tuple = ()) -> tuple:
    """Upper bounds on JSON types and on string enum/const values, not validation. None means unconstrained or
    unknown; an empty set means impossible. Siblings (including $ref) intersect; anyOf/oneOf branches union.
    Only same-resource JSON pointers are followed, from the supplied parameters root, never from a property.
    Anchors, remote refs, nested resource IDs and cycles are unknown; strict validation remains the final check."""
    if schema is False:
        return set(), set()
    if not isinstance(schema, dict) or depth > 8:
        return None, None
    if "$id" in schema and schema is not root:
        root = None                         # a nested resource changes the base of its references

    def intersect(a, b):
        return b if a is None else a if b is None else a & b

    def merge(a, b):
        return intersect(a[0], b[0]), intersect(a[1], b[1])

    def types_of(values):
        kinds = set(values)
        if "number" in kinds:
            kinds.add("integer")            # JSON Schema's number includes integer
        return kinds

    t = schema.get("type")
    types = types_of([t]) if isinstance(t, str) else types_of(t) if isinstance(t, list) and \
        all(isinstance(x, str) for x in t) else None
    hint = types, None
    for key in ("enum", "const"):
        values = schema.get(key) if key == "enum" else [schema[key]] if key in schema else None
        if isinstance(values, list):
            hint = merge(hint, (types_of(_json_type(v) for v in values),
                                {v for v in values if isinstance(v, str)}))
    ref = schema.get("$ref")
    if isinstance(ref, str) and root is not None and ref not in refs and (ref == "#" or ref.startswith("#/")):
        target = root
        for part in unquote(ref[2:]).split("/") if ref != "#" else ():
            part = part.replace("~1", "/").replace("~0", "~")
            if isinstance(target, dict) and (target is root or not isinstance(target.get("$id"), str)):
                target = target.get(part)
            elif isinstance(target, list) and len(part) <= len(str(len(target))) and part.isdecimal() and \
                    str(int(part)) == part and int(part) < len(target):
                target = target[int(part)]
            else:
                target = None
                break
        hint = merge(hint, _schema_hints(target, root, depth + 1, refs + (ref,)))
    for key in ("allOf", "anyOf", "oneOf"):
        branches = schema.get(key)
        if not isinstance(branches, list):
            continue
        combined: tuple = (None, None) if key == "allOf" else (set(), set())
        for branch in branches:
            more = _schema_hints(branch, root, depth + 1, refs)
            if key == "allOf":
                combined = merge(combined, more)
            else:
                combined = tuple(None if a is None or b is None else a | b for a, b in zip(combined, more))
        hint = merge(hint, combined)
    return hint


def cut_arguments(text: str):
    """The members a call's JSON-object arguments got before the output ended inside them (a client replays what it was
    sent, and max_tokens cuts a call short), as a dict: every complete `"key": value` pair, and a string value the text
    ends inside as far as it goes.  A value that is cut short and is not a string (a number, an array, an object) and a
    cut key are left out.  {} when nothing is recoverable (`{`, `{"pa`); None when the text is not an object cut short
    (it does not start with `{` and then a key or the end: `{cut`, prose, a list).
    It reads with _Lenient, so the slips that class forgives (single quotes, a trailing comma) are forgiven here too."""
    t = text.strip()
    if not t.startswith("{") or len(t) > LENIENT_MAX:
        return None
    r = _Lenient(t)
    r.cut_ok = True
    r.i = 1
    first = r.peek()
    if first and first not in ('"', "'"):
        # `{cut`, `{malformed_json`: not a JSON object cut short, kept as text (#510)
        return None
    out: dict = {}
    while True:
        c = r.peek()
        if not c or c == "}":
            break
        if c == ",":
            r.i += 1
            continue
        if c not in ('"', "'"):
            break
        try:
            key = r.string()
            if r.cut or r.peek() != ":":
                break
            r.i += 1
            value = r.value()
        except (ValueError, RecursionError):
            break
        if r.cut:
            if isinstance(value, str):
                out[key] = value
            break
        if r.i >= len(t) and not isinstance(value, str):
            break                               # a number or word the text ends in: it may be cut
        out[key] = value
    return out


def _schema_types(schema, depth: int = 0, *, root=None) -> set | None:
    """A conservative type bound; unknown branches cannot erase a sibling's explicit type constraint."""
    return _schema_hints(schema, root, depth)[0]


def _text_only(schema, root=None) -> bool:
    """Raw string text can stream immediately only when no enum/const may disambiguate JSON quoting."""
    types, strings = _schema_hints(schema, root)
    return types == {"string"} and strings is None


_SKIP = object()          # upstream #5493: param_value's "leave this parameter out"


def _reopens(schema, root=None) -> bool:
    """upstream #5493: whether a close followed by the same parameter again is part of the value (#4078): a string
    parameter's, or one with no declared type."""
    types = _schema_hints(schema, root)[0]
    return types is None or "string" in types


def param_value(text: str, schema, root=None, notes: list | None = None):
    """Restore parameter types, not validity. Plain string schemas keep raw text. An enum/const permits removing
    JSON string quotes only when the raw text is excluded and the decoded string is a candidate; ambiguity keeps
    the original. Nullable strings still decode null, but keep e.g. 1 as text when numbers are not allowed.
    #4427: where no string is allowed, an array or object json.loads refuses is read with lenient_json (a note says
    so in `notes`), and one that arrived as a JSON string of itself is decoded once more."""
    types, strings = _schema_hints(schema, root)
    if types == {"string"} and (strings is None or text in strings):
        return text
    # upstream #5979: the model's Python spelling (True / False) of a declared boolean is the boolean
    if types is not None and "boolean" in types and "string" not in types and text.strip().lower() in ("true", "false"):
        return text.strip().lower() == "true"
    # upstream #5493: a declared non-string parameter written empty or as a null placeholder (null / None) is null when
    # its type allows null, else _SKIP (left out, not sent as the text)
    bare = text.strip()
    if types is not None and "string" not in types and (not bare or bare.lower() in ("null", "none")):
        return None if "null" in types else _SKIP
    try:
        value = json.loads(text)
    except ValueError:
        if types is not None and "string" not in types:
            # upstream bee824a22 (#4754): Python-form numbers JSON refuses (+10, 010, .5) are the declared integer /
            # number
            # (a declared "number" reads as {"number", "integer"} in `types`, so the schema's own "type" says whether
            # an integer was asked for: +10 is 10 for "integer" and ["integer", "number"], 10.0 for "number")
            named = schema.get("type") if isinstance(schema, dict) else None
            whole = named == "integer" or (isinstance(named, list) and "integer" in named) or "number" not in types
            if "integer" in types and whole and re.fullmatch(r"[+-]?[0-9]+", bare):
                return int(bare)
            if "number" in types and re.fullmatch(r"[+-]?(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+)(?:[eE][+-]?[0-9]+)?", bare):
                n = float(bare)
                if math.isfinite(n):
                    return n
        if not text.lstrip().startswith(("[", "{", "`")) or (types is not None and "string" in types):
            return text
        try:
            value = lenient_json(text)
        except (ValueError, RecursionError):
            return text
        if notes is not None:
            notes.append("a damaged JSON " + ("array" if isinstance(value, list) else "object")
                         + " argument was read as written")
        return value
    kind = _json_type(value)
    if kind == "string" and types and types <= {"array", "object"} and value.lstrip().startswith(("[", "{")):
        return param_value(value, schema, root, notes)
    if kind == "string" and strings is not None and (text in strings or value not in strings):
        return text
    if types is not None and "string" in types and kind not in types:
        return text
    return value


def _properties(schema) -> dict:
    """A tool's parameters' properties, {} where the schema has none or is not an object (a boolean schema)."""
    params = schema.get("parameters") if isinstance(schema, dict) else None
    props = params.get("properties") if isinstance(params, dict) else None
    return props if isinstance(props, dict) else {}


def parse_tool_call(body: str, schema: dict | None = None, notes: list | None = None,
                    recover: bool = False) -> ToolCall:
    """`<function=NAME>\\n<parameter=P>\\nVALUE\\n</parameter>...</function>` -> ToolCall. Values are JSON-decoded
    when the tool's schema says the parameter is not a string (or, without a schema, when they parse as JSON
    objects/arrays/numbers/booleans)."""
    body = body.strip()
    if not body.startswith("<function=") or ">" not in body:
        raise ValueError("malformed tool call: " + body[:80])
    name = body[len("<function="):body.index(">")].strip()      # `<function= capture>`: the model's stray space
    rest = body[body.index(">") + 1:]
    props = _properties(schema)
    root = schema.get("parameters") if isinstance(schema, dict) else None
    args = {}
    if recover and rest.lstrip().startswith(ARGS_START):     # #6032 R3: the arguments as one JSON object
        got = json_args(rest.lstrip(), props, notes, root)
        if got is None:
            raise ValueError("malformed tool call: " + body[:80])
        args.update(got[0])
        rest = rest.lstrip()[got[1]:]
    elif recover and rest.lstrip().startswith("{"):
        # #6022 (recover): `<function=T>{"k": v}</function>` - a JSON object as the body is the call's arguments (it was
        # run with none)
        got = json_args(ARGS_START + rest.lstrip(), props, notes, root)
        if got is not None and re.fullmatch(r"\s*(?:</function>)?\s*", rest.lstrip()[got[1] - len(ARGS_START):]):
            args.update(got[0])
            rest = ""
    openers = dropped = 0
    while True:
        # upstream #6032 R2 (recover): where a parameter may begin, a damaged opener of a declared parameter not set yet
        # counts (opener_at); anything else is skipped to the next `<parameter=`, as before
        unset = {n for n in props if n not in args} if recover else None
        s = rest.lstrip()
        got = opener_at(s, unset, final=True) if (recover and not s.startswith(PARAM_START)) else None
        if got is None:
            if "<parameter=" not in rest:
                break
            rest = rest[rest.index("<parameter="):]
            if recover:
                got = opener_at(rest, unset, final=True)
            if got is None:                                  # the old reading (a `<parameter=` with no `>` raises)
                got = (param_name(rest[len(PARAM_START):rest.index(">")]), rest.index(">") + 1, None)
            s = rest
        pname, length, inline = got
        openers += 1
        dropped += pname is None
        # #6032 R4: an element
        closer = "</%s>" % pname if recover and pname and s.startswith("<%s>" % pname) else None
        rest = s[length:]
        pschema = props.get(pname) if pname else None
        if inline is not None:                               # the colon form carries its value on the opener's line
            value = inline
        else:
            # upstream #5493: a close followed by the same name again is part of a string value (#4078); a non-string
            # parameter written twice is two writes of it, and the later one counts
            end = param_end(rest, final=True, name=pname, allow_reopen=_reopens(pschema, root),
                            names={n for n in props if n not in args and n != pname} if recover else None,
                            closer=closer)
            value = rest[:end] if end >= 0 else rest
            rest = rest[end + close_len(rest, end, closer):] if end >= 0 else ""
            if value.startswith("\n"):
                value = value[1:]
            if value.endswith("\n"):
                value = value[:-1]
        v = param_value(value, pschema, root, notes) if pname else _SKIP
        if v is not _SKIP:
            args[pname] = v
    if recover and openers and dropped == openers and not args:
        # #6032 R5: every parameter the model began was unreadable (an opener cut before its `>`, a stub): the call is
        # not run with no arguments at all - it goes back as the text it is
        raise ValueError("malformed tool call: no readable parameter: " + body[:80])
    if recover and unwrap_arguments(args, props):
        args = args["arguments"]
    return ToolCall(name=name, arguments=args)


def unwrap_arguments(args: dict, props: dict) -> bool:
    """#5804 (recover): the model wrote the whole argument object as one `<parameter=arguments>` (a JSON object) and
    the tool has no parameter of that name: that object is the call's arguments.  True when `args` is exactly that one
    parameter."""
    return list(args) == ["arguments"] and isinstance(args["arguments"], dict) and "arguments" not in props


# ------------------------------------------------------------------------------------------------ harmony
HARMONY_ENDS = ("<|end|>", "<|return|>", "<|call|>")
HARMONY_MESSAGE = "<|message|>"
HARMONY_TURN = "<|start|>assistant"                    # the generation prompt's end
HARMONY_NO_THINKING = "<|channel|>analysis<|message|><|end|>"


class HarmonyParser:
    """The output parser for the Harmony format (llm-jp-4.1's `chat_format=llm-jp-harmony-v1`; GPT-OSS's), with
    OutputParser's interface (feed, finish, state).  After the generation prompt the model writes messages
    `[<|start|>assistant[ to=functions.NAME]]<|channel|>CHANNEL[ to=functions.NAME][<|constrain|>TYPE]<|message|>BODY`
    ended by <|end|>, <|return|> or <|call|>: the analysis channel's body is the reasoning, final's and a commentary
    without a recipient (a preamble) the answer, a body to `functions.NAME` that tool's JSON arguments.  As llama.cpp's
    LLM-jp-4.1 parser (common/parsers/llm-jp-harmony.cpp): a space after a special token is the tokenizer's, and
    parallel calls are consecutive messages, each ended by <|end|>."""

    def __init__(self, thinking: bool = True, tools: list[dict] | None = None, **_):
        self.thinking = thinking
        self.names = {t.get("name") for t in tools or []}
        self.head = True             # reading a message's header (up to <|message|>)
        self.buf = ""                # text held: a header, or a body's tail that may begin a terminator
        self.channel = ""
        self.to: str | None = None
        self.args = ""               # a call's body so far
        self.lead = False            # the body's first character is the tokenizer's space after <|message|>
        self.pending: list = []      # OutputParser's: calls waiting in the reasoning (none here)
        self.rescued = self.refused = 0

    @property
    def state(self) -> str:
        if self.head:
            return "content" if self.channel in ("final", "commentary") else "reasoning" if self.thinking else "content"
        if self.to is not None:
            return "call"
        return "reasoning" if self.channel == "analysis" else "content"

    def _header(self, text: str) -> None:
        to = re.search(r"to=functions\.([A-Za-z0-9_.\-]+)", text)
        self.to = to.group(1) if to else None
        ch = re.search(r"<\|channel\|>\s*([A-Za-z]+)", text)
        self.channel = ch.group(1) if ch else self.channel or "final"

    def _body_event(self, text: str) -> list[Event]:
        if not text:
            return []
        if self.to is not None:
            self.args += text
            return []
        return [Event("reasoning" if self.channel == "analysis" else "content", text)]

    def _end_message(self) -> list[Event]:
        out: list[Event] = []
        if self.to is not None:
            try:
                args = json.loads(self.args) if self.args.strip() else {}
            except ValueError:
                args = None
            if isinstance(args, dict):
                out.append(Event("tool_call", call=ToolCall(name=self.to, arguments=args)))
            else:                    # not a call's JSON: the text it is
                out.append(Event("content", self.args))
        self.head, self.to, self.args = True, None, ""
        return out

    def feed(self, delta: str) -> list[Event]:
        self.buf += delta
        out: list[Event] = []
        while self.buf:
            if self.head:
                at = self.buf.find(HARMONY_MESSAGE)
                if at < 0:
                    return out       # the header is still arriving
                self._header(self.buf[:at])
                self.buf = self.buf[at + len(HARMONY_MESSAGE):]
                self.head, self.lead = False, True
                continue
            if self.lead:
                if self.buf.startswith(" "):
                    self.buf = self.buf[1:]
                self.lead = False
            ends = [(self.buf.find(e), e) for e in HARMONY_ENDS if e in self.buf]
            if ends:
                at, tag = min(ends)
                out += self._body_event(self.buf[:at])
                self.buf = self.buf[at + len(tag):]
                out += self._end_message()
                continue
            # hold back a tail that could start a terminator
            keep = 0
            for e in HARMONY_ENDS:
                for k in range(min(len(e) - 1, len(self.buf)), 0, -1):
                    if self.buf.endswith(e[:k]):
                        keep = max(keep, k)
                        break
            out += self._body_event(self.buf[:len(self.buf) - keep])
            self.buf = self.buf[len(self.buf) - keep:]
            return out
        return out

    def finish(self, reason: str | None = None) -> list[Event]:
        out: list[Event] = []
        if not self.head:
            out += self._body_event(self.buf)
            self.buf = ""
            if self.to is not None and reason in (None, "stop"):
                # the stop token ended it: `<|call|>` is one (as in llama.cpp), and the engine does not pass it on
                out += self._end_message()
            elif self.to is not None:   # a call the output was cut inside: its text, never a call
                out.append(Event("content", self.args))
                self.to, self.args = None, ""
        elif self.buf.strip() and HARMONY_MESSAGE not in self.buf:
            self.buf = ""            # an unfinished header carries no text
        return out


RCALL_MAX = 32768        # the most of a `<tool_call>` inside the reasoning held while waiting for its end (#804)


class OutputParser:
    """Incremental parser of the model's text. Feed deltas; get events. A tag split across deltas is held back
    until it is complete, so clients never see `<tool_` or `</thi`."""
    _line_parts: list[str]              # the current line in pieces (the `line` property)
    _line_blocks: list[str] | None

    def __init__(self, thinking: bool = True, tools: list[dict] | None = None, stream_tools: bool = False,
                 recover: bool = False, declared_only: bool = False, deferred: bool = False):
        self.state = "reasoning" if thinking else "content"
        self.buf = ""
        self.lead = False
        self.schemas = {t.get("name"): t for t in tools or []}
        # #4663: the request defers tools (Anthropic's tool search, see anthropic_tools): the client holds back more
        # tools than it declares, so a call for an undeclared name is a call too - Claude Code runs it, where text
        # ended the turn.
        self.deferred = deferred
        # #804/#3075: in the answer too, a call counts only for a tool the request declared.  With tools declared a
        # call for another name stays text (the Responses spelling `namespace__tool` of a `namespace.tool` counts as
        # that tool); `declared_only` extends the rule to a request that declared none (Chat Completions, Anthropic:
        # the client has no tool to run; Responses but for a Codex compaction request, which keeps its conversation's
        # tools in the prompt only, and whose calls come back as before).
        self.declared_only = declared_only
        # A declared namespace.member tool is also accepted as namespace__member (PR #3556), namespace=member (Qwen
        # writes the <function= tag's equals sign into the name), namespace_member (one underscore dropped) and the
        # bare member (upstream fd8d37e93, #3972), without advertising a duplicate tool.  Only an unambiguous alias
        # counts (two namespaces can end in the same member), and an exactly declared name always wins over an alias.
        aliases: dict[str, set[str]] = {}
        for name in self.schemas:
            if isinstance(name, str) and "." in name:
                namespace, member = name.rsplit(".", 1)
                for alias in (f"{namespace}__{member}", f"{namespace}={member}", f"{namespace}_{member}", member):
                    aliases.setdefault(alias, set()).add(name)
        # #3796 (upstream 805ebe17b): the same tool written with every `__` collapsed to `_`
        # (`mcp__playwright__browser_navigate` as `mcp_playwright_browser_navigate`, the way Qwen often flattens a
        # name) is that tool.  Only a spelling of a declared name: a reordered or renamed tool (`server1_label_values`
        # for `label_values_mcp_server1`) is not guessed.
        for name in self.schemas:
            if isinstance(name, str) and "__" in name:
                aliases.setdefault(name.replace("__", "_"), set()).add(name)
                # #5281: a namespaced MCP tool (`mcp__assets__get_asset_info`) written with its last segment only
                # (`get_asset_info`), as the bare member of a dotted name is accepted above; only when exactly one
                # declared tool ends that way
                last = name.rsplit("__", 1)[1]
                if last:
                    aliases.setdefault(last, set()).add(name)
        # #4751: a programmatic-tool-calling harness (the DeepSeek Harness's PTC mode) shows the model its tools as
        # `tools.write(...)` inside code, and the model then writes the native call with that spelling (`tools.write`,
        # also `functions.write`).  The prefix is not part of the declared name; the same unambiguous-alias rule
        # applies, an exactly declared name always wins.
        for name in self.schemas:
            if isinstance(name, str) and name:
                for prefix in ("tools.", "functions.", "default_api."):
                    aliases.setdefault(prefix + name, set()).add(name)
        self.aliases = {alias: next(iter(names)) for alias, names in aliases.items()
                        if len(names) == 1 and alias not in self.schemas}
        self.unread = 0              # #3441: `<tool_call>` blocks that opened like a call and were returned as text
        self.unread_names: list[str] = []   # the names (first few) of those naming a tool the request did not declare
        # stream_tools: a tool call is also reported while it is being written - "tool_start" (its name and id) as
        # soon as the name is known, then "tool_args" pieces of its JSON arguments (string parameters character by
        # character; other types whole, once complete) - before the final "tool_call".  Without it, a client sees
        # nothing until the call is complete, which for a large file write can be many minutes.
        self.stream_tools = stream_tools
        # #804/#1058: calls found inside the reasoning wait here as [raw text, ToolCall | None] until the turn shows
        # they were acts: only whitespace (or more calls) after them, then the end of the turn or `</think>`.  The
        # reasoning text before them is tracked (code fence, inline code, current line) to tell an act from a quote.
        self.pending: list[list] = []
        self.rescued = 0             # calls delivered from the reasoning
        self.notes: list[str] = []   # #4427: what was repaired in the calls' arguments (a damaged JSON array or object)
        self.after_think = False     # #4077: the reasoning ended at a bare `</think>` (another in the answer: a stray)
        self.stray_think = 0         # ... and the strays of that kind dropped from the answer
        self.refused = 0             # declared calls kept as reasoning text (quoted, or the turn was cut)
        self.fence, self.line, self.ticks = "", "", 0
        # #4703: the line before was blank / this is inside an indented (4-space) code block
        self._prev_blank = self._indented = False
        # recover (on in the server, off for a bare OutputParser; the config's "tool_call_recovery": false turns it
        # off): a declared tool's call written next to the template's
        # form is the call - `<parameter=NAME>` as the opener, JSON inside <tool_call>, a bare <function=NAME> at the
        # start of a line outside code, or a second call in the same <tool_call> (which otherwise merges into one).
        self.recover = recover
        self.bare = False            # the call being read opened with <function= alone (no <tool_call> around it)
        self.rbare = False           # #5804: the call held in the reasoning ("rcall") opened with <function= alone
        self.batch = False           # the call being finished is followed by another in the same wrapper
        self._final = False          # #4701: the turn has ended: a call held for what might follow it is decided now
        # #5276/#5409: the end of a misspelt opener or a prefixed name still arriving (see _prefixed)
        self._held = ""
        self._reset_scan()

    def _ok_at(self, p: int) -> bool:
        """self.buf[p] would open a call: at the start of a line, outside a code fence and inline code."""
        snap = self._snap()
        self._track(self.buf[:p])
        ok = self._opener_ok()
        self._restore(snap)
        return ok

    def _opens_at(self, p: int) -> bool:
        """self.buf[p] would open a call as `<tool_call>` does: in the reasoning at the start of a line outside code, in
        the answer anywhere outside code (a fence, inline code, an indented block)."""
        if self.state == "reasoning":
            return self._ok_at(p)
        snap = self._snap()
        self._track(self.buf[:p])
        ok = not self._in_code()
        self._restore(snap)
        return ok

    def _callable(self, name: str) -> bool:
        """#3075: may a call for `name` be a call?  A tool the request declared (or its Responses spelling); with
        none declared, only where `declared_only` is off."""
        if not self.schemas:
            return not self.declared_only
        return name in self.schemas or name in self.aliases or self.deferred

    def _loose_open(self, after: str):
        """#3441: `after` (text after a `<tool_call>`, whitespace dropped) opens a call in a drifted form that names a
        declared tool -> (name, length of the opener); "wait" while it may still become one; None otherwise."""
        m = _LOOSE_OPEN.match(after)
        if m:
            name = m.group(1)
            return (name, m.end()) if self._callable(name) else None
        if ">" in after or "\n" in after or not after.startswith("<"):
            return None
        if _LOOSE_PART.match(after):
            bare = after[1:].strip().lstrip("=").strip()
            if after[1:].lstrip().startswith(("=", "function")):
                return "wait"
            if any(k.startswith(bare) for k in list(self.schemas) + list(self.aliases) if isinstance(k, str)):
                return "wait"
        return None

    def _follower(self, after: str) -> str:
        """recover: what a `<tool_call>` followed (after whitespace) by `after` opens: "call" (`<function=`), "param"
        (a declared tool's name written as `<parameter=NAME>`), "json" (`{`), "wait" (not decided yet) or "text"."""
        if after.startswith(FUNC_START):
            return "call"
        missing = _MISSING_FUNC_OPEN.match(after)
        if missing:
            return "missing" if self._callable(missing.group(1)) else "text"
        if "\n" not in after:
            partial = after.strip()
            if partial and "function=".startswith(partial):
                return "wait"
            m = re.fullmatch(r"function\s*=\s*([A-Za-z0-9_.\-]*)", partial)
            if m and any(name.startswith(m.group(1)) for name in self.schemas if isinstance(name, str)):
                return "wait"
        if after.startswith(PARAM_START):
            gt = after.find(">")
            if gt < 0:
                return "wait" if "\n" not in after else "text"
            return "param" if after[len(PARAM_START):gt] in self.schemas else "text"
        if after.startswith("{"):
            return "json" if self.schemas else "text"
        if after.startswith("```") or (after and "```".startswith(after)):
            # #2037: the JSON call inside a Markdown fence (```json ... ```) in the explicit wrapper
            m = _JSON_FENCE.match(after)
            if m:
                rest = after[m.end():]
                if rest.startswith("{"):
                    return "json" if self.schemas else "text"
                return "wait" if not rest else "text"
            return "wait" if "\n" not in after else "text"
        if not after or FUNC_START.startswith(after) or PARAM_START.startswith(after):
            return "wait"
        return "text"

    def _bare_opener(self) -> tuple[int, bool]:
        """recover: the first `<function=` in self.buf that opens a call without the wrapper (start of a line, outside
        code, a declared tool) -> (position, decided); (-1, False) for none.  A name still arriving is undecided."""
        p = self.buf.find(FUNC_START)
        while p >= 0:
            if self._ok_at(p):
                gt = self.buf.find(">", p)
                raw_name = self.buf[p + len(FUNC_START):gt if gt >= 0 else len(self.buf)]
                name = raw_name.strip()
                name = self.aliases.get(name, name)
                if gt < 0 and re.fullmatch(r"\s*[^<>\s]*\s*", raw_name):
                    return p, False
                if gt >= 0 and name in self.schemas:
                    return p, True
            p = self.buf.find(FUNC_START, p + 1)
        return -1, False

    def _dialect_opener(self):
        """#6022 (recover): where Claude's dialect starts without the template's wrapper in the answer:
        `<function_calls>` followed by an `<invoke name="T">` of a declared tool (outside code), or such an `<invoke
        name="T">` alone at the start of a line that ends its line.
        -> (position, "fc" | "invoke" | "hold") or None; "hold": the opener or its name is still arriving."""
        found = None
        for tag in (FUNCTION_CALLS_START, "<invoke "):
            p = self.buf.find(tag)
            while p >= 0 and (found is None or p < found[0]):
                if tag == "<invoke ":
                    ok = self._ok_at(p)
                    rest = self.buf[p:]
                else:
                    ok = self._opens_at(p)
                    rest = self.buf[p + len(tag):].lstrip()
                    if ok and not rest:
                        found = (p, "hold")
                        break
                if ok:
                    m = _INVOKE_OPEN.match(rest)
                    if m:
                        line_end = rest[m.end():].lstrip(" \t")[:1]
                        if tag == "<invoke ":
                            if not line_end:
                                found = (p, "hold")
                                break
                            if line_end == "\n" and self._callable(m.group(1)):
                                found = (p, "invoke")
                                break
                        elif self._callable(m.group(1)):
                            found = (p, "fc")
                            break
                    elif _INVOKE_PART.match(rest) and "\n" not in rest:
                        found = (p, "hold")
                        break
                p = self.buf.find(tag, p + 1)
        return found

    def _bare_in_reasoning(self):
        """#5804 (recover): a `<function=NAME>` of a declared tool that opens a call in the reasoning without its
        wrapper (start of a line, outside code)
        -> (position, decided), None for none; a name still arriving is undecided."""
        b, decided = self._bare_opener()
        return (b, decided) if b >= 0 else None

    @property
    def line(self) -> str:
        """Materialize the current line only for a decision, newline, or state snapshot."""
        if self._line_blocks is not None:
            self._line_blocks.extend(self._line_parts)
            self._line_parts = ["".join(self._line_blocks)]
            self._line_blocks = None
        elif len(self._line_parts) > 1:
            self._line_parts = ["".join(self._line_parts)]
        return self._line_parts[0] if self._line_parts else ""

    @line.setter
    def line(self, text: str):
        self._line_parts = [text] if text else []
        self._line_blocks = None

    def _track(self, text: str) -> str:
        """Follow the reasoning text that has gone out: the open code fence, the current line and the inline-code
        parity of the paragraph.  Returns the text."""
        # Most token fragments stay on the current line. Fence and line
        # bookkeeping happens only when a newline completes that line.
        if "\n" not in text:
            if text:
                self._line_parts.append(text)
                if len(self._line_parts) == 64:
                    # Release small fragment objects in bounded groups. Completed
                    # blocks are never recopied until the full line is requested.
                    if self._line_blocks is None:
                        self._line_blocks = []
                    self._line_blocks.append("".join(self._line_parts))
                    self._line_parts.clear()
            return text
        parts = text.split("\n")
        for k, part in enumerate(parts):
            if k < len(parts) - 1:
                line = self.line + part if self._line_parts or self._line_blocks else part
                self._line_parts.clear()
                self._line_blocks = None
                s = line.lstrip()
                was_fenced = bool(self.fence)
                if self.fence:
                    # #4702: a fence is closed only by a line of at least as many of the same character as opened it
                    # (CommonMark), so a
                    # ``` example inside a ```` file stays inside it
                    run = len(s) - len(s.lstrip(self.fence[0]))
                    if run >= len(self.fence):
                        self.fence = ""
                elif s[:3] in ("```", "~~~") and (s[0] * 3) not in s[3:]:
                    self.fence = s[0] * (len(s) - len(s.lstrip(s[0])))
                elif not s:
                    self.ticks = 0
                elif self.ticks % 2:
                    # a span opened at a line's end goes on until closed
                    self.ticks = (self.ticks + line.count("`")) % 2
                else:
                    # #4923: the odd count of a line with one unpaired backtick in its text ("use `run_commands to
                    # check") was carried to the end of the paragraph, so every `<tool_call>` after it read as quoted
                    # inline code and the turn ended with no call.  Only a backtick that ends its line, or is followed
                    # on it by markup (`<tool_call>`), opens a span that goes on to the next line (the quote "`" + line
                    # break + call + "`").
                    odd = line.count("`") % 2
                    self.ticks = 1 if odd and line[line.rfind("`") + 1:].strip()[:1] in ("", "<") else 0
                # #4703: an indented (4 spaces or a tab) code block starts after a blank line and goes on until a line
                # that is not indented
                if was_fenced or self.fence:
                    self._prev_blank = self._indented = False
                elif not s:
                    self._prev_blank = True
                elif line.startswith(("    ", "\t")):
                    self._indented = self._prev_blank or self._indented
                    self._prev_blank = False
                else:
                    self._prev_blank = self._indented = False
            else:
                if part:
                    self._line_parts.append(part)
        return text

    def _opener_ok(self) -> bool:
        """The reasoning text so far puts a `<tool_call>` at the start of a line, outside a code fence and outside
        inline code (`_in_code`)."""
        return not self.fence and not self.line.strip() and self.ticks % 2 == 0 and not self._in_indented_block()

    def _in_indented_block(self) -> bool:
        """#4703: the current line is part of an indented code block: four spaces or a tab, after a blank line or
        another such line."""
        return (self._prev_blank or self._indented) and self.line.startswith(("    ", "\t"))

    def _snap(self) -> tuple:
        return (self.fence, self.line, self.ticks, self._prev_blank, self._indented)

    def _restore(self, snap: tuple) -> None:
        self.fence, self.line, self.ticks, self._prev_blank, self._indented = snap

    def _reset_track(self) -> None:
        self.fence, self.line, self.ticks = "", "", 0
        self._prev_blank = self._indented = False

    def _in_code(self) -> bool:
        """The text so far leaves the next character inside a code fence, inline code or an indented code block."""
        return bool(self.fence) or (self.ticks + self.line.count("`")) % 2 == 1 or self._in_indented_block()

    def _quoted_think_end(self, i: int) -> bool | None:
        """#537: the `</think>` at self.buf[i] is the model quoting the tag while it thinks, not the end of the
        thinking: written right after an OPENING quote or backtick and followed by anything but a line break, as in
        `</think>` or "</think>...", or closed by the same quote right after it.  A quote that closes a sentence
        (`...number only."</think>2`) is no quoting of the tag: GLM ends short reasoning that way and answers on the
        same line, which the old test kept as reasoning, leaving the answer empty.  None while the character after the
        tag has not arrived."""
        head = self.line + self.buf[:i]
        before = self.buf[i - 1] if i else self.line[-1:]
        after = self.buf[i + len(THINK_END):]
        if before not in ('"', "'", "`"):
            # #6218: a backtick span the model opened (an odd count in the paragraph so far) that the tag sits inside,
            # written on its own line: "the marker `" + line break + tag + line break + "`" - the backtick after the
            # tag closes the span the tag is quoted in
            if (head.rstrip().endswith("`") or not head.strip()) and (self.ticks + head.count("`")) % 2 == 1:
                if not after.strip():
                    return None
                return after.lstrip().startswith("`")
            return False
        # #5605: a quote that CLOSES a sentence ("... said "Reply with the number only."" + tag + answer) is not the
        # opening of a quotation the tag is in: a quote right after a non-space character (a full stop, a letter)
        # closes one.  A quote after a space, a bracket, an `=` or at the start of a line opens one, as in
        # `returns "</think>`.  (Two upstream lanes fixed #5605; both rules are kept: the pair rule, then the
        # closing-quote rule.)
        if not after:
            return None
        if after[0] == before:                       # "</think>" / `</think>`: the tag inside a pair of quotes
            return True
        prev = head[-2:-1]
        if prev and not prev.isspace() and prev not in "([{<=:,;":
            return False
        return after[0] not in "\r\n"

    def _stray_think_at(self) -> int:
        """#4077: where self.buf has a `</think>` at the start of a line, in the answer of a turn whose reasoning
        already ended at one; -1 for none.  A fence or backtick is no reason to keep it: the reasoning may have left
        one open, and the answer's own tracking of code is wrong from there on."""
        k = self.buf.find(THINK_END)
        while k >= 0:
            before = self.buf[:k]
            if not (before.rsplit("\n", 1)[1] if "\n" in before else self.line + before).strip():
                return k
            k = self.buf.find(THINK_END, k + 1)
        return -1

    def _release(self, deliver: bool) -> list[Event]:
        """Settle the calls waiting in self.pending: events for real calls, or all of it back as reasoning text."""
        out = []
        for raw, call in self.pending:
            if call is not None and deliver:
                out.append(Event("tool_call", call=call))
                self.rescued += 1
            elif call is None or not deliver:
                if call is not None:
                    self.refused += 1
                if raw:
                    out.append(Event("reasoning", self._track(raw)))
        self.pending = []
        return out

    def _reset_scan(self):
        self._call_end_checked = 0   # closing-tag search progress for the current wrapped call
        self._function_end_checked = 0
        self._scan_parts = []        # upstream #4686: consumed call body, kept in chunks until a walk needs the whole
        self.sp = 0                  # how much of self.buf (the call body) the scanner has consumed
        self.ss = "name"             # name -> between -> str|raw -> ... -> function_done -> done
        self.scall = None            # the ToolCall being streamed (its id is reused by the final event)
        self.sfirst = True
        self.sval_started = False
        self.sprops = {}             # the streamed call's parameters' schemas
        self.sroot = None            # their reference resolution root (the full parameters schema)
        self.spname = ""             # the parameter being streamed
        self.sheld = {}              # upstream #5493: the non-string parameters, sent when the call closes (a later
        #                              write wins)
        self.svstart = 0             # where its string value starts in self.buf (#4078)
        self.sset = set()            # upstream #6032 R2: the parameters set so far (a damaged opener counts only for
        #                              the others)
        self.snames = None           # ... the declared ones not set yet, for the current value's end (recover only)
        self.scloser = None          # upstream #6032 R4: `</NAME>` when the current value was opened as the element
        #                              `<NAME>`

    def _restore_scan_body(self) -> None:
        """upstream #4686: put compacted call text back before a structural walk or the final parse."""
        if not self._scan_parts:
            return
        prefix = "".join(self._scan_parts)
        shift = len(prefix)
        self.buf = prefix + self.buf
        self.sp += shift
        if self.ss == "str" and self.sval_started:
            self.svstart += shift
        if self._call_end_checked >= 0:
            self._call_end_checked += shift
        if self._function_end_checked >= 0:
            self._function_end_checked += shift
        self._scan_parts.clear()

    def _compact_scan_body(self) -> None:
        """upstream #4686: move streamed, consumed call text out of the string that grows on every token.  An open
        string value is walked from its start (param_end over buf[svstart:]) each time a closer may have arrived: its
        text stays in self.buf, so only what lies before it moves out (upstream 5b068304c)."""
        cut = self.sp
        if self.ss == "str" and self.sval_started:
            cut = min(cut, self.svstart)
        if cut < 4096:
            return
        self._scan_parts.append(self.buf[:cut])
        self.buf = self.buf[cut:]
        self.sp -= cut
        if self.ss == "str" and self.sval_started:
            self.svstart -= cut
        if self._call_end_checked >= 0:
            self._call_end_checked = max(0, self._call_end_checked - cut)
        if self._function_end_checked >= 0:
            self._function_end_checked = max(0, self._function_end_checked - cut)

    def _function_end(self) -> int:
        """upstream #4686: the full function walk only when newly appended text holds a possible closer."""
        start = max(0, self._function_end_checked - len(FUNC_END) + 1)
        if self.buf.find(FUNC_END, start) < 0:
            self._function_end_checked = len(self.buf)
            return -1
        self._restore_scan_body()
        end = function_end(self.buf)
        self._function_end_checked = -1 if end >= 0 else len(self.buf)
        return end

    def _call_names(self):
        """#6032 R4 (recover): the declared parameters of the tool the call in self.buf names, for call_end's walk."""
        if not self.recover:
            return None
        m = re.match(r"\s*<function=([^<>\n]*)>", self.buf)
        if not m:
            return None
        name = m.group(1).strip()
        name = self.aliases.get(name, name)
        return set(_properties(self.schemas.get(name))) or None   # (a boolean schema has none)

    def _wrapped_call_end(self) -> int:
        # Without a closing tag the structural walk cannot finish. Search only newly appended text plus enough
        # overlap for a tag split across feeds.  A literal closing tag inside a value needs structural disambiguation:
        # once a candidate appears, compacted text comes back for the structural walk; if it was a literal tag in a
        # value, the next walk waits for another newly appended candidate instead of walking the unchanged body for
        # every token (upstream #4686, recovery included).
        if self._call_end_checked >= 0:
            start = max(0, self._call_end_checked - len(CALL_END) + 1)
            if self.buf.find(CALL_END, start) < 0:
                self._call_end_checked = len(self.buf)
                return -1
        self._restore_scan_body()
        names = self._call_names()
        end = call_end(self.buf, self._final, names=names)
        # #4701: a complete closing sequence may be waiting for lookahead or finish(), not another closing tag.
        pending = end < 0 and not self._final and call_end(self.buf, final=True, names=names) >= 0
        self._call_end_checked = -1 if end >= 0 or pending else len(self.buf)
        return end

    def _scan(self) -> list[Event]:
        """Advance the streaming view of the call body in self.buf (see stream_tools)."""
        out = []

        def args(s):
            if s:
                out.append(Event("tool_args", s, call=self.scall))
        while True:
            rest = self.buf[self.sp:]
            if self.ss == "name":
                a = rest.find("<function=")
                b = rest.find(">", a + 10) if a >= 0 else -1
                if b < 0:
                    return out
                raw_name = rest[a + 10:b]
                name = raw_name.strip()
                name = self.aliases.get(name, name)
                if raw_name != raw_name.strip() and name not in self.schemas:
                    return out
                self.scall = ToolCall(name=name, arguments={})
                tool = self.schemas.get(name)
                self.sprops = _properties(tool)
                self.sroot = tool.get("parameters") if isinstance(tool, dict) else None
                out.append(Event("tool_start", call=self.scall))
                args("{")
                self.sp += b + 1
                self.ss = "between"
            elif self.ss == "between":
                stripped = rest.lstrip()
                self.sp += len(rest) - len(stripped)
                got = None
                if self.recover and stripped.startswith("{") and not self.sset:
                    # #6022: `<function=T>{"k": v}</function>`: the JSON object is the call's arguments (as
                    # parse_tool_call reads it), held until whole
                    whole = json_args(ARGS_START + stripped, self.sprops, root=self.sroot)
                    if whole is None:
                        return out
                    tail = stripped[whole[1] - len(ARGS_START):].strip()
                    if tail == "" or tail.startswith(FUNC_END) or FUNC_END.startswith(tail):
                        for k, v in whole[0].items():
                            self.sheld.pop(k, None)
                            self.sheld[k] = v
                            self.sset.add(k)
                        self.sp += whole[1] - len(ARGS_START)
                        continue
                if self.recover and (stripped.startswith(ARGS_START) or (stripped and ARGS_START.startswith(stripped))):
                    # #6032 R3: the arguments as one JSON object: held until it is whole, then sent with the call's
                    # other non-string values when it closes (as parse_tool_call reads it); a cut one sends nothing
                    whole = (json_args(stripped, self.sprops, root=self.sroot) if stripped.startswith(ARGS_START)
                             else None)
                    if whole is None:
                        return out
                    for k, v in whole[0].items():
                        self.sheld.pop(k, None)
                        self.sheld[k] = v
                        self.sset.add(k)
                    self.sp += whole[1]
                    continue
                if self.recover and not stripped.startswith(FUNC_END):
                    # upstream #6032 R2: the opener parse_tool_call reads here (a damaged one of a declared parameter
                    # not set yet, or the colon form); other text up to the next `<parameter=` is skipped, as there
                    got = opener_at(stripped, set(self.sprops) - self.sset, self._final)
                    if got == "pending":
                        return out
                    if got is None and not stripped.startswith("<parameter="):
                        k = stripped.find("<parameter=")
                        if k < 0:
                            return out          # trailing text, or a tag still arriving: wait
                        self.sp += k
                        continue
                if got is not None or stripped.startswith("<parameter="):
                    if got is None:
                        b = stripped.find(">")
                        if b < 0:
                            return out
                        got = (param_name(stripped[11:b]), b + 1, None)   # upstream #6032 R1: None = read, dropped
                    pname, length, inline = got
                    self.sp += length
                    self.spname = pname
                    self.snames = (set(self.sprops) - self.sset - {pname}) if self.recover else None
                    text_only = pname is not None and _text_only(self.sprops.get(pname), self.sroot)
                    # upstream #6032 R4: a value opened as the element `<NAME>` may end at `</NAME>` (as parse_tool_call
                    # reads it)
                    self.scloser = ("</%s>" % pname if self.recover and pname and stripped.startswith("<%s>" % pname)
                                    else None)
                    if inline is not None:      # the colon form: its value is on the opener's line
                        v = param_value(inline, self.sprops.get(pname), self.sroot) if pname else _SKIP
                        if v is not _SKIP and text_only:
                            args(("" if self.sfirst else ",") + json.dumps(pname) + ":" + json.dumps(v))
                            self.sfirst = False
                            self.sset.add(pname)
                        elif v is not _SKIP:
                            self.sheld.pop(pname, None)
                            self.sheld[pname] = v
                            self.sset.add(pname)
                        continue
                    if text_only:
                        self.sset.add(pname)
                        args(("" if self.sfirst else ",") + json.dumps(pname) + ':"')
                        self.sfirst = False
                        self.ss, self.sval_started = "str", False
                    else:
                        self.ss = "raw"
                elif stripped.startswith("</function>"):
                    self.sp += len("</function>")
                    self.ss = "function_done"
                else:
                    return out          # a tag still arriving (or trailing text): wait
            elif self.ss == "str":
                if not self.sval_started:
                    if not rest:
                        return out
                    if rest[0] == "\n":          # the value's leading newline is not part of it
                        self.sp += 1
                        rest = rest[1:]
                    self.sval_started = True
                    self.svstart = self.sp
                # the walk starts at the value (a quoted call's opener counts); only a close in the new text can end it
                closers = (PARAM_END, self.scloser) if self.scloser else (PARAM_END,)
                end = param_end(self.buf[self.svstart:], self._final, name=self.spname, allow_reopen=True,
                                names=self.snames, closer=self.scloser) if any(c in rest for c in closers) else -1
                end = end - (self.sp - self.svstart) if end >= 0 else end
                if end >= 0:
                    value = rest[:end]
                    if value.endswith("\n"):
                        value = value[:-1]
                    args(json.dumps(value)[1:-1] + '"')
                    self.sp += end + close_len(rest, end, self.scloser)
                    self.ss = "between"
                    continue
                # an undecided </parameter> (-2) is held from its start, like a tag still arriving
                safe = min(k for k in (rest.find(c) for c in closers) if k >= 0) if end == -2 \
                    else len(rest) - self._hold(rest, closers)
                if safe > 0 and rest[safe - 1] == "\n":   # may be the trailing newline before </parameter>
                    safe -= 1
                if safe > 0:
                    args(json.dumps(rest[:safe])[1:-1])
                    self.sp += safe
                return out
            elif self.ss == "raw":
                end = param_end(rest, self._final, name=self.spname,       # parse_tool_call's rule (#5493)
                                allow_reopen=_reopens(self.sprops.get(self.spname), self.sroot), names=self.snames,
                                closer=self.scloser)
                if end < 0:
                    return out
                value = rest[:end]
                if value.startswith("\n"):
                    value = value[1:]
                if value.endswith("\n"):
                    value = value[:-1]
                v = (param_value(value, self.sprops.get(self.spname), self.sroot)   # the note: with the final call
                     if self.spname is not None else _SKIP)
                if v is not _SKIP:
                    # upstream #5493: held until the call closes, so a parameter the model writes again (anywhere later
                    # in the call) streams once, with the value the final call has (the later write)
                    self.sheld.pop(self.spname, None)
                    self.sheld[self.spname] = v
                    self.sset.add(self.spname)
                self.sp += end + close_len(rest, end, self.scloser)
                self.ss = "between"
            else:
                return out

    def _close_scan(self) -> list[Event]:
        """Close streamed JSON only after its call boundary has been confirmed."""
        out: list[Event] = []
        if self.scall is None or self.ss == "done":
            return out
        tail = ""
        if self.ss == "str":
            tail += '"'
        elif self.ss == "raw" and self.spname is not None:   # a value the output ended inside: its text, as before
            self.sheld.pop(self.spname, None)
            self.sheld[self.spname] = self.buf[self.sp:].strip("\n")
        held = self.sheld
        if self.recover and self.sfirst and unwrap_arguments(held, self.sprops):
            held = held["arguments"]            # #5804: a lone `<parameter=arguments>` object is the call's arguments
        for k, v in held.items():               # upstream #5493: the non-string parameters, each once, the later write
            tail += ("" if self.sfirst else ",") + json.dumps(k) + ":" + json.dumps(v, ensure_ascii=False)
            self.sfirst = False
        self.sheld = {}
        tail += "}"
        out.append(Event("tool_args", tail, call=self.scall))
        self.ss = "done"
        return out

    def _hold(self, text: str, tags: tuple[str, ...]) -> int:
        """Length of the longest suffix of `text` that is a proper prefix of one of `tags`."""
        best = 0
        size = len(text)
        for tag in tags:
            if len(tag) <= 1:
                continue
            # Only a suffix beginning with the tag's first character can match.
            # Ordinary token deltas contain no '<': one C-level search replaces
            # checking every possible prefix. Earliest candidate is the longest.
            start = text.find(tag[0], max(0, size - len(tag) + 1))
            while start >= 0 and size - start > best:
                if tag.startswith(text[start:]):
                    best = size - start
                    break
                start = text.find(tag[0], start + 1)
        return best

    def _prefixed(self, fresh: int) -> str:
        """#5409: a call written `<function=tools NAME>` (a space for the dot of `tools.NAME`) is the call NAME, and one
        written `<function=tools>` (the bare prefix) is the one declared tool that takes the call's parameters.
        Rewrites the complete ones in self.buf to `<function=NAME>` and returns the tail of a call still arriving, which
        feed() holds back until the next delta.  Only a declared name is ever the result; with none, or with two that
        fit, the text is left as it was. `fresh`: the length of the text just added; only the end of the buffer (that
        and a little before it) can hold a new match, so a long call is not scanned again with every token."""
        held = self._opener(fresh)
        start = max(0, len(self.buf) - fresh - _PREFIX_LOOKBACK)
        buf = self.buf[start:]
        if "<function=" in buf:
            def spaced(m):
                name = m.group(2).strip()
                return f"<function={name}>" if name in self.schemas else m.group(0)
            buf = _PREFIX_SPACED.sub(spaced, buf)
            m = _PREFIX_PART.search(buf)
            if m and not held:
                held, buf = buf[m.start():], buf[:m.start()]
            if self.recover:
                def bare(m):
                    body = m.group(3)
                    names = re.findall(r"<parameter=\s*([^<>\s]+?)\s*>", body)
                    tool = None if m.group(2) in self.schemas else self._tool_by_params(names)
                    return f"{m.group(1)}<function={tool}>{body}</function>" if tool else m.group(0)
                buf = _PREFIX_BARE.sub(bare, buf)
                m = _PREFIX_BARE_OPEN.search(buf)
                if m and m.group(1) not in self.schemas and not held and len(buf) - m.start() <= RCALL_MAX:
                    held, buf = buf[m.start():], buf[:m.start()]
            if buf != self.buf[start:]:
                self.buf = self.buf[:start] + buf
        return held

    def _opener(self, fresh: int) -> str:
        """#5276: `<toolcall>` / `<tool-call>` (the opener written as text, misspelled) followed by `<function=NAME>` of
        a declared tool is `<tool_call>`, where a `<tool_call>` would open a call (in the reasoning the start of a line,
        outside code; in the answer outside code).  Prose that names the misspelled tag, or any other follower, is left
        as it was.  Returns the tail of an opener still arriving (held back by feed)."""
        start = max(0, len(self.buf) - fresh - _PREFIX_LOOKBACK)
        if "<tool" not in self.buf[start:]:
            return ""
        buf = self.buf
        pos, out = start, []
        for m in _MISSPELT_OPEN.finditer(buf, start):
            name = m.group(2).strip()
            name = self.aliases.get(name, name)
            if name in self.schemas and self._opens_at(m.start()):
                out.append(buf[pos:m.start()] + CALL_START + m.group(1) + FUNC_START + m.group(2) + ">")
                pos = m.end()
        buf = buf[:start] + "".join(out) + buf[pos:] if out else buf
        held = ""
        part = _MISSPELT_PART.search(buf, start)
        tail = buf[part.start():] if part else ""
        if not tail:                                        # the opener itself still arriving: a prefix of `<toolcall>`
            for k in range(min(len(buf), 11), 5, -1):
                if any(w.startswith(buf[-k:]) for w in ("<toolcall>", "<tool-call>")) and buf[-k:] != "<tool_":
                    tail = buf[-k:]
                    break
        if tail:
            held, buf = tail, buf[:len(buf) - len(tail)]
        self.buf = buf
        return held

    def _tool_by_params(self, names: list[str]) -> str | None:
        """The one declared tool whose properties hold every one of `names` and whose required parameters are all in
        them."""
        found = []
        for tname, schema in self.schemas.items():
            params = (schema or {}).get("parameters") or {}
            props = params.get("properties") or {}
            if names and all(n in props for n in names) and all(r in names for r in params.get("required") or []):
                found.append(tname)
        return found[0] if len(found) == 1 else None

    def feed(self, delta: str) -> list[Event]:
        if self._held:
            delta, self._held = self._held + delta, ""
        self.buf += delta
        if self.schemas and delta:
            self._held = self._prefixed(len(delta))
        return self._feed()

    def _feed(self) -> list[Event]:
        out: list[Event] = []
        while True:
            if self.state == "rcall":
                # #804: a `<tool_call>` inside the reasoning, with tools declared.  It is held whole (never streamed
                # as a call) until it ends: a declared name is then a tool call, anything else stays reasoning text.
                lead = 0 if self.rbare else len(CALL_START)   # #5804: a bare `<function=` call has no wrapper
                body = self.buf[lead:]
                end, think = call_end(body, hold=False), body.find(THINK_END)
                drop = len(CALL_END)
                # #4427: as in the answer, a following call opener confirms an omitted closer
                f = function_end(body)
                if f >= 0 and (end < 0 or f < end) and re.match(r"<tool_call>\s*<function=\s*[^<>\s]+\s*>",
                                                                body[f:].lstrip()):
                    end, drop = f, 0
                if self.rbare:
                    end, drop = f, 0                 # ... it ends at its own `</function>`
                if end >= 0 and (think < 0 or think >= end):
                    call = None
                    try:
                        name = body[:end].strip()[len("<function="):].split(">", 1)[0].strip()
                        name = self.aliases.get(name, name)
                        if self._callable(name):
                            call = parse_tool_call(body[:end], self.schemas.get(name), self.notes, recover=self.recover)
                            call.name = name
                    except ValueError:
                        pass
                    raw = self.buf[:lead + end + drop]
                    if call is not None or self.pending:
                        self.pending.append([raw, call])    # settled by what follows it (see __init__)
                    else:
                        out.append(Event("reasoning", self._track(raw)))
                    self.buf = body[end + drop:]
                    self.state, self.rbare = "reasoning", False
                elif think >= 0:                     # the thinking ended inside it: it never was a call
                    out += self._release(False)
                    out.append(Event("reasoning", self._track(self.buf[:lead + think])))
                    self.buf = body[think:]
                    self.state, self.rbare = "reasoning", False
                elif len(body) > RCALL_MAX:          # a tag in the prose that never closes: stop holding back
                    out += self._release(False)
                    out.append(Event("reasoning", self._track(self.buf)))
                    self.buf = ""
                    self.state, self.rbare = "reasoning", False
                else:
                    return out
            elif self.state == "reasoning":
                if self.pending:
                    # calls wait for what follows: more calls or whitespace keep them, `</think>` makes them acts,
                    # any other text means they were quoted
                    stripped = self.buf.lstrip()
                    if len(stripped) < len(self.buf):
                        self.pending[-1][0] += self.buf[:len(self.buf) - len(stripped)]
                        self.buf = stripped
                    if not self.buf:
                        return out
                    bare = self._bare_in_reasoning() if self.recover and self.schemas else None
                    if self.buf.startswith(CALL_START):
                        self.state = "rcall"
                    elif bare is not None and bare[0] == 0:          # #5804: a bare `<function=` call after a call
                        if not bare[1]:
                            return out
                        self.state, self.rbare = "rcall", True
                    elif self.buf.startswith(THINK_END):
                        out += self._release(True)
                    elif CALL_START.startswith(self.buf) or THINK_END.startswith(self.buf) \
                            or (bare is None and self.recover and self.schemas and FUNC_START.startswith(self.buf)):
                        return out
                    else:
                        out += self._release(False)
                    continue
                i = self.buf.find(THINK_END)
                tool = self.buf.find(CALL_START) if self.schemas else -1
                bare = self._bare_in_reasoning() if self.recover and self.schemas else None
                if bare is not None and (i < 0 or bare[0] < i) and (tool < 0 or bare[0] < tool):
                    # #5804 (recover): a declared tool's `<function=NAME>` at the start of a line, outside code, is a
                    # call in the reasoning too
                    if bare[0]:
                        out.append(Event("reasoning", self._track(self.buf[:bare[0]])))
                        self.buf = self.buf[bare[0]:]
                    if not bare[1]:                  # the name has not arrived: held
                        return out
                    self.state, self.rbare = "rcall", True
                    continue
                if tool >= 0 and (i < 0 or tool < i):
                    if tool:
                        out.append(Event("reasoning", self._track(self.buf[:tool])))
                    if self._opener_ok():
                        self.buf = self.buf[tool:]
                        self.state = "rcall"
                    else:                            # mid-sentence, in a fence or in inline code: a quote
                        out.append(Event("reasoning", self._track(CALL_START)))
                        self.buf = self.buf[tool + len(CALL_START):]
                    continue
                if i < 0:
                    keep = self._hold(self.buf,
                                      ((THINK_END, CALL_START, FUNC_START) if self.recover else (THINK_END, CALL_START))
                                      if self.schemas else (THINK_END,))
                    if len(self.buf) > keep:
                        out.append(Event("reasoning", self._track(self.buf[:len(self.buf) - keep])))
                        self.buf = self.buf[len(self.buf) - keep:]
                    return out
                quoted = self._quoted_think_end(i)
                if quoted is None:                   # held like a partial tag until the character after it is here
                    if i:
                        out.append(Event("reasoning", self._track(self.buf[:i])))
                        self.buf = self.buf[i:]
                    return out
                if quoted:                           # #537: a quoted `</think>` is reasoning text
                    out.append(Event("reasoning", self._track(self.buf[:i + len(THINK_END)])))
                    self.buf = self.buf[i + len(THINK_END):]
                    continue
                if i:
                    out.append(Event("reasoning", self._track(self.buf[:i])))
                self.buf = self.buf[i + len(THINK_END):]
                self.state, self.lead, self.after_think = "content", True, True
                self._reset_track()                               # the answer's own text starts here
            elif self.state == "content":
                if self.lead:                                   # newlines right after </think> or a call
                    stripped = self.buf.lstrip("\n")
                    if not stripped:
                        self.buf = ""
                        return out
                    self.buf, self.lead = stripped, False
                i = self.buf.find(CALL_START)
                stray = self._stray_think_at() if self.after_think else -1
                if stray >= 0 and (i < 0 or stray < i):
                    # #4077: the reasoning ended at a `</think>` the model wrote while thinking (quoting the format on a
                    # line of its own), and this is the one that ends it.  What came between was reasoning, already sent
                    # as the answer; the tag is dropped and the text after it starts over, so a call that follows is
                    # not taken for a quote by the code fence or backtick the reasoning left open.
                    if stray:
                        out.append(Event("content", self._track(self.buf[:stray])))
                    self.buf = self.buf[stray + len(THINK_END):]
                    self._reset_track()
                    self.lead = True
                    self.stray_think += 1
                    continue
                b, decided = self._bare_opener() if self.recover and self.schemas else (-1, False)
                if b >= 0 and (i < 0 or b < i):
                    # recover: <function=NAME> alone at the start of a line - the call without its <tool_call>.
                    # Until the name has arrived it is held.
                    j = b
                    while j > 0 and self.buf[j - 1] in "\n \t":     # the line break and an indent before the opener
                        j -= 1
                    if j > 0 and self.buf[:j].strip():
                        out.append(Event("content", self._track(self.buf[:j])))
                    if not decided:
                        self.buf = self.buf[j:]
                        return out
                    self.buf = self.buf[b:]
                    self.state, self.bare = "call", True
                    continue
                dialect = self._dialect_opener() if self.recover and self.schemas else None
                if dialect is not None and (i < 0 or dialect[0] < i):
                    # #6022 (recover): Claude's dialect without the template's wrapper - `<function_calls>` around
                    # `<invoke>`, or a bare `<invoke name="T">` alone at the start of a line (a declared tool): the
                    # wrapper is put in front and the "invoke" state reads it
                    p, what = dialect
                    if what == "hold":
                        j = p
                        while j > 0 and self.buf[j - 1] == "\n":
                            j -= 1
                        if j > 0:
                            out.append(Event("content", self._track(self.buf[:j])))
                        self.buf = self.buf[j:]
                        return out
                    if what == "fc":
                        self.buf = self.buf[:p] + CALL_START + self.buf[p + len(FUNCTION_CALLS_START):]
                        q = self.buf.find(FUNCTION_CALLS_END, p)
                        if q >= 0:
                            self.buf = self.buf[:q] + CALL_END + self.buf[q + len(FUNCTION_CALLS_END):]
                    else:
                        self.buf = self.buf[:p] + CALL_START + "\n" + self.buf[p:]
                    continue
                if i < 0:
                    # Hold a partial tag AND the newlines before it: if a tool call follows, they are dropped,
                    # so emitting them early would make streamed and whole outputs differ.
                    tags: tuple[str, ...] = (CALL_START, THINK_END) if self.after_think else (CALL_START,)
                    if self.recover and self.schemas:
                        tags += (FUNCTION_CALLS_START, "<invoke ")
                    j = len(self.buf) - self._hold(self.buf, tags)
                    if self.recover and self.schemas:            # a line that may be becoming a bare <function=
                        k = self.buf.rfind("\n") + 1
                        tail = self.buf[k:]
                        opener = tail.lstrip(" \t")         # indented like _opener_ok allows: whole parsing takes it
                        if tail and FUNC_START.startswith(opener) and self._ok_at(k + len(tail) - len(opener)):
                            j = min(j, k)
                    while j > 0 and self.buf[j - 1] == "\n":
                        j -= 1
                    if j > 0:
                        out.append(Event("content", self._track(self.buf[:j])))
                        self.buf = self.buf[j:]
                    return out
                # #1058: an opener inside a code fence or inline code is text (a quoted example), never a call.  The
                # text before it decides, so streamed and whole outputs agree.  Mid-sentence openers still count.
                snap = self._snap()
                self._track(self.buf[:i])
                in_code = self._in_code()
                self._restore(snap)
                if in_code:
                    out.append(Event("content", self._track(self.buf[:i + len(CALL_START)])))
                    self.buf = self.buf[i + len(CALL_START):]
                    continue
                # A call is `<tool_call>` and then (after whitespace) `<function=`; the tag with anything else after
                # it is prose that names the format ("I'll use a <tool_call> block") - content, not a malformed call
                # that ends the request.  Until its follower has arrived it is held, like a partial tag.
                after = self.buf[i + len(CALL_START):].lstrip()
                if self.recover and self.schemas and after.startswith("<i"):
                    # #5969 (recover): Claude's `<invoke name="NAME">` dialect for a declared tool, its opener ending
                    # its line: held whole as the "invoke" state reads it (a cut one, an undeclared name, or prose that
                    # only names the tag - more text on the opener's line - stays the text it is)
                    m = _INVOKE_OPEN.match(after)
                    line_end = after[m.end():].lstrip(" \t")[:1] if m else ""
                    if m and not line_end:
                        return out                       # what follows the opener decides
                    if m and line_end == "\n" and self._callable(m.group(1)):
                        if i and self.buf[:i].strip():
                            out.append(Event("content", self._track(self.buf[:i].rstrip("\n"))))
                        self.buf = self.buf[i + len(CALL_START):]
                        self.state = "invoke"
                        continue
                    if not m and _INVOKE_PART.match(after) and "\n" not in after:
                        j = i
                        while j > 0 and self.buf[j - 1] == "\n":
                            j -= 1
                        if j > 0:
                            out.append(Event("content", self._track(self.buf[:j])))
                            self.buf = self.buf[j:]
                        return out
                if self.schemas and not after.startswith(FUNC_START):
                    loose = self._loose_open(after)
                    if loose == "wait":                  # may still become one: held like a partial tag
                        j = i
                        while j > 0 and self.buf[j - 1] == "\n":
                            j -= 1
                        if j > 0:
                            out.append(Event("content", self._track(self.buf[:j])))
                            self.buf = self.buf[j:]
                        return out
                    if loose:                            # #3441: `<=NAME>` etc. is `<function=NAME>`
                        at = len(self.buf) - len(after)
                        self.buf = self.buf[:at] + FUNC_START + loose[0] + ">" + after[loose[1]:]
                        continue
                kind = self._follower(after) if self.recover else None
                if (after.startswith("<") and kind is None and not FUNC_START.startswith(after)
                        and not after.startswith(FUNC_START)):
                    self.unread += 1
                if kind in ("param", "json", "missing"):
                    if i and self.buf[:i].strip():
                        out.append(Event("content", self._track(self.buf[:i].rstrip("\n"))))
                    rest = self.buf[i + len(CALL_START):]
                    if kind == "param":                  # <parameter=NAME> as the opener: the call's <function=NAME>
                        at = len(rest) - len(after)
                        rest = rest[:at] + FUNC_START + after[len(PARAM_START):]
                    elif kind == "missing":              # `function=NAME>`: the opening angle bracket was dropped
                        match = _MISSING_FUNC_OPEN.match(after)
                        assert match is not None         # the kind says it matched
                        at = len(rest) - len(after)
                        rest = rest[:at] + FUNC_START + match.group(1) + ">" + after[match.end():]
                    self.buf = rest
                    self.state = "json" if kind == "json" else "call"
                    continue
                if kind == "wait":
                    after = ""
                cut = re.match(r"<function=([^<>\s]+)[ \t]*\n", after) if self.recover else None
                if cut and self.aliases.get(cut.group(1), cut.group(1)) in self.schemas:
                    # #6032 R6 (recover): `<function=NAME` + line break - the `>` left out - of a declared tool
                    at = len(self.buf) - len(after)
                    self.buf = self.buf[:at] + FUNC_START + cut.group(1) + ">\n" + after[cut.end():]
                    continue
                is_call = re.match(r"<function=(\s*[^<>\s]+\s*)>", after)
                if is_call:
                    raw_name = is_call.group(1)
                    name = raw_name.strip()
                    name = self.aliases.get(name, name)
                    if raw_name != raw_name.strip() and name not in self.schemas:
                        is_call = None     # only declared names gain whitespace normalization
                is_partial = FUNC_START.startswith(after) or re.fullmatch(r"<function=\s*[^<>\s]*\s*", after)
                if not is_call and not is_partial:
                    out.append(Event("content", self._track(self.buf[:i + len(CALL_START)])))
                    self.buf = self.buf[i + len(CALL_START):]
                    continue
                if not is_call:
                    j = i
                    while j > 0 and self.buf[j - 1] == "\n":
                        j -= 1
                    if j > 0:
                        out.append(Event("content", self._track(self.buf[:j])))
                        self.buf = self.buf[j:]
                    return out
                if self.schemas or self.declared_only:
                    # #3075: a call counts only for a declared tool; the name must be complete to tell
                    gt = after.find(">")
                    nl = after.find("\n")
                    if gt < 0 and nl < 0:
                        j = i
                        while j > 0 and self.buf[j - 1] == "\n":
                            j -= 1
                        if j > 0:
                            out.append(Event("content", self._track(self.buf[:j])))
                            self.buf = self.buf[j:]
                        return out
                    if gt >= 0 and (nl < 0 or gt < nl) and not self._callable(after[len(FUNC_START):gt].strip()):
                        # logged by the server: otherwise "the model ignored my tool" is the only clue
                        self.unread += 1
                        if len(self.unread_names) < 5:
                            self.unread_names.append(after[len(FUNC_START):gt].strip()[:80])
                        out.append(Event("content", self._track(self.buf[:i + len(CALL_START)])))
                        self.buf = self.buf[i + len(CALL_START):]
                        continue
                if i and self.buf[:i].strip():
                    out.append(Event("content", self._track(self.buf[:i].rstrip("\n"))))
                self.buf = self.buf[i + len(CALL_START):]
                self.state = "call"
            elif self.state == "json":
                # recover: `{"name": ..., "arguments": {...}}` inside the wrapper is the call once its </tool_call> has
                # arrived and it names a declared tool; anything else is the text it is
                i = self.buf.find(CALL_END)
                if i < 0:
                    return out
                call = json_tool_call(self.buf[:i].strip(), self.schemas)
                if call is None:
                    out.append(Event("content", self._track(CALL_START + self.buf[:i + len(CALL_END)])))
                else:
                    out.append(Event("tool_call", call=call))
                self.buf = self.buf[i + len(CALL_END):]
                self.state, self.lead = "content", call is not None
            elif self.state == "invoke":
                # #5969 (recover): `<invoke name="T">` + `<parameter name="K">V</parameter>` ... `</invoke>`, its
                # `</tool_call>` after it or left out at the end of the output: read in the template's form; a body that
                # does not read, or one the output ends inside, is the text it is
                m = _INVOKE_END.search(self.buf)
                if m is None:
                    return out                          # not whole yet; at the end, finish() returns it as text
                rest = self.buf[m.end():]
                tail = rest.lstrip()
                if not tail and not self._final:
                    return out                          # its `</tool_call>` may still come
                if tail and (CALL_END.startswith(tail) or FUNCTION_CALLS_END.startswith(tail)) and not self._final:
                    return out
                # #6022: `</function_calls>` too
                closer = CALL_END if tail.startswith(CALL_END) \
                    else FUNCTION_CALLS_END if tail.startswith(FUNCTION_CALLS_END) else None
                drop = len(rest) - len(tail) + len(closer) if closer else 0
                raw = self.buf[:m.end()]
                call = None
                try:
                    opened = _INVOKE_OPEN.match(raw.lstrip())
                    if opened is None:
                        raise ValueError("no invoke opener")
                    name = opened.group(1)
                    name = self.aliases.get(name, name)
                    call = parse_tool_call(invoke_to_template(raw), self.schemas.get(name), self.notes, recover=True)
                    call.name = name
                except (ValueError, AttributeError):
                    call = None
                if call is None:
                    out.append(Event("content", self._track(CALL_START + raw + (CALL_END if drop else ""))))
                else:
                    out.append(Event("tool_call", call=call))
                self.buf = self.buf[m.end() + drop:]
                self.state, self.lead = "content", call is not None
            else:
                drop = None
                if self.bare:                       # no wrapper: the call ends at its </function>; a stray
                    i = self._function_end()        # </tool_call> after it is the wrapper's closer, dropped
                    if i >= 0:
                        after = self.buf[i:].lstrip()
                        if after.startswith(CALL_END):
                            drop = len(self.buf[i:]) - len(after) + len(CALL_END)
                        elif after and not CALL_END.startswith(after):
                            drop = 0
                else:
                    i = self._wrapped_call_end()
                    if i >= 0:
                        drop = len(CALL_END)
                    # Only a following call opener confirms an omitted wrapper closer.
                    f = self._function_end()
                    nxt = self.buf[f:].lstrip() if f >= 0 else ""
                    if re.match(r"<tool_call>\s*<function=\s*[^<>\s]+\s*>", nxt):
                        i, drop = f, 0
                    if self.recover:                # a batch in one wrapper: the next call follows </function>
                        if nxt.startswith(PARAM_START) and self._follower(nxt) == "param":
                            at = len(self.buf) - len(nxt)
                            self.buf = self.buf[:at] + FUNC_START + nxt[len(PARAM_START):]
                            nxt = self.buf[at:]
                        if nxt.startswith(FUNC_START):
                            i, drop, self.batch = f, 0, True
                if self.stream_tools:
                    if i >= 0:
                        whole, self.buf = self.buf, self.buf[:i]     # scan only the body
                        # #5040: its end is known, so no closing tag is still arriving
                        final, self._final = self._final, True
                        out += self._scan()
                        out += self._close_scan()
                        self.buf, self._final = whole, final
                    else:
                        out += self._scan()
                if i < 0 or drop is None:
                    if self.stream_tools:
                        self._compact_scan_body()
                    return out
                body = self.buf[:i]
                self.buf = self.buf[i + drop:]
                batch, self.batch = self.batch, False
                out += self._finish_call(body)
                if batch and out[-1].kind == "tool_call":
                    self.state, self.lead = "call", False

    def finish(self, reason: str | None = None) -> list[Event]:
        """End of generation: flush whatever is held (an unterminated tool call is returned as content; one that was
        already announced stays unfinished: its JSON is not closed and no "tool_call" follows it, #211).  `reason` is
        how the turn ended: calls waiting from the reasoning become real calls only on a natural stop (None = stop);
        a turn cut by max tokens (or cancelled, or failed) keeps them as reasoning text (#1058)."""
        out = []
        self._restore_scan_body()               # upstream #4686: the call body moved out while it streamed comes back
        if self._held:
            self.buf, self._held = self.buf + self._held, ""
        if self.buf and self.state in ("call", "rcall", "invoke"):
            # #4701: a call held after its closing tags (could a file's text go on?) is whole now
            self._final = True
            out += self.feed("")
        self._restore_scan_body()               # the final feed may compact a still-incomplete streamed call again
        if self.state == "rcall" and reason in (None, "stop"):
            # #4427: a call in the reasoning whose `</function>` is the end of the output (the `</tool_call>` left out)
            # is as whole as one in the answer: it waits for the turn's end like the others
            body = self.buf[0 if self.rbare else len(CALL_START):]
            f = function_end(body)
            name = body.lstrip()[len(FUNC_START):].split(">", 1)[0].strip() if f >= 0 else ""
            name = self.aliases.get(name, name)
            if f >= 0 and not body[f:].strip() and self._callable(name):
                try:
                    call = parse_tool_call(body[:f], self.schemas.get(name), self.notes, recover=self.recover)
                    call.name = name
                    self.pending.append([self.buf, call])
                    self.buf, self.state, self.rbare = "", "reasoning", False
                except ValueError:
                    pass
        if self.pending:
            out += self._release(reason in (None, "stop") and self.state == "reasoning" and not self.buf)
        if self.state == "call" and self.bare and function_end(self.buf) >= 0:
            i = function_end(self.buf)          # recover: a bare call is whole at its </function>
            body, rest = self.buf[:i], self.buf[i:].strip()
            self.buf = ""
            out += self._finish_call(body)
            if rest and not CALL_END.startswith(rest):
                out.append(Event("content", self._track(rest)))
            return out
        if self.state == "call" and self.stream_tools and self.scall is not None:
            out += self._scan()                 # the output ended inside a call that was already announced
            if self.ss == "function_done" and not self.buf[self.sp:].strip() and reason in (None, "stop"):
                # A natural end immediately after </function> retains native streaming recovery.
                out += self._close_scan()
                announced = self.scall
                final: ToolCall | None = announced
                try:            # the final event carries the call's arguments, as the one _finish_call sends does
                    read = parse_tool_call(self.buf, self.schemas.get(announced.name), self.notes, recover=self.recover)
                    read.name, read.id = announced.name, announced.id
                    final = read
                except ValueError as e:
                    if "no readable parameter" in str(e):
                        final = None            # #6032 R5: never run with no arguments (it stays announced, unfinished)
                if final is not None:
                    out.append(Event("tool_call", call=final))
            elif self.ss == "function_done":
                out.append(Event("content", self._track(CALL_START + self.buf)))
            self.buf = ""
            self._reset_scan()
            return out
        if self.state == "reasoning" and self.buf == THINK_END:   # the output ends on a held `</think>`: its end
            self.buf, self.state = "", "content"
        if self.buf:
            # an unfinished call inside the reasoning (#804) is reasoning text, never a call
            kind = {"reasoning": "reasoning", "rcall": "reasoning", "content": "content"}.get(self.state, "content")
            text = CALL_START + self.buf if self.state in ("call", "json", "invoke") and not self.bare else self.buf
            out.append(Event(kind, text))
            self.buf = ""
        return out

    def _finish_call(self, body: str) -> list[Event]:
        """A whole call body in the template's form -> its tool_call event.  With recover, one that still does not
        parse is the text it is instead of an error that ends the request."""
        raw_name = body.strip()[len(FUNC_START):].split(">", 1)[0]
        name = raw_name.strip()
        name = self.aliases.get(name, name)
        bare, self.bare = self.bare, False
        try:
            if raw_name != raw_name.strip() and name not in self.schemas:
                raise ValueError("whitespace tool name is not declared")
            call = parse_tool_call(body, self.schemas.get(name), self.notes, recover=self.recover)
            call.name = name
        except ValueError:
            if not self.recover:
                raise
            self._reset_scan()
            self.state = "content"
            return [Event("content", self._track(body if bare else CALL_START + body + CALL_END))]
        if self.scall is not None:
            call.id = self.scall.id
        self._reset_scan()
        self.state, self.lead = "content", True
        return [Event("tool_call", call=call)]


class GlmOutputParser(OutputParser):
    """OutputParser for GLM's call form: the output rewritten as it streams (GlmCallRewriter) before it is parsed."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.rewriter = GlmCallRewriter()

    def feed(self, delta: str) -> list[Event]:
        return super().feed(self.rewriter.feed(delta))

    def finish(self, reason: str | None = None) -> list[Event]:
        return super().feed(self.rewriter.flush()) + super().finish(reason)


class Inkling2MiniOutputParser(OutputParser):
    """OutputParser for Inkling-2-Mini (upstream's InklingOutputParser): the model writes its text and its calls in
    channels (`<|content_text|>`, `<|content_invoke_tool_json|>` ...), which serve/inkling2mini_dialect.py turns into
    text and `<tool_call>` frames with a JSON body as the output arrives.  It starts in the content state, for the model
    opens a channel for whatever it writes, and reads the JSON body whatever the server's tool_call_recovery says: that
    body is the dialect's own form here, not a model that drifted."""

    def __init__(self, thinking: bool = True, tools: list[dict] | None = None, stream_tools: bool = False,
                 recover: bool = False, declared_only: bool = False, deferred: bool = False):
        super().__init__(thinking=False, tools=tools, stream_tools=stream_tools, recover=True,
                         declared_only=declared_only, deferred=deferred)
        from serve import inkling2mini_dialect
        self._dialect = inkling2mini_dialect.Inkling2MiniDialect()

    def _dispatch(self, chunks: list[tuple[str, str]]) -> list[Event]:
        out: list[Event] = []
        for kind, chunk in chunks:
            if kind == "reasoning":
                out.append(Event("reasoning", chunk))
            elif chunk:
                out.extend(super().feed(chunk))
        return out

    def feed(self, delta: str) -> list[Event]:
        return self._dispatch(self._dialect.feed_channels(delta))

    def finish(self, reason: str | None = None) -> list[Event]:
        out = self._dispatch(self._dialect.feed_channels("", final=True))
        return out + super().finish(reason)
