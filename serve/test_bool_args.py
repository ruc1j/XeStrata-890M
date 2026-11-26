# SPDX-FileCopyrightText: 2026 Niko1221 and the Strata contributors
# SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
# SPDX-License-Identifier: LGPL-3.0-or-later AND MIT
"""upstream #5979: a declared boolean parameter came back as the string "True" / "False" (the model's Python spelling),
not as true / false.  The types declared through anyOf / oneOf are read too (#2371)."""

import unittest

from serve.frontend import OutputParser, param_value, parse_tool_call

TOOL = {"name": "write_note", "parameters": {"type": "object", "properties": {
    "formatted": {"type": "boolean"}, "title": {"type": "string"}, "pinned": {"type": ["boolean", "null"]},
    "label": {"type": ["boolean", "string"]}}}}
UNION = {"name": "set_flags", "parameters": {"type": "object", "properties": {
    "any_bool": {"anyOf": [{"type": "boolean"}, {"type": "null"}]},
    "one_bool": {"oneOf": [{"type": "boolean"}, {"type": "null"}]},
    "nested": {"anyOf": [{"oneOf": [{"type": "integer"}, {"type": "boolean"}]}, {"type": "null"}]},
    "text_or_bool": {"anyOf": [{"type": "boolean"}, {"type": "string"}]},
    "count": {"oneOf": [{"type": "integer"}, {"type": "null"}]}}}}


class BooleanArguments(unittest.TestCase):
    def test_the_python_spelling_of_a_declared_boolean_is_the_boolean(self):
        for text, want in (("True", True), ("False", False), ("true", True), ("false", False), (" TRUE ", True)):
            self.assertIs(param_value(text, {"type": "boolean"}), want, text)
        self.assertIs(param_value("False", {"type": ["boolean", "null"]}), False)

    def test_strings_and_other_words_are_left_as_they_were(self):
        self.assertEqual(param_value("True", {"type": "string"}), "True")
        self.assertEqual(param_value("True", {"type": ["boolean", "string"]}), "True",
                         "a string is allowed: the text stays text")
        self.assertEqual(param_value("True", {"type": "boolean", "enum": ["True"]}), "True", "an enum of the text")
        self.assertEqual(param_value("yes", {"type": "boolean"}), "yes")
        self.assertEqual(param_value("True", None), "True", "no schema: as before")

    def test_a_whole_call_and_the_streamed_parser_give_booleans(self):
        body = ("<function=write_note>\n<parameter=formatted>\nTrue\n</parameter>\n<parameter=title>\nTrue\n"
                "</parameter>\n</function>")
        self.assertEqual(parse_tool_call(body, TOOL).arguments, {"formatted": True, "title": "True"})
        parser = OutputParser(thinking=False, tools=[TOOL], stream_tools=True)
        events = parser.feed("<tool_call>\n" + body + "\n</tool_call>") + parser.finish()
        calls = [e for e in events if e.kind == "tool_call"]
        self.assertEqual(len(calls), 1, events)
        self.assertEqual(calls[0].call.arguments["formatted"], True)
        self.assertEqual(calls[0].call.arguments["title"], "True")


class UnionTypes(unittest.TestCase):
    def test_types_declared_through_anyof_oneof_are_read(self):
        props = UNION["parameters"]["properties"]
        self.assertIs(param_value("True", props["any_bool"]), True)
        self.assertIs(param_value("false", props["one_bool"]), False)
        self.assertIs(param_value("True", props["nested"]), True)
        self.assertEqual(param_value("True", props["text_or_bool"]), "True")
        self.assertEqual(param_value("42", props["count"]), 42)

    def test_a_whole_call_with_union_types(self):
        body = ("<function=set_flags>\n<parameter=any_bool>\nTrue\n</parameter>\n"
                "<parameter=one_bool>\nfalse\n</parameter>\n<parameter=count>\n42\n</parameter>\n</function>")
        self.assertEqual(parse_tool_call(body, UNION).arguments,
                         {"any_bool": True, "one_bool": False, "count": 42})


if __name__ == "__main__":
    unittest.main()
