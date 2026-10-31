# SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
# SPDX-License-Identifier: LGPL-3.0-or-later
"""serve/runconfig.py: a number that is not finite is refused (json reads Infinity and NaN as floats; a NaN top_p
let through would be saved as a `NaN` literal and stop the next start)."""
import unittest

from serve import runconfig


class NonFiniteNumbers(unittest.TestCase):
    def test_infinite_and_nan_values_are_value_errors(self):
        for key, v in (("sampling.top_p", float("nan")), ("sampling.top_p", float("inf")),
                       ("sampling.temperature", float("-inf")), ("reasoning_budget_tokens", float("inf")),
                       ("sampling.top_k", float("nan"))):
            with self.assertRaises(ValueError, msg=(key, v)):
                runconfig.check(key, v, {})

    def test_finite_values_still_pass(self):
        self.assertEqual(runconfig.check("sampling.top_p", 0.9, {}), 0.9)
        self.assertEqual(runconfig.check("reasoning_budget_tokens", 7, {}), 7)


if __name__ == "__main__":
    unittest.main()
