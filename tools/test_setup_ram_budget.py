# SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
# SPDX-License-Identifier: LGPL-3.0-or-later
"""The RAM budget rule of setup (resident_budget_gib): the RAM less 17 GB, the reading rounded up to the size the PC is
sold with; a GPU that shares the RAM keeps the reading less 24 GB (upstream 8780e9fdc and c8d70bbcd)."""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import setup  # noqa: E402

MODEL = "UD-Q4_K_XL"


class RamBudget(unittest.TestCase):
    def test_a_discrete_card_leaves_17_gb(self):
        whole = int(setup.MODELS[MODEL]["arena_gb"] / 1.073741824)
        for ram, want in ((32, 15), (48, 31), (64, 47), (96, 79), (128, 111)):
            with self.subTest(ram=ram):
                self.assertEqual(setup.resident_budget_gib(MODEL, ram), min(want, whole))

    def test_a_small_pc_keeps_at_least_8(self):
        self.assertEqual(setup.resident_budget_gib(MODEL, 16), 8)

    def test_a_kv_cache_in_ram_comes_out_of_it(self):
        self.assertEqual(setup.resident_budget_gib(MODEL, 64, kv_ram_gb=1.7), 45)

    def test_the_reading_counts_as_the_size_the_pc_is_sold_with(self):
        self.assertEqual(setup.nominal_ram_gb(62.5), 64.0)      # Linux's MemTotal on a 64 GB PC
        self.assertEqual(setup.nominal_ram_gb(63.1), 64.0)      # Windows on the same PC
        self.assertEqual(setup.nominal_ram_gb(60.0), 60.0)      # a PC that really has less: 3% under 64 is 62.1
        self.assertEqual(setup.nominal_ram_gb(31.2), 32.0)
        self.assertEqual(setup.nominal_ram_gb(270.0), 272.0)    # above 256: a multiple of 8
        self.assertEqual(setup.resident_budget_gib(MODEL, 62.5), setup.resident_budget_gib(MODEL, 64))
        self.assertEqual(setup.resident_budget_gib(MODEL, 60.0), 43)

    def test_a_gpu_that_shares_the_ram_keeps_24_gb_and_the_reading(self):
        self.assertEqual(setup.resident_budget_gib(MODEL, 64, uma=True), 40)
        self.assertEqual(setup.resident_budget_gib(MODEL, 62.4, uma=True), 38)
        self.assertEqual(setup.resident_budget_gib(MODEL, 64, kv_ram_gb=1.7, uma=True), 38)
        self.assertTrue(setup.shares_ram({"integrated": True}))
        self.assertFalse(setup.shares_ram({"integrated": False}))
        self.assertFalse(setup.shares_ram(None))

    def test_the_note_names_the_room_the_rule_left(self):
        said = []
        warn, setup.warn = setup.warn, said.append
        try:
            self.assertEqual(setup.budget_choice(MODEL, 64, 60), 60)
            self.assertEqual(setup.budget_choice(MODEL, 64, 60, uma=True), 60)
            self.assertEqual(setup.budget_choice(MODEL, 64, None), 47)
            self.assertEqual(setup.budget_choice(MODEL, 64, None, uma=True), 40)
        finally:
            setup.warn = warn
        self.assertEqual(len(said), 2)
        self.assertIn("47 GiB: the RAM less 17 GB", said[0])
        self.assertIn("40 GiB: the RAM less 24 GB", said[1])


if __name__ == "__main__":
    unittest.main()
