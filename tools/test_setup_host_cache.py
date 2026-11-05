# SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
# SPDX-License-Identifier: LGPL-3.0-or-later
"""XeStrata-890M: setup's rule for the expert cache in host USM (uma_host_cache), chosen from what the device reports -
an integrated AMD GPU whose carve-out does not hold the experts, the RAM holding the model, one GPU, no RAM budget -
and the Japanese draft subset (--draft-vocab ja).  Pure functions: no GPU, no downloads, no prompts.

    python -m unittest tools.test_setup_host_cache
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import setup  # noqa: E402

# what setup's gpu_info reports for the Radeon 890M (gfx1150: a 512 MB carve-out, the GTT about half of the RAM)
APU = {"name": "AMD Radeon 880M / 890M (gfx1150)", "vendor": "amd", "integrated": True, "vram_gb": 0.0,
       "carve_gb": 0.5, "shared_gb": 31.0, "uma_gb": 31.5, "gfx": "gfx1150"}
CARD = {"name": "AMD Radeon RX 9060 XT (gfx1200)", "vendor": "amd", "integrated": False, "vram_gb": 16.0}
INTEL_IGPU = {"name": "Intel UHD Graphics 770", "vendor": "intel", "integrated": True, "vram_gb": 0.0}


class HostCacheRule(unittest.TestCase):
    def test_the_890m_with_the_ram_for_the_model(self):
        self.assertTrue(setup.uma_host_cache(APU, "IQ2_XS", False, None, False))
        self.assertTrue(setup.uma_host_cache(APU, "IQ3_XXS", False, None, False))

    def test_not_in_the_low_ram_mode(self):
        self.assertFalse(setup.uma_host_cache(APU, "IQ2_XS", True, None, False))

    def test_not_for_a_ram_budget_model(self):
        self.assertFalse(setup.uma_host_cache(APU, "UD-IQ4_XS", False, 30.0, False))

    def test_not_with_a_layer_split(self):
        self.assertFalse(setup.uma_host_cache(APU, "IQ2_XS", False, None, True))

    def test_not_for_a_discrete_card_or_another_makers_integrated_gpu(self):
        self.assertFalse(setup.uma_host_cache(CARD, "IQ2_XS", False, None, False))
        self.assertFalse(setup.uma_host_cache(INTEL_IGPU, "IQ2_XS", False, None, False))

    def test_not_when_the_carve_out_holds_the_experts(self):
        # a Strix Halo with a 64 GB carve-out: the GPU's own pool holds IQ2_XS's experts, the default path is kept
        halo = dict(APU, name="AMD Radeon 8060S (gfx1151)", gfx="gfx1151", carve_gb=64.0, shared_gb=32.0, uma_gb=96.0)
        self.assertFalse(setup.uma_host_cache(halo, "IQ2_XS", False, None, False))
        # a carve-out smaller than the experts: the host cache, as on the 890M
        self.assertTrue(setup.uma_host_cache(dict(halo, carve_gb=16.0), "IQ2_XS", False, None, False))

    def test_a_gpu_record_without_a_carve_out_counts_as_none(self):
        self.assertTrue(setup.uma_host_cache({"vendor": "amd", "integrated": True}, "IQ2_XS", False, None, False))


class V31SetsOnUnifiedMemory(unittest.TestCase):
    """upstream 86ffdbeb1 (uma_holds_experts) beside the fork's host-cache rule: on the 890M a v3.1 Unsloth set whose
    experts the RAM holds gets no RAM budget, and then uma_host_cache puts every expert into host USM; a set too big for
    the RAM keeps the budget plan (no host cache); the sets measured under the budget mode are never touched."""
    RAM = 62.6      # ram_gb() on a 64 GB PC
    FITS = ("UD-IQ1_M", "UD-Q2_0", "UD-IQ2_XS", "UD-Q2_K_XL")
    TOO_BIG = ("UD-IQ3_XXS", "UD-Q3_K_XL", "UD-IQ4_XS-v31")

    def test_the_sets_the_ram_holds_take_the_host_cache_path(self):
        for m in self.FITS:
            with self.subTest(model=m):
                self.assertTrue(setup.MODELS[m].get("uma_pool_auto"))
                self.assertTrue(setup.uma_holds_experts(m, self.RAM, APU))
                self.assertGreaterEqual(self.RAM, setup.MODELS[m]["arena_gb"] / 1.073741824
                                        + setup.UNSLOTH_RESIDENT_BESIDE_GB)
                self.assertTrue(setup.uma_host_cache(APU, m, False, None, False))   # budget None: the host cache wins

    def test_the_sets_too_big_for_the_ram_keep_the_budget(self):
        for m in self.TOO_BIG:
            with self.subTest(model=m):
                self.assertFalse(setup.uma_holds_experts(m, self.RAM, APU))
                budget = setup.resident_budget_gib(m, self.RAM)
                self.assertFalse(setup.uma_host_cache(APU, m, False, budget, False))
        self.assertTrue(setup.uma_holds_experts("UD-Q3_K_XL", 96.0, APU))        # with the RAM for it

    def test_the_measured_sets_keep_their_plans(self):
        for m in ("UD-IQ4_XS", "UD-Q4_K_XL", "UD-Q5_K_XL", "UD-Q6_K_XL", "Q8_0"):
            self.assertNotIn("uma_pool_auto", setup.MODELS[m])
            self.assertFalse(setup.uma_holds_experts(m, 124.0, APU), m)

    def test_a_large_carve_out_holds_them_in_the_pool_and_keeps_the_default_path(self):
        # a Strix Halo with a 64 GB carve-out (upstream's case): the pool, carve-out + shared memory, holds UD-IQ2_XS
        # beside the fixed part; no budget, and no host cache either (the carve-out holds the experts)
        halo = dict(APU, name="AMD Radeon 8060S (gfx1151)", gfx="gfx1151", carve_gb=64.0, shared_gb=32.0, uma_gb=96.0)
        self.assertTrue(setup.uma_holds_experts("UD-IQ2_XS", self.RAM, halo))
        self.assertFalse(setup.uma_host_cache(halo, "UD-IQ2_XS", False, None, False))
        small = dict(halo, carve_gb=48.0, shared_gb=8.0, uma_gb=56.0)      # a pool too small for the experts + 16.5
        self.assertFalse(setup.uma_holds_experts("UD-IQ2_XS", self.RAM, small))

    def test_not_for_a_card_or_another_makers_integrated_gpu(self):
        for gpu in (CARD, INTEL_IGPU, {"vram_gb": 12.0}, None):
            self.assertFalse(setup.uma_holds_experts("UD-IQ2_XS", self.RAM, gpu))

    def test_the_new_sets_are_after_the_forks_and_never_the_default(self):
        names = [m for m in setup.MODELS if "unsloth" in setup.MODELS[m].get("families", ())]
        self.assertEqual(names[:5], ["UD-IQ4_XS", "UD-Q4_K_XL", "UD-Q5_K_XL", "UD-Q6_K_XL", "Q8_0"])
        self.assertEqual(sorted(names[5:]), sorted(self.FITS + self.TOO_BIG))
        for m in names[5:]:
            self.assertTrue(setup.MODELS[m].get("experimental"), m)        # sorted after the regular sizes in the menu
            self.assertNotIn("huihui-unsloth", setup.MODELS[m]["families"])


class JapaneseDraftSubset(unittest.TestCase):
    def test_the_subset_is_a_choice_and_shipped(self):
        self.assertEqual(setup.DRAFT_VOCABS["ja"], "draft_vocab_ja.bin")
        self.assertTrue((ROOT / "data" / setup.DRAFT_VOCABS["ja"]).exists())
        self.assertIn("ja", setup.DRAFT_VOCAB_MIB)

    def test_the_default_stays_upstreams(self):
        self.assertEqual(setup.DRAFT_VOCABS.get("cjk"), "draft_vocab.bin")
        self.assertEqual(next(iter(setup.DRAFT_VOCABS)), "cjk")


class KfdIntegrated(unittest.TestCase):
    """The KFD node's signs of a processor's own graphics (kfd_integrated)."""

    def test_the_890m_node_has_no_local_memory_and_no_cpu_cores(self):
        # Linux 7.2 on the Ryzen AI 9 HX 370: the CPU cores are a node of their own
        self.assertTrue(setup.kfd_integrated({"cpu_cores_count": "0", "simd_count": "32", "local_mem_size": "0",
                                              "gfx_target_version": "110500"}))

    def test_a_node_with_cpu_cores_as_upstream_reads_it(self):
        self.assertTrue(setup.kfd_integrated({"cpu_cores_count": "16", "simd_count": "80", "local_mem_size": "0"}))

    def test_a_card_has_local_memory(self):
        self.assertFalse(setup.kfd_integrated({"cpu_cores_count": "0", "simd_count": "64",
                                               "local_mem_size": str(16 * 2**30)}))

    def test_without_the_property_nothing_is_inferred(self):
        self.assertFalse(setup.kfd_integrated({"cpu_cores_count": "0", "simd_count": "64"}))
        self.assertFalse(setup.kfd_integrated({"cpu_cores_count": "x", "local_mem_size": "0"}))


if __name__ == "__main__":
    unittest.main()
