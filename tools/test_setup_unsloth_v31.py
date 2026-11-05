# SPDX-FileCopyrightText: 2026 Niko1221 and the Strata contributors
# SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
# SPDX-License-Identifier: LGPL-3.0-or-later AND MIT
"""upstream 9ed3c7789 (0.2.11): Unsloth's v3.1 Qwen3.8-Flash-Next sets (revision d9620119, 2026-11-04) in setup.py's
catalog: UD-IQ1_M, UD-Q2_0, UD-IQ2_XS, UD-Q2_K_XL, UD-IQ3_XXS, UD-Q3_K_XL and the new UD-IQ4_XS (catalog name
UD-IQ4_XS-v31), the pinned UD-Q4_K_XL / UD-IQ4_XS / UD-Q5_K_XL / UD-Q6_K_XL / Q8_0 unchanged at 38bb39ee, UD-Q4_K_XL's
second (four-shard, n-gram table last) layout taken from local files, the manual-load rule (#444), and a header
regression for every set of the repository.  The tensor directories are upstream's tests/fixtures/unsloth_v31/
headers.json.gz (the first 4 MB of each shard, read with HTTP range requests; tools/unsloth_v31_fixture.py): the tests
that read it are skipped while it is not in the repository.  Mocked: no GPU, no downloads.  XeStrata's copy keeps the
pure checks of upstream's file: its setup.main() runs, the Mac app and strata_control have no counterpart here.

    python -m unittest tools.test_setup_unsloth_v31
"""
from __future__ import annotations

import re
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))
import setup  # noqa: E402
import strata_mcp  # noqa: E402
import unsloth_v31_fixture as FX  # noqa: E402

HAVE_FIXTURE = FX.FIXTURE.exists()
DOC = FX.load() if HAVE_FIXTURE else {"sets": {}}
needs_fixture = unittest.skipUnless(HAVE_FIXTURE, "upstream's tests/fixtures/unsloth_v31/headers.json.gz is not here")
REV31 = "d96201194f103c906994f283f504209dc8d2a10c"
REV_OLD = "38bb39ee97821de2c9009abb7e93950eec396e66"
NEW = ["UD-IQ1_M", "UD-Q2_0", "UD-IQ2_XS", "UD-Q2_K_XL", "UD-IQ3_XXS", "UD-Q3_K_XL", "UD-IQ4_XS-v31"]
HUB = {m: ("UD-IQ4_XS" if m == "UD-IQ4_XS-v31" else m) for m in NEW}     # the name on the Hub
FAM = setup.FAMILIES["unsloth"]
TABLE_BYTES = 28800138432             # per_layer_token_embd in IQ4_NL: the last shard of every set below Q5


def shards_of(set_name):
    return DOC["sets"][set_name]["shards"]


def pins_of(set_name):
    return {s["name"]: (s["bytes"], s["sha256"]) for s in shards_of(set_name)}


class Catalog(unittest.TestCase):
    def test_every_new_set_is_pinned(self):
        for model in NEW:
            with self.subTest(model=model):
                d = setup.MODELS[model]
                names = [setup.model_file(FAM, model, i) for i in range(1, d["shards"] + 1)]
                for n in names:
                    self.assertIn(n, setup.UNSLOTH_V31_SHARDS)
                    self.assertEqual(FAM["sha256"][n], setup.UNSLOTH_V31_SHARDS[n])
                    self.assertRegex(setup.UNSLOTH_V31_SHARDS[n][1], r"^[0-9a-f]{64}$")
                self.assertEqual(d["download_gb"], round(sum(setup.UNSLOTH_V31_SHARDS[n][0] for n in names) / 1e9, 1))
                self.assertEqual(d["revision"], REV31)
                self.assertEqual(d["quant"], HUB[model])
                self.assertEqual(d["families"], ("unsloth",))
                self.assertTrue(d["budget"] and d["vision"] is True and d["experimental"])
                self.assertEqual(d["ram_gb"], 48)                               # the RAM-budget floor of UD-Q4_K_XL
                self.assertEqual(setup.model_quant(model), HUB[model])
                self.assertEqual(setup.UNSLOTH_V31_SHARDS[names[-1]][0], TABLE_BYTES)   # the n-gram table, last

    @needs_fixture
    def test_every_new_set_is_pinned_by_the_hub_files(self):
        for model in NEW:
            with self.subTest(model=model):
                d, shards = setup.MODELS[model], shards_of(HUB[model])
                self.assertEqual(d["shards"], len(shards))
                names = [setup.model_file(FAM, model, i) for i in range(1, d["shards"] + 1)]
                self.assertEqual(names, [s["name"] for s in shards])
                for s in shards:
                    self.assertEqual(setup.UNSLOTH_V31_SHARDS[s["name"]], (s["bytes"], s["sha256"]))

    @needs_fixture
    def test_arena_is_the_routed_experts_bytes_of_the_headers(self):
        for model in NEW:
            with self.subTest(model=model):
                exp = 0
                for s in shards_of(HUB[model]):
                    for name, type_id, shape, _off in (s["header"] or {"tensors": []})["tensors"]:
                        if name.endswith("_exps.weight") and "shexp" not in name:
                            t = FX.G.TensorInfo(name, shape, type_id, FX.G.GGML_TYPES[type_id], 0)
                            exp += t.expected_bytes()
                self.assertEqual(setup.MODELS[model]["arena_gb"], round(exp / 1e9, 1))

    def test_the_pinned_sets_stay_at_their_commit(self):
        self.assertEqual(setup.HF_REVISIONS[setup.UNSLOTH_REPO], REV_OLD)
        self.assertEqual(setup.UNSLOTH_V31_REVISION, REV31)
        self.assertEqual(setup.hf(setup.UNSLOTH_REPO),
                         f"https://huggingface.co/unsloth/Qwen3.8-Flash-Next-GGUF/resolve/{REV_OLD}/")
        for model in ("UD-IQ4_XS", "UD-Q4_K_XL", "UD-Q5_K_XL", "UD-Q6_K_XL", "Q8_0"):
            self.assertNotIn("revision", setup.MODELS[model], model)
            self.assertEqual(setup.model_hf(FAM, model),
                             f"https://huggingface.co/unsloth/Qwen3.8-Flash-Next-GGUF/resolve/{REV_OLD}/{model}/")
        self.assertEqual(setup.model_file(FAM, "UD-IQ4_XS", 3), "Qwen3.8-Flash-Next-UD-IQ4_XS-00003-of-00003.gguf")
        for model in NEW:
            self.assertEqual(setup.model_hf(FAM, model),
                             f"https://huggingface.co/unsloth/Qwen3.8-Flash-Next-GGUF/resolve/{REV31}/{HUB[model]}/")
        # huihui-ai's folders ("dirs") go through the same helper, at their own repository's commit
        hh = setup.FAMILIES["huihui-unsloth"]
        self.assertTrue(setup.model_hf(hh, "UD-IQ4_XS").endswith("/unsloth-UD-IQ4_XS/"))

    def test_the_old_pins_are_untouched(self):
        self.assertEqual(sum(b for b, _ in setup.UNSLOTH_SHARDS.values()), 111334654784)
        self.assertEqual(sum(b for b, _ in setup.UNSLOTH_IQ4_XS_SHARDS.values()), 93682584224)
        self.assertEqual(setup.UNSLOTH_IQ4_XS_SHARDS["Qwen3.8-Flash-Next-UD-IQ4_XS-00002-of-00003.gguf"][1],
                         "577a38a2392b40ca2193cea502e1d92f60b8cd370675d308e0ec21885d9daaa7")
        # no name is pinned twice with different bytes except UD-Q4_K_XL's, which are the alternative layout
        for n in setup.UNSLOTH_V31_SHARDS:
            self.assertNotIn(n, setup.UNSLOTH_SHARDS)
            self.assertNotIn(n, setup.UNSLOTH_IQ4_XS_SHARDS)
            self.assertNotIn(n, setup.UNSLOTH_BIG_SHARDS)

    @needs_fixture
    def test_the_unchanged_sets_have_the_pinned_shards_at_the_new_commit(self):
        # Q5 / Q6 / Q8 were not re-uploaded: the Hub's files at d9620119 are the pinned ones
        for set_name in ("UD-Q5_K_XL", "UD-Q6_K_XL", "Q8_0"):
            for n, pin in pins_of(set_name).items():
                self.assertEqual(setup.UNSLOTH_BIG_SHARDS[n], pin, n)
        # and the three-shard UD-IQ4_XS at the pinned commit is not the four-shard set at the new one
        self.assertTrue(set(pins_of("UD-IQ4_XS")).isdisjoint(setup.UNSLOTH_IQ4_XS_SHARDS))

    def test_a_file_is_pinned_by_the_commit_its_url_names(self):
        n = "Qwen3.8-Flash-Next-UD-IQ1_M-00002-of-00003.gguf"
        self.assertEqual(setup.file_pin(setup.model_hf(FAM, "UD-IQ1_M") + n), setup.UNSLOTH_V31_SHARDS[n])
        q4 = "Qwen3.8-Flash-Next-UD-Q4_K_XL-00003-of-00004.gguf"
        self.assertEqual(setup.file_pin(setup.model_hf(FAM, "UD-Q4_K_XL") + q4), setup.UNSLOTH_SHARDS[q4])
        self.assertEqual(setup.file_pin(setup.hf(setup.UNSLOTH_REPO).replace(REV_OLD, REV31) + "UD-Q4_K_XL/" + q4),
                         setup.UNSLOTH_Q4_V31_SHARDS[q4])
        self.assertNotEqual(setup.UNSLOTH_SHARDS[q4], setup.UNSLOTH_Q4_V31_SHARDS[q4])
        self.assertIsNone(setup.file_pin("https://huggingface.co/unsloth/Qwen3.8-Flash-Next-GGUF/resolve/" + REV31
                                         + "/UD-IQ1_S/x-00001-of-00003.gguf"))

    def test_the_model_names_resolve_in_both_directions(self):
        self.assertEqual(setup.FAMILIES["unsloth"]["tag"] + "UD-IQ4_XS-v31", "unsloth-UD-IQ4_XS-v31")
        with tempfile.TemporaryDirectory() as t:
            for stem, model in (("xestrata-unsloth-ud-iq4_xs-v31", "UD-IQ4_XS-v31"),
                                ("xestrata-unsloth-ud-iq4_xs", "UD-IQ4_XS"), ("xestrata-unsloth-ud-iq1_m", "UD-IQ1_M"),
                                ("xestrata-unsloth-ud-q3_k_xl", "UD-Q3_K_XL")):
                p = Path(t) / (stem + ".json")
                p.write_text('{"args": ["--max-context", "8192"]}')
                ch = setup.choices_from_config(p)
                self.assertEqual((ch["family"], ch["model"]), ("unsloth", model), stem)

    def test_the_approved_texts(self):
        want = {"UD-IQ1_M": "~1-bit (Unsloth Dynamic v3.1), the smallest download",
                "UD-Q2_0": "2-bit (Unsloth Dynamic v3.1)", "UD-IQ2_XS": "2-bit i-quant (Unsloth Dynamic v3.1)",
                "UD-Q2_K_XL": "2-bit (Unsloth Dynamic v3.1) with some 3-bit experts",
                "UD-IQ3_XXS": "3-bit i-quant (Unsloth Dynamic v3.1)",
                "UD-Q3_K_XL": "3-bit (Unsloth Dynamic v3.1) with some 4-bit experts",
                "UD-IQ4_XS-v31": "~4-bit i-quant (Unsloth Dynamic v3.1), a newer file than UD-IQ4_XS with other "
                                 "tensor types"}
        for m, a in want.items():
            self.assertTrue(setup.MODELS[m]["about"].startswith(a), m)         # XeStrata adds "not tested" after it
            self.assertEqual(strata_mcp.FALLBACK_MODELS[m]["about"], setup.MODELS[m]["about"])
        for m in ("UD-IQ1_M", "UD-Q2_0", "UD-IQ2_XS", "UD-Q2_K_XL", "UD-IQ3_XXS", "UD-Q3_K_XL", "UD-IQ4_XS",
                  "UD-Q4_K_XL", "UD-Q5_K_XL", "UD-Q6_K_XL", "Q8_0"):
            self.assertIn(m, setup.SUPPORTED_GGUFS)
        self.assertIn("(Unsloth's UD-IQ1_S, other K-quant files) cannot be used", setup.SUPPORTED_GGUFS)
        self.assertIn("UD-IQ1_S mixes two expert formats in one layer", setup.UNSUPPORTED_WHY["UD-IQ1_S"])
        self.assertEqual(FAM["by"], "Unsloth's 1- to 8-bit quantizations")


@needs_fixture
class LastShardHoldsTheTable(unittest.TestCase):
    """The n-gram table (per_layer_token_embd, IQ4_NL) is a shard of its own, the LAST one, in every set below Q5."""

    def test_the_table_is_the_last_shard_and_shard_1_holds_no_tensor(self):
        for set_name in ("UD-IQ1_M", "UD-IQ1_S", "UD-Q2_0", "UD-IQ2_XS", "UD-Q2_K_XL", "UD-IQ3_XXS", "UD-Q3_K_XL",
                         "UD-IQ4_XS", "UD-Q4_K_XL"):
            with self.subTest(set=set_name):
                shards = shards_of(set_name)
                last = shards[-1]
                self.assertEqual(last["bytes"], TABLE_BYTES)
                self.assertEqual([t[0] for t in last["header"]["tensors"]], ["per_layer_token_embd.weight"])
                self.assertEqual(last["header"]["tensors"][0][1], 20)                 # IQ4_NL
                self.assertEqual(last["header"]["tensors"][0][2], [160, 320001536])
                for s in shards[1:-1]:
                    self.assertNotIn("per_layer_token_embd.weight", [t[0] for t in s["header"]["tensors"]])
                self.assertIsNone(shards[0]["header"])              # metadata and tokenizer only (past the 4 MB read)

    def test_the_table_shard_is_the_same_bytes_in_each_layout(self):
        by_count = {}
        for set_name in ("UD-IQ1_M", "UD-Q2_0", "UD-IQ2_XS", "UD-Q2_K_XL", "UD-IQ3_XXS", "UD-Q3_K_XL", "UD-IQ4_XS",
                         "UD-Q4_K_XL"):
            shards = shards_of(set_name)
            by_count.setdefault(len(shards), set()).add(shards[-1]["sha256"])
        # one file for the 3-shard sets, one for the 4-shard sets (they differ in the split header)
        self.assertEqual({k: len(v) for k, v in by_count.items()}, {3: 1, 4: 1})


class OwnFiles(unittest.TestCase):
    """#444: --gguf-dir goes through gguf_dir_shards / gguf_choice / gguf_unsupported."""

    @staticmethod
    def names(model):
        d = setup.MODELS[model]
        return [setup.model_file(FAM, model, i) for i in range(1, d["shards"] + 1)]

    def test_the_new_files_are_no_longer_refused(self):
        for model in NEW:
            for n in self.names(model):
                self.assertIsNone(setup.gguf_unsupported(n), n)
            self.assertEqual(setup.gguf_choice(self.names(model)[0]), ("unsloth", model), model)

    def test_ud_iq1_s_stays_refused(self):
        for i in (1, 2, 3):
            self.assertEqual(setup.gguf_unsupported("Qwen3.8-Flash-Next-UD-IQ1_S-0000%d-of-00003.gguf" % i), "UD-IQ1_S")
        self.assertNotIn("UD-IQ1_S", setup.MODELS)
        self.assertIsNone(setup.gguf_choice("Qwen3.8-Flash-Next-UD-IQ1_S-00001-of-00003.gguf"))

    def test_the_bf16_and_other_unsloth_files_stay_refused(self):
        self.assertEqual(setup.gguf_unsupported("Qwen3.8-Flash-Next-BF16-00001-of-00008.gguf"), "BF16")
        self.assertIsNone(setup.gguf_unsupported("mmproj-BF16.gguf"))

    def _folder(self, t, names):
        for n in names:
            (Path(t) / n).write_bytes(b"")
        return Path(t)

    def test_gguf_dir_shards_finds_all_shards_by_the_count_in_the_name(self):
        for model in NEW:
            with tempfile.TemporaryDirectory() as t:
                folder = self._folder(t, self.names(model))
                got = setup.gguf_dir_shards(folder, FAM, model)
                self.assertEqual([p.name for p in got], self.names(model), model)

    def test_the_v31_iq4_xs_is_not_taken_for_the_pinned_one(self):
        with tempfile.TemporaryDirectory() as t:
            folder = self._folder(t, self.names("UD-IQ4_XS-v31"))                   # the four-shard files
            got = setup.gguf_dir_shards(folder, FAM, "UD-IQ4_XS")                   # asked for the three-shard set
            self.assertEqual(got[0].name, "Qwen3.8-Flash-Next-UD-IQ4_XS-00001-of-00003.gguf")
            self.assertFalse(got[0].exists())
            problem = setup.gguf_dir_problem(folder, got[0], FAM, "UD-IQ4_XS")
            self.assertIsNotNone(problem)
            self.assertIn("--family unsloth --model UD-IQ4_XS-v31", problem[1])
            ok = setup.gguf_dir_shards(folder, FAM, "UD-IQ4_XS-v31")
            self.assertTrue(all(p.exists() for p in ok))
            self.assertIsNone(setup.gguf_dir_problem(folder, ok[0], FAM, "UD-IQ4_XS-v31"))

    def test_ud_iq1_s_in_a_folder_is_named_as_not_usable(self):
        iq1s = ["Qwen3.8-Flash-Next-UD-IQ1_S-0000%d-of-00003.gguf" % i for i in (1, 2, 3)]
        with tempfile.TemporaryDirectory() as t:
            folder = self._folder(t, iq1s)
            first = setup.gguf_dir_shards(folder, FAM, "UD-IQ1_M")[0]
            problem = setup.gguf_dir_problem(folder, first, FAM, "UD-IQ1_M")
            self.assertIsNotNone(problem)
            self.assertIn("UD-IQ1_S", problem[0] + problem[1])
            self.assertIn(setup.UNSUPPORTED_WHY["UD-IQ1_S"], problem[1])
            msg, hint = setup.gguf_dir_problem(folder, Path(t) / iq1s[0], FAM, "UD-IQ1_S")
            self.assertIn(setup.UNSUPPORTED_WHY["UD-IQ1_S"], hint)


class SecondQ4Layout(unittest.TestCase):
    """UD-Q4_K_XL at d9620119: the same tensors, the table in a shard of its own.  The pinned files stay what setup
    downloads; this layout is accepted from files already here."""

    def test_the_names_are_the_pinned_ones_and_the_pins_differ_in_every_shard(self):
        self.assertEqual(sorted(setup.UNSLOTH_Q4_V31_SHARDS), sorted(setup.UNSLOTH_SHARDS))
        for n, pin in setup.UNSLOTH_Q4_V31_SHARDS.items():
            self.assertNotEqual(pin[0], setup.UNSLOTH_SHARDS[n][0], n)      # the shard boundaries moved: no size shared
            self.assertEqual(FAM["sha256"][n], setup.UNSLOTH_SHARDS[n])     # the family's pin (the download) stays
            self.assertEqual(FAM["sha256_alt"][n], (pin,))

    @needs_fixture
    def test_the_second_layout_is_the_hubs(self):
        for n, pin in setup.UNSLOTH_Q4_V31_SHARDS.items():
            self.assertEqual(pins_of("UD-Q4_K_XL")[n], pin)
        self.assertEqual(sum(b for b, _ in setup.UNSLOTH_Q4_V31_SHARDS.values()),
                         sum(s["bytes"] for s in shards_of("UD-Q4_K_XL")))

    @staticmethod
    def _file(name, size, exists=True):
        return types.SimpleNamespace(name=name, exists=lambda: exists, stat=lambda: types.SimpleNamespace(st_size=size))

    def test_fam_pin_picks_the_layout_by_the_size_on_the_disk(self):
        for n, old in setup.UNSLOTH_SHARDS.items():
            new = setup.UNSLOTH_Q4_V31_SHARDS[n]
            self.assertEqual(setup.fam_pin(FAM, self._file(n, old[0])), old, n)
            self.assertEqual(setup.fam_pin(FAM, self._file(n, new[0])), new, n)
            self.assertEqual(setup.fam_pin(FAM, self._file(n, 123)), old, n)      # neither: verify_sha256 reports it
            self.assertEqual(setup.fam_pin(FAM, self._file(n, new[0], exists=False)), old, n)   # nothing there yet
        other = "Qwen3.8-Flash-Next-UD-IQ1_M-00003-of-00003.gguf"
        self.assertEqual(setup.fam_pin(FAM, self._file(other, 1)), setup.UNSLOTH_V31_SHARDS[other])

    def test_the_catalog_reads_both_layouts_as_ud_q4_k_xl(self):
        first = "Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf"
        self.assertEqual(setup.gguf_choice(first), ("unsloth", "UD-Q4_K_XL"))
        self.assertIsNone(setup.gguf_unsupported(first))

    def test_verify_sha256_takes_the_second_layout_from_the_users_folder(self):
        """--gguf-dir with the new files: each is checked against ITS pin; the old pin would have said 'not the
        published file'."""
        with tempfile.TemporaryDirectory() as t:
            f = Path(t) / "Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf"
            f.write_bytes(b"x" * 100)
            alt = (100, "ab" * 32)
            fam = {"sha256": {f.name: setup.UNSLOTH_SHARDS[f.name]}, "sha256_alt": {f.name: (alt,)}}
            self.assertEqual(setup.fam_pin(fam, f), alt)
            with mock.patch.object(setup, "_hash_file", return_value=alt[1]):
                setup.verify_sha256(f, *setup.fam_pin(fam, f), own=True)
            self.assertIn("sha256 " + alt[1], f.with_name(f.name + ".done").read_text())


@needs_fixture
class HeaderRegression(unittest.TestCase):
    """Every set of the repository at d9620119: the pins and the tensor directories agree with each other."""

    def test_thirteen_sets(self):
        self.assertEqual(sorted(DOC["sets"]), sorted(["BF16", "Q8_0", "UD-IQ1_M", "UD-IQ1_S", "UD-IQ2_XS", "UD-IQ3_XXS",
                                                      "UD-IQ4_XS", "UD-Q2_0", "UD-Q2_K_XL", "UD-Q3_K_XL", "UD-Q4_K_XL",
                                                      "UD-Q5_K_XL", "UD-Q6_K_XL"]))
        self.assertEqual(DOC["revision"], REV31)

    def test_split_keys_and_tensor_counts(self):
        for set_name, d in DOC["sets"].items():
            with self.subTest(set=set_name):
                shards = d["shards"]
                m = re.search(r"-(\d{5})-of-(\d{5})\.gguf$", shards[0]["name"])
                self.assertEqual(len(shards), int(m.group(2)))
                names = [s["name"] for s in shards]
                self.assertEqual(names, sorted(names))
                for i, s in enumerate(shards, 1):
                    self.assertIn("-%05d-of-%05d.gguf" % (i, len(shards)), s["name"])
                    h = s["header"]
                    if h is None:
                        continue
                    self.assertEqual(h["split"], [i - 1, len(shards), 1224], s["name"])
                have = [s for s in shards if s["header"]]
                if len(have) == len(shards) - 1 and shards[0]["header"] is None:   # every shard but the metadata one
                    seen = [t[0] for s in have for t in s["header"]["tensors"]]
                    self.assertEqual(len(seen), len(set(seen)))
                    self.assertEqual(len(seen), 1224)

    def test_every_tensor_lies_inside_its_pinned_shard(self):
        for set_name, d in DOC["sets"].items():
            for s in d["shards"]:
                h = s["header"]
                if not h:
                    continue
                with self.subTest(shard=s["name"]):
                    end = 0
                    for name, type_id, shape, off in h["tensors"]:
                        t = FX.G.TensorInfo(name, shape, type_id, FX.G.GGML_TYPES.get(type_id, "?"), off)
                        self.assertIsNotNone(t.expected_bytes(), f"{name}: type {type_id}")
                        end = max(end, off + t.expected_bytes())
                    self.assertLessEqual(h["data_start"] + end, s["bytes"])
                    self.assertGreaterEqual(h["data_start"] + end, s["bytes"] - 64)   # it ends with its last tensor

    def test_the_catalog_sets_have_the_tensor_mix_of_the_plan(self):
        """Routed gate/up types by layer, down types, the Q5_0 downs (the fork's kernels read Q5_0 downs)."""
        want = {
            "UD-IQ1_M": ({"IQ2_XXS": 23, "IQ2_S": 18, "IQ2_XS": 7}, {"Q2_0": 46, "IQ4_NL": 2}),
            "UD-Q2_0": ({"IQ2_XXS": 18, "IQ2_S": 30}, {"Q2_0": 39, "IQ4_NL": 9}),
            "UD-IQ2_XS": ({"IQ2_XXS": 8, "IQ2_S": 40}, {"IQ4_NL": 24, "Q2_0": 23, "Q5_0": 1}),
            "UD-Q2_K_XL": ({"IQ2_XXS": 5, "IQ2_S": 26, "IQ3_XXS": 17}, {"IQ4_NL": 31, "Q2_0": 16, "Q5_0": 1}),
            "UD-IQ3_XXS": ({"IQ2_S": 26, "IQ3_XXS": 21, "IQ4_XS": 1}, {"IQ4_NL": 45, "Q2_0": 3}),
            "UD-Q3_K_XL": ({"IQ2_S": 4, "IQ3_XXS": 38, "IQ4_XS": 6}, {"IQ4_NL": 26, "Q5_0": 20, "Q2_0": 2}),
            "UD-IQ4_XS": ({"IQ3_XXS": 28, "IQ4_XS": 17, "IQ2_S": 3}, {"Q5_0": 33, "IQ4_NL": 13, "Q2_0": 2}),
            "UD-Q4_K_XL": ({"Q4_K": 47, "Q5_K": 1}, {"Q5_1": 43, "Q8_0": 5}),
            "UD-Q5_K_XL": ({"Q5_K": 47, "Q6_K": 1}, {"Q8_0": 48}),
            "UD-Q6_K_XL": ({"Q6_K": 47, "Q8_0": 1}, {"Q8_0": 48}),
        }
        for set_name, (gu, down) in want.items():
            gate, d = {}, {}
            for s in shards_of(set_name):
                for name, type_id, _shape, _off in (s["header"] or {"tensors": []})["tensors"]:
                    ty = FX.G.GGML_TYPES[type_id]
                    if re.fullmatch(r"blk\.\d+\.ffn_gate_exps\.weight", name):
                        gate[ty] = gate.get(ty, 0) + 1
                    elif re.fullmatch(r"blk\.\d+\.ffn_down_exps\.weight", name):
                        d[ty] = d.get(ty, 0) + 1
            with self.subTest(set=set_name):
                self.assertEqual(gate, gu)
                self.assertEqual(d, down)

    def test_ud_iq1_s_layer_18_has_different_gate_and_up_types(self):
        types_ = {}
        for s in shards_of("UD-IQ1_S"):
            for name, type_id, _shape, _off in (s["header"] or {"tensors": []})["tensors"]:
                if name in ("blk.18.ffn_gate_exps.weight", "blk.18.ffn_up_exps.weight"):
                    types_[name] = FX.G.GGML_TYPES[type_id]
        self.assertEqual(types_, {"blk.18.ffn_gate_exps.weight": "IQ2_S", "blk.18.ffn_up_exps.weight": "IQ2_XXS"})


if __name__ == "__main__":
    unittest.main()
