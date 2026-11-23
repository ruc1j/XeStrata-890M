# SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
# SPDX-License-Identifier: LGPL-3.0-or-later
# upstream Strata's short names for Inkling-2-Mini -> the model's name in full (AGENTS.md, "Model names: written out, never abbreviated").
# Apply to an upstream file or patch before it goes into XeStrata: sed -E -f tools/upstream/inkling2mini_names.sed
s#strata/inkling2/inkling2_loop#strata/inkling2mini/inkling2mini_loop#g
s#\binkling2_loop\b#inkling2mini_loop#g
s#\bINKLING2_LOOP#INKLING2MINI_LOOP#g
s#\binkling2::#inkling2mini::#g
s#namespace inkling2\b#namespace inkling2mini#g
s#\bink_sycl_backend\b#inkling2mini_sycl_backend#g
s#\bink_sycl\b#inkling2mini_sycl#g
s#\bink2\b#inkling2mini#g
s#\bink_#inkling2mini_#g
s#\bINK_#INKLING2MINI_#g
s#\bIK2_#INKLING2MINI_#g
s#\bINKLING_ENABLE_#INKLING2MINI_ENABLE_#g
s#\bInk([A-Z])#Inkling2Mini\1#g
s#strata-inkling\b#strata-inkling-2-mini#g
s#strata_inkling\b#strata_inkling2mini#g
s#strata::inkling2\b#strata::inkling2mini#g
s#\bkInklingRmsEps\b#kInkling2MiniRmsEps#g
s#\beval_inkling2_expert#eval_inkling2mini_expert#g
s#\bINKLING_(VERSION|DEFINE_STBI)\b#INKLING2MINI_\1#g
s#\bparse_ink_manifest\b#parse_inkling2mini_manifest#g
s#\bSyclInkBackend\b#SyclInkling2MiniBackend#g
s#\[inkling2\]#[inkling2mini]#g
s#\[inkling2-vision\]#[inkling2mini-vision]#g
# the Python side: upstream's short names for Inkling-2-Mini -> the model's name in full
s#strata_control import inkling_family, inkling_fit#serve import inkling2mini_family, inkling2mini_fit#g
s#from strata_control import inkling_fit#from serve import inkling2mini_fit#g
s#from strata_control import inkling_family#from serve import inkling2mini_family#g
s#\binkling_(family|fit|kind|dialect|tokenizer|mock_engine|source|engine|images|folder|offered|pc|menu_lines|have|verify|fetch|tokenizer_dir)\b#inkling2mini_\1#g
s#\bis_inkling(_exe)?\b#is_inkling2mini\1#g
s#\bInkling(Dialect|OutputParser)\b#Inkling2Mini\1#g
s#\bNotInklingTokenizer\b#NotInkling2MiniTokenizer#g
s#\bSTRATA_INKLING_SOURCE\b#STRATA_INKLING2MINI_SOURCE#g
s#\bstrata_inkling_images\b#strata_inkling2mini_images#g
s#strata-inkling-images#strata-inkling-2-mini-images#g
s#inkling-sources\.json#inkling-2-mini-sources.json#g
s#component:strata-inkling\b#component:strata-inkling-2-mini#g
s#strata-inkling\b#strata-inkling-2-mini#g
s#strata-inkling-2-mini-2-mini#strata-inkling-2-mini#g
