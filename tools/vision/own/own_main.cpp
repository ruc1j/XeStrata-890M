// SPDX-FileCopyrightText: 2026 Niko1221 and the Strata contributors
// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later AND MIT
// tools/vision/own/own_main.cpp - strata-vision-own: the picture reader on our own tower alone (no ggml, no llama.cpp, no mtmd).
//
//   strata-vision-own --mmproj <mmproj.gguf> [--gpu] [--threads N] [--max-tokens N] [--min-tokens N]    (--model, --tower, --flash-attn and --gpu-pci are accepted and ignored)
// The same protocol as strata-vision (see own_reader.h).  It builds in seconds, which is what the tower's CPU checks use; the shipped program is still
// strata-vision, which carries this reader as `--tower own`.  --gpu is refused: the own tower here reads on the CPU only.
#include "own_reader.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

int main(int argc, char** argv) {
    strata_vit::OwnReaderOptions o;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto next = [&]() -> std::string {
            if (i + 1 >= argc) { std::fprintf(stderr, "%s needs a value\n", a.c_str()); std::exit(2); }
            return argv[++i];
        };
        if (a == "--mmproj") o.mmproj = next();
        else if (a == "--gpu") o.gpu = true;
        else if (a == "--threads") o.threads = std::atoi(next().c_str());
        else if (a == "--max-tokens") o.max_tokens = std::atoi(next().c_str());
        else if (a == "--min-tokens") o.min_tokens = std::atoi(next().c_str());
        else if (a == "--model" || a == "--tower" || a == "--flash-attn" || a == "--gpu-pci") next();
        else if (a == "--version") { std::printf("strata-vision-own\n"); return 0; }
        else { std::fprintf(stderr, "unknown argument %s\n", a.c_str()); return 2; }
    }
    if (o.mmproj.empty()) { std::fprintf(stderr, "usage: strata-vision-own --mmproj <mmproj.gguf> [--gpu] [--threads N] [--max-tokens N] [--min-tokens N]\n"); return 2; }
    return strata_vit::own_reader_run(o);
}
