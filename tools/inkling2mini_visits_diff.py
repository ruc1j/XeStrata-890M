# SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
# SPDX-License-Identifier: LGPL-3.0-or-later
"""tools/inkling2mini_visits_diff.py A.bin B.bin [VISITS] - compares two dumps of the stages of Inkling-2-Mini's
visits (strata-inkling-2-mini with INKLING2MINI_DUMP_VISITS=<file>; a record: int32 tag = visit * 10 + stage, int32
count, the floats) and prints for each record the difference's size relative to A's (the norm of A - B over the norm
of A) and the largest single difference.  Stages: 1 the residual stream after the attention half, 2 the MoE's normed
input, 3 the router's logits, 9 the visit's output.  Run the same tokens through two backends (--cpu and the GPU's) to
see at which visit and stage they part.  VISITS: only the first that many visits."""
import struct
import sys


def records(path):
    out = []
    with open(path, "rb") as f:
        while True:
            head = f.read(8)
            if len(head) < 8:
                return out
            visit, count = struct.unpack("<ii", head)
            out.append((visit, struct.unpack(f"<{count}f", f.read(4 * count))))


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if len(argv) not in (2, 3):
        print(__doc__)
        return 2
    a, b = records(argv[0]), records(argv[1])
    if len(argv) == 3:
        a, b = ([r for r in x if r[0] // 10 < int(argv[2])] for x in (a, b))
    if len(a) != len(b):
        print(f"the dumps differ in length: {len(a)} and {len(b)} records")
    for (va, xa), (vb, xb) in zip(a, b):
        if va != vb or len(xa) != len(xb):
            print(f"record mismatch: visit {va} ({len(xa)} values) against visit {vb} ({len(xb)})")
            return 1
        na = sum(x * x for x in xa) ** 0.5
        nd = sum((x - y) ** 2 for x, y in zip(xa, xb)) ** 0.5
        mx = max(abs(x - y) for x, y in zip(xa, xb))
        print(f"visit {va // 10:2d} stage {va % 10}: |A| {na:10.4f}  |A-B|/|A| {nd / na if na else float('nan'):.3e}  "
              f"max |a-b| {mx:.3e}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
