# SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
# SPDX-License-Identifier: LGPL-3.0-or-later
"""The findings a change adds: tools/lint/run.sh's output on the changed files before the change (BASE) and after it
(FINAL), compared with the line numbers dropped, so a finding that only moved is not new.  Prints the findings FINAL
has more of and exits 1 when there are any (AGENTS.md, Lints: the files a change touches get no new findings; findings
that were there before may stay).

    python3 tools/lint/new_findings.py BASE.txt FINAL.txt
"""
from __future__ import annotations

import collections
import re
import sys

# a finding line of the lints run.sh runs: "path:line[:col]: ...", a tool's "warning:"/"error:" line, a pylint-style
# "[C0301]" code (cmake-lint), and run.sh's own reports (gitleaks, a lint whose tool is missing)
FINDING = re.compile(r"^\S+:\d+[:\s]|(warning|error|style|performance|portability|information):|\[[\w.-]+\]"
                     r"|possible secret|: not run \(")
LINE = re.compile(r":\d+(:\d+)?(?=[:\s])")


def findings(path: str) -> collections.Counter:
    counts: collections.Counter = collections.Counter()
    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if line and FINDING.search(line) and ": note:" not in line:   # mypy's notes are not findings
                counts[LINE.sub("", line)] += 1
    return counts


def main(argv: list[str]) -> int:
    if len(argv) != 3:
        print(__doc__.strip().splitlines()[-1].strip())
        return 2
    base, final = findings(argv[1]), findings(argv[2])
    new = [(k, final[k] - base[k]) for k in final if final[k] > base[k]]
    print(f"findings: {sum(base.values())} before the change, {sum(final.values())} after it, {len(new)} new")
    for k, n in new:
        print(f"  +{n} {k}")
    return 1 if new else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
