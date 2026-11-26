#!/bin/bash
# SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
# SPDX-License-Identifier: LGPL-3.0-or-later
#
# tools/lint/run.sh [FILE...] - the lints of AGENTS.md on the given files, by kind; without files, on the files that
# differ from HEAD (changed or new, not deleted).  Run from the repository root.  Exits 1 when any lint reports, and
# when a lint could not run because its tool is not installed (the tools are listed at the end; STRATA_LINT_ALLOW_SKIP=1
# accepts that and exits 0 when nothing else reports, and a comma-separated list of lints, STRATA_LINT_ALLOW_SKIP=clang-tidy
# for the CI, which has no build, accepts those alone).  The tools come from the distribution or from .lint/
# (docs/DEVTOOLS.md#lints): .lint/venv/bin, .lint/bin and .lint/node_modules/.bin are put on the PATH here.
# STRATA_LINT_BUILD (or XESTRATA_LINT_BUILD, upstream XeStrata's name) names the build folder whose compile_commands.json
# clang-tidy reads (default build/free; configure it with -DCMAKE_EXPORT_COMPILE_COMMANDS=ON); STRATA_CLANG_TIDY names a
# clang-tidy that knows SYCL (default: oneAPI's under /opt/intel/oneapi/compiler).  src/kernels/xe/sdpa_cudnn.cpp is
# only compiled in a contrib build that finds cudnn.h: tools/lint/cudnn-headers.sh puts the headers under $HOME and
# prints the STRATA_CUDNN_ROOT to configure that build with, so the source is checked here instead of being skipped.
set -u
cd "$(git rev-parse --show-toplevel)" || exit 2
C=tools/lint
BUILD=${XESTRATA_LINT_BUILD:-${STRATA_LINT_BUILD:-build/free}}
PATH=$PWD/.lint/venv/bin:$PWD/.lint/bin:$PWD/.lint/node_modules/.bin:$PATH
export PATH
CLANG_TIDY=${STRATA_CLANG_TIDY:-$(find /opt/intel/oneapi/compiler -path '*/bin/compiler/clang-tidy' 2>/dev/null | sort -V | tail -1)}
# else the clang-tidy of the intel/llvm built from source (docs/DEVTOOLS.md#lints: clang-tools-extra), which knows SYCL too
[ -z "$CLANG_TIDY" ] && [ -x .tools/intel-llvm/install/bin/clang-tidy ] && CLANG_TIDY=.tools/intel-llvm/install/bin/clang-tidy

if [ $# -gt 0 ]; then
  files=("$@")
else
  mapfile -t files < <( { git diff --name-only --diff-filter=d HEAD; git ls-files --others --exclude-standard; } | sort -u)
fi
[ ${#files[@]} -gt 0 ] || { echo "nothing to lint"; exit 0; }
all=("${files[@]}")
# Other projects' files, community benchmark records and license texts are kept as received: gitleaks and reuse only.
# XeStrata's patches in third_party are linted.
mapfile -t files < <(printf '%s\n' "${files[@]}" | while read -r f; do
  case "$f" in ""|third_party/*/*/*|bench/XeStrataBenchResults/*|LICENSES/*)
    case "$f" in third_party/*/*/patches/*) ;; *) continue ;; esac ;;
  esac
  echo "$f"
done)

status=0
skipped=()
run() {   # run NAME TOOL ARGS...: run a lint, or say it is not installed (and count it: the run then fails)
  local name=$1 tool=$2; shift 2
  if ! command -v "$tool" >/dev/null 2>&1 && [ ! -x "$tool" ]; then
    echo "== $name: not run ($tool not installed)"
    skipped+=("$name")
    return
  fi
  echo "== $name"
  "$tool" "$@" || status=1
}
pick() {  # pick PATTERN: the files whose path matches the extended regex
  printf '%s\n' "${files[@]}" | grep -E "$1" || true
}
code() {  # code PATTERN: as pick, without the lint configurations here (each tool reads its own dialect)
  pick "$1" | grep -v "^$C/" || true
}

mapfile -t patches < <(pick '^third_party/[^/]+/[^/]+/patches/.*\.patch$')
mapfile -t py < <(code '\.py$')
mapfile -t sh < <(pick '\.sh$')
mapfile -t cpp < <(pick '\.(cpp|hpp)$')
mapfile -t cmake < <(pick '(^|/)CMakeLists\.txt$|\.cmake$')
mapfile -t md < <(pick '\.md$')
mapfile -t css < <(pick '\.css$')
mapfile -t js < <(code '\.js$')
mapfile -t html < <(pick '\.html$')
mapfile -t text < <(pick '\.(py|sh|cpp|hpp|cu|h|inc|cmake|md|css|js|html|jinja|txt|toml|ini|yaml|rb|mjs)$|CMakeLists\.txt$')

if [ ${#patches[@]} -gt 0 ]; then
  run "patch changes" python3 "$C/patches.py" "${patches[@]}"
fi
if [ ${#py[@]} -gt 0 ]; then
  run flake8 flake8 --config "$C/flake8.ini" "${py[@]}"
  run ruff ruff check --quiet --config "$C/ruff.toml" "${py[@]}"
  run mypy mypy --config-file "$C/mypy.ini" --no-error-summary "${py[@]}"
fi
[ ${#sh[@]} -gt 0 ] && run shellcheck shellcheck "${sh[@]}"
if [ ${#cpp[@]} -gt 0 ]; then
  if [ -n "$CLANG_TIDY" ] && [ -f "$BUILD/compile_commands.json" ]; then
    tidy=("$CLANG_TIDY" --quiet -p "$BUILD" --config-file="$C/clang-tidy.yaml" --warnings-as-errors='*')
    # The folders of an engine carried over from upstream Strata as upstream wrote it (Inkling-2-Mini's, and the
    # codecs of the .strata v1.1 files it reads): three checks are off there.  They flag the idiom of that code - bit
    # fields unpacked from signed integers, index products in int, sums narrowed to the stored width - about a
    # thousand times, and writing each one out would leave no line as upstream has it, for the next integration to
    # carry over.  Every other check stays on there, and the code XeStrata writes elsewhere keeps all of them.  The
    # tests of that engine taken from upstream (tests/core/inkling2mini_*) include its headers whole, so the three are
    # off for them as well.
    carried='^((src|include/strata)/(inkling2mini|quant)/|tests/core/inkling2mini_)'
    relaxed=("--checks=-bugprone-signed-bitwise,-bugprone-implicit-widening-of-multiplication-result,-bugprone-narrowing-conversions")
    mapfile -t tus < <(python3 -c 'import json, sys; print("\n".join(sorted({e["file"] for e in json.load(open(sys.argv[1]))})))' \
      "$BUILD/compile_commands.json")
    # a source this build does not compile (another GPU maker's, as the AMD-only HIP files on a build without HIP):
    # clang-tidy would guess its command from a neighbor and report the headers it cannot find, so it is left out
    cxx=()
    cxxc=()
    for f in "${cpp[@]}"; do
      [[ $f == *.cpp ]] || continue
      if grep -qxF "$PWD/$f" <<<"$(printf '%s\n' "${tus[@]}")"; then   # not a pipe: grep -q ends it early
        if [[ $f =~ $carried ]]; then cxxc+=("$f"); else cxx+=("$f"); fi
      else
        echo "== clang-tidy: $f skipped (not compiled in $BUILD)"
      fi
    done
    mapfile -t hpp < <(printf '%s\n' "${cpp[@]}" | grep '\.hpp$')
    # Select only these source files. A header filter alone can match an external path containing /src/../.
    lines_of() { python3 -c 'import json, os, sys; print(json.dumps([{"name": os.path.abspath(f)} for f in sys.argv[1:]]))' "$@"; }
    if [ ${#cxx[@]} -gt 0 ]; then
      run clang-tidy "${tidy[@]}" --header-filter='.*' --line-filter="$(lines_of "${cxx[@]}")" "${cxx[@]}"
    fi
    if [ ${#cxxc[@]} -gt 0 ]; then
      run "clang-tidy (an engine carried over from upstream: three checks off)" "${tidy[@]}" "${relaxed[@]}" \
        --header-filter='.*' --line-filter="$(lines_of "${cxxc[@]}")" "${cxxc[@]}"
    fi
    # A header is checked inside a source that includes it, with that source's command: alone it gets the command of
    # whichever source's path is nearest, often one without -fsycl, and misses the macros a source defines before the
    # include (strata/kernels/device_arch_tag.hpp).  --line-filter keeps the header's findings only.
    if [ ${#hpp[@]} -gt 0 ]; then
      tu_of() {  # tu_of HEADER: the first source that includes it, itself or through other headers (four deep)
        local want=("$1") next f inc t
        for _ in 1 2 3 4; do
          next=()
          for f in "${want[@]}"; do
            case "$f" in include/*) inc="[<\"]${f#include/}[>\"]" ;; *) inc="\"([^\"]*/)?${f##*/}\"" ;; esac
            t=$(grep -l -E "^#include $inc" "${tus[@]}" 2>/dev/null | head -n 1)
            [ -n "$t" ] && { echo "$t"; return; }
            mapfile -t -O "${#next[@]}" next < <(grep -rl -E --include='*.h' --include='*.hpp' "^#include $inc" src include 2>/dev/null)
          done
          [ ${#next[@]} -gt 0 ] || return
          want=("${next[@]}")
        done
      }
      for h in "${hpp[@]}"; do
        tu=$(tu_of "$h")
        htidy=("${tidy[@]}")
        [[ $h =~ $carried ]] && htidy+=("${relaxed[@]}")
        if [ -n "$tu" ]; then
          run "clang-tidy $h (in ${tu#"$PWD"/})" "${htidy[@]}" --header-filter='.*' --line-filter="[{\"name\":\"$PWD/$h\"}]" "$tu"
        else
          run "clang-tidy $h" "${htidy[@]}" --header-filter='.*' --line-filter="[{\"name\":\"$PWD/$h\"}]" "$h"
        fi
      done
    fi
  else
    echo "== clang-tidy: not run (needs a clang-tidy that knows SYCL, STRATA_CLANG_TIDY or oneAPI's, and $BUILD/compile_commands.json)"
    skipped+=(clang-tidy)
  fi
  run cppcheck cppcheck --quiet --error-exitcode=1 --std=c++20 --enable=warning,performance --library="$C/cppcheck.cfg" \
    --inline-suppr \
    --suppress=missingIncludeSystem --suppress=noCopyConstructor --suppress=noOperatorEq "${cpp[@]}"
fi
[ ${#cmake[@]} -gt 0 ] && run cmake-lint cmake-lint -c "$C/cmake-format.py" -- "${cmake[@]}"
if [ ${#md[@]} -gt 0 ]; then
  run markdownlint mdl --style "$C/mdl_style.rb" "${md[@]}"
  run lychee lychee --offline --no-progress --include-fragments "${md[@]}"
fi
[ ${#css[@]} -gt 0 ] && run stylelint stylelint --config "$C/stylelint.config.mjs" \
  --config-basedir .lint "${css[@]}"
[ ${#js[@]} -gt 0 ] && run eslint eslint --no-eslintrc -c "$C/eslintrc.js" "${js[@]}"
for f in "${html[@]}"; do run "tidy $f" tidy -config "$C/tidy.conf" -errors -quiet "$f"; done
[ ${#text[@]} -gt 0 ] && run codespell codespell --config "$C/codespell.cfg" "${text[@]}"
for f in "${all[@]}"; do
  [ -f "$f" ] || continue
  gitleaks detect --no-git --redact --no-banner --exit-code 1 --source "$f" >/dev/null 2>&1 \
    || { echo "== gitleaks: a possible secret in $f (gitleaks detect --no-git --redact --source $f)"; status=1; }
done
# upstream's name where a user reads XeStrata's (messages, environment variables, programs, documents)
if [ ${#files[@]} -gt 0 ]; then
  echo "== upstream's name"
  # advisory here: this fork keeps the STRATA_* names and strata-inkling-2-mini, which the lint reports on almost every
  # file; CI's new_findings.py still counts the findings a change adds
  python3 "$C/upstream_name.py" "${files[@]}" || true
  # Each file's SPDX header against where its code comes from (AGENTS.md, Licenses).
  echo "== license classes"
  python3 tools/license_class.py "${files[@]}" || status=1
fi
run reuse reuse lint --quiet
if [ ${#skipped[@]} -gt 0 ]; then
  echo "== not run, their tools are not installed: ${skipped[*]} (docs/DEVTOOLS.md#lints)"
  allow=${STRATA_LINT_ALLOW_SKIP:-0}
  missing=()
  for n in "${skipped[@]}"; do
    [ "$allow" = 1 ] || [[ ",$allow," == *",$n,"* ]] || missing+=("$n")
  done
  if [ ${#missing[@]} -eq 0 ]; then
    echo "   accepted (STRATA_LINT_ALLOW_SKIP=$allow)"
  else
    status=1
  fi
fi
exit $status
