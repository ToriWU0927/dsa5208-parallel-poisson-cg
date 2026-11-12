#!/usr/bin/env bash
# E1 — serial runtime scaling with problem size (grading item R2).
#
# Part A (time per iteration): fixed-iteration runs, problem "ones", over N
#         and over nnz (via d = 2/3 and stencil radius r = 1/2).
# Part B (time to solution):   solve-mode runs to tol, smaller sizes only,
#         because the iteration count grows like O(n).
#
# Usage:
#   scripts/run_e1_serial.sh [OUT_CSV]          # full sizes (run on Vanda/Atlas)
#   QUICK=1 scripts/run_e1_serial.sh            # small sizes for a local check
# Environment overrides: EXE, REPEAT (part A), SOLVE_REPEAT (part B, default
#   REPEAT), FIXED_ITERS, TOL, RUN_TIMEOUT (seconds per configuration, default
#   840 s, below the 900 s watchdog limit; a timeout stops the script so a hang cannot burn core-hours), and the
#   size lists N2_FIXED, N3_FIXED, N2_SOLVE, N3_SOLVE (space separated).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
EXE="${EXE:-$ROOT/build/poisson_serial}"
OUT="${1:-$ROOT/results/e1_serial.csv}"
REPEAT="${REPEAT:-3}"
FIXED_ITERS="${FIXED_ITERS:-100}"
TOL="${TOL:-1e-8}"
RUN_TIMEOUT="${RUN_TIMEOUT:-840}"   # below the watchdog stall limit (900 s)

if [[ "${QUICK:-0}" == 1 ]]; then
    N2_FIXED="${N2_FIXED:-128 256 512 1024}"
    N3_FIXED="${N3_FIXED:-32 64 128}"
    N2_SOLVE="${N2_SOLVE:-128 256 512}"
    N3_SOLVE="${N3_SOLVE:-32 64}"
    REPEAT=1
else
    N2_FIXED="${N2_FIXED:-128 256 512 1024 2048 4096 8192}"
    N3_FIXED="${N3_FIXED:-32 64 128 256 384 512}"
    N2_SOLVE="${N2_SOLVE:-128 256 512 1024 2048 4096}"
    N3_SOLVE="${N3_SOLVE:-32 64 128 256 384}"
fi
SOLVE_REPEAT="${SOLVE_REPEAT:-$REPEAT}"
# every configuration is time-limited (coreutils timeout; skipped where it is missing, e.g. macOS)
TO=()
command -v timeout >/dev/null 2>&1 && TO=(timeout --kill-after=20 "$RUN_TIMEOUT")

[[ -x "$EXE" ]] || { echo "missing $EXE (run: make serial)" >&2; exit 1; }
mkdir -p "$(dirname "$OUT")"
"$EXE" --dim 2 --n 8 --repeat 1 --header | head -n 1 > "$OUT"   # CSV header only

run() {  # run REPEATS options...: append data rows (no header) of one configuration
    local rep=$1; shift
    echo "  $(date +%T) $*" >&2
    local rc=0
    ${TO[@]+"${TO[@]}"} "$EXE" --repeat "$rep" "$@" >> "$OUT" || rc=$?
    if [[ $rc -eq 124 || $rc -eq 137 ]]; then
        echo "timeout after ${RUN_TIMEOUT}s: $* -> aborting" >&2; exit 3
    elif [[ $rc -ne 0 ]]; then
        echo "run failed (rc=$rc): $*" >&2; exit "$rc"
    fi
}

echo "E1-A: time per iteration (fixed $FIXED_ITERS iterations)" >&2
for r in 1 2; do
    for n in $N2_FIXED; do run "$REPEAT" --dim 2 --n "$n" --radius "$r" --problem ones --fixed-iters "$FIXED_ITERS"; done
    for n in $N3_FIXED; do run "$REPEAT" --dim 3 --n "$n" --radius "$r" --problem ones --fixed-iters "$FIXED_ITERS"; done
done

echo "E1-B: time to solution (tol $TOL, $SOLVE_REPEAT repetition(s))" >&2
for r in 1 2; do
    for n in $N2_SOLVE; do run "$SOLVE_REPEAT" --dim 2 --n "$n" --radius "$r" --problem ones --tol "$TOL"; done
    for n in $N3_SOLVE; do run "$SOLVE_REPEAT" --dim 3 --n "$n" --radius "$r" --problem ones --tol "$TOL"; done
done

echo "wrote $OUT" >&2
