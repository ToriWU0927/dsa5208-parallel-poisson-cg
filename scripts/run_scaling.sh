#!/usr/bin/env bash
# MPI / MPI+OpenMP scaling experiments (E2 E3 E5 E6, diag, e6_2d). Runs INSIDE a
# PBS job on whole, exclusive Vanda CPU nodes (72 cores each) and only uses
# the process counts that belong to the job's node count, so the same script
# is submitted once per node count (see scripts/pbs/submit_scaling.sh).
#
#   scripts/run_scaling.sh e2|e3|e5|e6|diag|e6_2d
#
# diag:  the E2 configurations with --sync-timers 1 (t_bc, t_wait; 11/11 re-run for E4)
# e6_2d: the 2D half of E6 again (11/11, after the 2D first-touch fix in fill_all)
#
# Every configuration is one mpirun; ranks are spread evenly over the nodes
# (ppr = ranks per node) and bound to cores; OpenMP threads are bound to the
# cores of their rank (OMP_PLACES=cores, OMP_PROC_BIND=close).
# Output: results/<exp>_vanda_n<nodes>.csv (CSV rows of poisson_mpi).
#
# Environment overrides: EXE, SERIAL, REPEAT (3), ITERS (200), CPN (72 cores per node), OUT,
#   RUN_TIMEOUT (seconds per configuration, default 600)
#
# Guard against burning core-hours on a hung run: every
# configuration runs under `timeout`; a run that times out ABORTS the whole job
# (exit 3), because a hang is almost always systemic (network, file system,
# launcher) and the following runs would hang too. A run that fails normally
# (e.g. an impossible decomposition, exit 2) is skipped and the job continues.
# Progress lines with time stamps go to stderr (the job output file).
set -uo pipefail

EXP="${1:?usage: run_scaling.sh e2|e3|e5|e6|diag|e6_2d}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
EXE="${EXE:-$ROOT/build/poisson_mpi}"
SERIAL="${SERIAL:-$ROOT/build/poisson_serial}"
REPEAT="${REPEAT:-3}"
ITERS="${ITERS:-200}"
CPN="${CPN:-72}"
RUN_TIMEOUT="${RUN_TIMEOUT:-600}"
HOSTFILE="${PBS_NODEFILE:?must run inside a PBS job}"
NODES=$(sort -u "$HOSTFILE" | wc -l)
OUT="${OUT:-$ROOT/results/${EXP}_vanda_n${NODES}.csv}"
# never append a re-run to an existing result file (the medians would mix two runs):
# a second run of the same experiment gets the PBS job id in its file name
if [[ -s "$OUT" && -z "${APPEND:-}" ]]; then
    OUT="${OUT%.csv}_${PBS_JOBID%%.*}.csv"
    echo "result file exists -> writing to $OUT" >&2
fi

# Launch remote ranks through PBS (tm) instead of ssh, so that they belong to
# the job (cleaned up by qdel, inside the job's cpuset). The foss/2023b Open MPI
# has the tm component but cannot find libpbs.so.0 without this (Vanda 11/11).
[[ -d /opt/pbs/lib ]] && export LD_LIBRARY_PATH="/opt/pbs/lib:${LD_LIBRARY_PATH:-}"

mkdir -p "$ROOT/results"
[[ -x "$EXE" ]] || { echo "missing $EXE" >&2; exit 1; }
if [[ ! -s "$OUT" ]]; then   # CSV header (time-limited like every other launch)
    timeout --kill-after=10 60 mpirun -np 1 --hostfile "$HOSTFILE" "$EXE" --dim 2 --n 8 --repeat 1 \
        --header > "$OUT.tmp" || { echo "header run failed or timed out -> aborting" >&2; exit 3; }
    head -n 1 "$OUT.tmp" > "$OUT"; rm -f "$OUT.tmp"
fi

# run RANKS THREADS -- program options...
run() {
    local ranks=$1 threads=$2; shift 3
    local ppn=$(( (ranks + NODES - 1) / NODES ))
    if (( ranks % NODES != 0 || ppn * threads > CPN )); then
        echo "skip: $ranks ranks x $threads threads does not fit $NODES nodes" >&2; return
    fi
    # Binding: one core per rank (pure MPI) or `threads` consecutive cores per
    # rank. Open MPI's own --map-by ppr:N:node:PE=t hangs at launch on Vanda
    # (foss/2023b, tm launch; 11/11 diagnostics: always across nodes, even for
    # `hostname`, and sometimes on one node), so with threads > 1 the ranks are
    # started unbound and scripts/bind_rank.sh binds each one with hwloc to the
    # cores Open MPI would use (local rank i -> logical cores i*t ... i*t+t-1).
    local map="ppr:${ppn}:node" bind="--bind-to core" wrap=()
    if (( threads > 1 )); then bind="--bind-to none"; wrap=("$ROOT/scripts/bind_rank.sh"); fi
    echo "  $(date +%T) [$EXP] nodes=$NODES ranks=$ranks threads=$threads $*" >&2
    guarded env OMP_NUM_THREADS=$threads OMP_PLACES=cores OMP_PROC_BIND=close \
        mpirun -np "$ranks" --hostfile "$HOSTFILE" --map-by "$map" $bind \
        -x OMP_NUM_THREADS -x OMP_PLACES -x OMP_PROC_BIND \
        ${wrap[@]+"${wrap[@]}"} "$EXE" --problem ones --repeat "$REPEAT" "$@"
}

# guarded CMD...: run with a time limit, append stdout to $OUT
guarded() {
    local t0=$SECONDS rc
    timeout --kill-after=20 "$RUN_TIMEOUT" "$@" >> "$OUT"
    rc=$?
    if (( rc == 124 || rc == 137 )); then
        echo "  $(date +%T) TIMEOUT after ${RUN_TIMEOUT}s -> aborting job to save core-hours" >&2
        exit 3
    elif (( rc != 0 )); then
        echo "  $(date +%T) FAILED rc=$rc (skipped, continuing)" >&2
    else
        echo "  $(date +%T) done in $(( SECONDS - t0 ))s" >&2
    fi
}

# pure-MPI process counts for this node count (strong / weak scaling)
case $NODES in
    1) PLIST="1 2 4 8 16 32 64 72" ;;
    2) PLIST="128 144" ;;
    4) PLIST="256 288" ;;
    *) PLIST="$(( NODES * CPN ))" ;;
esac

case $EXP in
e2)  # strong scaling: fixed global problem, r = 1
    if (( NODES == 1 )); then  # serial baseline Ts on the same (exclusive) node type
        for spec in "--dim 3 --n 512" "--dim 2 --n 8192"; do
            echo "  $(date +%T) [$EXP] serial $spec" >&2
            guarded "$SERIAL" --problem ones --repeat "$REPEAT" $spec --fixed-iters "$ITERS"
        done
    fi
    for p in $PLIST; do
        run "$p" 1 -- --dim 3 --n 512  --fixed-iters "$ITERS"
        run "$p" 1 -- --dim 2 --n 8192 --fixed-iters "$ITERS"
    done ;;
diag)  # E2 configurations with diagnostic timers: boundary fill (t_bc) and waiting before Allreduce (t_wait)
    for p in $PLIST; do
        run "$p" 1 -- --dim 3 --n 512  --fixed-iters "$ITERS" --sync-timers 1
        run "$p" 1 -- --dim 2 --n 8192 --fixed-iters "$ITERS" --sync-timers 1
    done ;;
e6_2d)  # 2D half of E6 (MPI x OpenMP), re-run after the 2D first-touch fix
    for tpr in 1 2 3 4 6 9 12 18 24 36 72; do
        p=$(( NODES * CPN / tpr ))
        run "$p" "$tpr" -- --dim 2 --n 8192 --fixed-iters "$ITERS"
    done ;;
e3)  # weak scaling: fixed problem per process (global n = L * processes per dimension)
    for p in $PLIST; do
        run "$p" 1 -- --dim 3 --local-n 192  --fixed-iters "$ITERS"
        run "$p" 1 -- --dim 2 --local-n 2048 --fixed-iters "$ITERS"
    done ;;
e5)  # communication variants at one full node or all nodes
    p=$(( NODES * CPN ))
    for spec in "--dim 3 --n 512" "--dim 2 --n 8192"; do
        for r in 1 2; do
            for halo in block overlap; do
                for dc in auto slab; do
                    for cg in std single; do
                        run "$p" 1 -- $spec --radius $r --halo $halo --decomp $dc --cg $cg \
                            --fixed-iters "$ITERS"
                    done
                done
            done
        done
    done ;;
e6)  # MPI x OpenMP: ranks per node x threads per rank = 72
    for tpr in 1 2 3 4 6 9 12 18 24 36 72; do
        p=$(( NODES * CPN / tpr ))
        run "$p" "$tpr" -- --dim 3 --n 512  --fixed-iters "$ITERS"
        run "$p" "$tpr" -- --dim 2 --n 8192 --fixed-iters "$ITERS"
    done ;;
*) echo "unknown experiment $EXP" >&2; exit 2 ;;
esac
echo "wrote $OUT" >&2
