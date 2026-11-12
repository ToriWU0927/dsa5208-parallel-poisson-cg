#!/usr/bin/env python3
"""V2 / V3 / V4 — the parallel programs must reproduce the serial solution.

For each case the serial program is run once as the reference and the MPI
program is run for several process counts p and program variants. Pass if
  * the iteration count differs from the serial one by at most 1, and
  * ||x_mpi - x_serial||_2 / ||x_serial||_2 < 1e-10.
The solutions are compared element-wise on the global grid (written with --save).

Results are not bit-identical because the order of the floating-point sums
in the inner products depends on p (floating-point addition is not
associative), hence the relative tolerance.

  V2: p in {1, 2, 3, 4, 7, 8, 12, ...}, including p that does not divide n and prime p
  V4: halo block vs overlap, decomposition auto vs slab, std vs single-reduction CG,
      pure MPI vs MPI+OpenMP
  V3 (--variant-set cuda, --mpi build/poisson_mpi_cuda): MPI + CUDA with CUDA-aware MPI
      and with host staging, std and single-reduction CG, auto and slab decomposition
      (one GPU per rank; with more ranks than GPUs, ranks share a GPU)

Must run where mpirun can start the largest p (inside a PBS job on the cluster).
Usage:
  python3 tests/v2_mpi_consistency.py [--procs 1 2 3 4 7 8 12] [--launcher "mpirun -np {p}"]
                                      [--out results/v2_mpi.csv]
"""
import argparse
import array
import csv
import io
import math
import os
import shlex
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# (dim, radius, n): n chosen so that most p do not divide it
CASES = [(2, 1, 97), (2, 2, 97), (3, 1, 37), (3, 2, 37)]
TOL_CG = 1e-12
TOL_X = 1e-10

# (label, extra MPI options, OMP threads, CG variant)
VARIANTS = [
    ("block", [], 1, "std"),
    ("overlap", ["--halo", "overlap"], 1, "std"),
    ("slab", ["--decomp", "slab"], 1, "std"),
    ("single", ["--cg", "single"], 1, "single"),
    ("overlap+single", ["--halo", "overlap", "--cg", "single"], 1, "single"),
    ("omp2", [], 2, "std"),
]
VARIANTS_CUDA = [
    ("cuda-aware", ["--gpu-aware", "1"], 1, "std"),
    ("staging", ["--gpu-aware", "0"], 1, "std"),
    ("aware+single", ["--gpu-aware", "1", "--cg", "single"], 1, "single"),
    ("staging+single", ["--gpu-aware", "0", "--cg", "single"], 1, "single"),
    ("aware+slab", ["--gpu-aware", "1", "--decomp", "slab"], 1, "std"),
]


def run(cmd, env=None, timeout=None):
    out = subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                         universal_newlines=True, env=env, timeout=timeout).stdout
    return next(csv.DictReader(io.StringIO(out)))


def read_bin(path):
    a = array.array("d")
    with open(path, "rb") as f:
        a.frombytes(f.read())
    return a


def rel_diff(a, b):
    num = math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))
    den = math.sqrt(sum(y * y for y in b))
    return num / den


def write_rows(path, rows):
    if not rows:
        return
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--serial", default=ROOT / "build" / "poisson_serial")
    ap.add_argument("--mpi", default=ROOT / "build" / "poisson_mpi")
    ap.add_argument("--procs", type=int, nargs="+", default=[1, 2, 3, 4, 7, 8, 12])
    ap.add_argument("--variant-set", choices=["cpu", "cuda"], default="cpu",
                    help="cpu: V2/V4 variants of poisson_mpi; cuda: V3 variants of poisson_mpi_cuda")
    ap.add_argument("--launcher", default="mpirun -np {p}")
    ap.add_argument("--max-cores", type=int, default=0,
                    help="skip runs with p*threads above this (0 = no limit)")
    ap.add_argument("--out", default=ROOT / "results" / "v2_mpi.csv")
    ap.add_argument("--timeout", type=float, default=120,
                    help="seconds per MPI run; a timeout aborts the whole test (a hang is "
                         "usually systemic, so the remaining runs would hang too)")
    args = ap.parse_args()

    tmp = Path(tempfile.mkdtemp(prefix="v2_", dir=os.environ.get("TMPDIR")))
    rows, ok = [], True
    print(f"{'dim':>3} {'r':>2} {'n':>4} {'variant':>15} {'p':>4} {'thr':>3} "
          f"{'decomp':>14} {'iters':>6} {'ser':>5} {'rel_diff':>10}  result")
    for dim, r, n in CASES:
        base = ["--dim", str(dim), "--n", str(n), "--radius", str(r), "--problem", "mms",
                "--tol", f"{TOL_CG:g}", "--repeat", "1", "--verify", "--header"]
        ref = {}
        for cgv in ("std", "single"):
            path = tmp / f"ser_{dim}_{r}_{cgv}.bin"
            rec = run([str(args.serial), *base, "--cg", cgv, "--save", str(path)])
            ref[cgv] = (int(rec["iters"]), read_bin(path))
        for label, extra, thr, cgv in (VARIANTS_CUDA if args.variant_set == "cuda" else VARIANTS):
            for p in args.procs:
                if args.max_cores and p * thr > args.max_cores:
                    continue
                if "slab" in label and n // p < r:  # a slab must be at least r layers thick
                    continue
                path = tmp / f"mpi_{dim}_{r}_{label}_{p}.bin"
                env = dict(os.environ, OMP_NUM_THREADS=str(thr))
                cmd = shlex.split(args.launcher.format(p=p)) + [str(args.mpi), *base, *extra,
                                                                "--save", str(path)]
                try:
                    rec = run(cmd, env, args.timeout)
                except subprocess.TimeoutExpired:
                    print(f"{dim:>3} {r:>2} {n:>4} {label:>15} {p:>4} {thr:>3}  TIMEOUT after "
                          f"{args.timeout:.0f}s -> aborting to save core-hours", flush=True)
                    write_rows(args.out, rows)
                    sys.exit(3)
                except subprocess.CalledProcessError as e:
                    print(f"{dim:>3} {r:>2} {n:>4} {label:>15} {p:>4} {thr:>3}  RUN FAILED: "
                          f"{e.stderr.strip()[:200]}")
                    ok = False
                    continue
                it_ref, x_ref = ref[cgv]
                it = int(rec["iters"])
                d = rel_diff(read_bin(path), x_ref)
                passed = abs(it - it_ref) <= 1 and d < TOL_X
                ok &= passed
                print(f"{dim:>3} {r:>2} {n:>4} {label:>15} {p:>4} {thr:>3} {rec['decomp']:>14} "
                      f"{it:>6} {it_ref:>5} {d:>10.2e}  {'PASS' if passed else 'FAIL'}")
                rows.append({"impl": rec["impl"], "dim": dim, "radius": r, "n": n, "variant": label,
                             "ranks": p, "gpus": rec["gpus"], "gpu_aware": rec["gpu_aware"],
                             "threads": thr, "decomp": rec["decomp"], "iters": it,
                             "iters_serial": it_ref, "rel_diff_x": d,
                             "err_inf": rec["err_inf"], "rel_res": rec["rel_res"],
                             "true_rel_res": rec["true_rel_res"],
                             "host": rec["host"], "pass": int(passed)})
                path.unlink()

    write_rows(args.out, rows)
    print(f"OVERALL: {'PASS' if ok else 'FAIL'}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
