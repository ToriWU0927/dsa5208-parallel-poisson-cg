#!/usr/bin/env python3
"""V1 — discretisation convergence order of the serial solver.

Solves the manufactured problem (--problem mms) on successively refined grids
for d = 2, 3 and stencil radius r = 1, 2, and reports the observed order
    p = log(e_coarse / e_fine) / log(h_coarse / h_fine),   e = max |u_h - u|.
Expected: p ~ 2 for r = 1 and p ~ 4 for r = 2.

The CG tolerance is set far below the discretisation error so that the solver
error does not pollute the measurement.

Usage: python3 tests/v1_convergence.py [--exe build/poisson_serial] [--out results/v1_convergence.csv]
"""
import argparse
import csv
import io
import math
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

CASES = {
    # (dim, radius): (grid sizes, CG tolerance)
    (2, 1): ([16, 32, 64, 128, 256], 1e-12),
    (2, 2): ([16, 32, 64, 128, 256], 1e-13),
    (3, 1): ([8, 16, 32, 64, 128], 1e-12),
    (3, 2): ([8, 16, 32, 64, 128], 1e-13),
}
EXPECTED = {1: 2.0, 2: 4.0}
ORDER_TOL = 0.25  # pass if |observed - expected| < ORDER_TOL on the finest pair


def run(exe, dim, r, n, tol):
    cmd = [str(exe), "--dim", str(dim), "--n", str(n), "--radius", str(r),
           "--problem", "mms", "--tol", f"{tol:g}", "--repeat", "1", "--verify", "--header"]
    out = subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                         universal_newlines=True).stdout
    return next(csv.DictReader(io.StringIO(out)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exe", default=ROOT / "build" / "poisson_serial")
    ap.add_argument("--out", default=ROOT / "results" / "v1_convergence.csv")
    args = ap.parse_args()

    rows, ok = [], True
    print(f"{'dim':>3} {'r':>2} {'n':>5} {'h':>10} {'iters':>6} {'err_inf':>11} {'order':>6}")
    for (dim, r), (sizes, tol) in CASES.items():
        prev = None
        for n in sizes:
            rec = run(args.exe, dim, r, n, tol)
            h = 1.0 / (n + 1)
            err = float(rec["err_inf"])
            order = math.log(prev[1] / err) / math.log(prev[0] / h) if prev else float("nan")
            print(f"{dim:>3} {r:>2} {n:>5} {h:>10.3e} {rec['iters']:>6} {err:>11.3e} {order:>6.2f}")
            rows.append({"dim": dim, "radius": r, "n": n, "h": h, "iters": rec["iters"],
                         "true_rel_res": rec["true_rel_res"], "err_inf": err, "order": order})
            prev = (h, err)
        final = rows[-1]["order"]
        passed = abs(final - EXPECTED[r]) < ORDER_TOL
        ok &= passed
        print(f"    -> finest-pair order {final:.2f}, expected {EXPECTED[r]:.0f}: "
              f"{'PASS' if passed else 'FAIL'}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {out}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
