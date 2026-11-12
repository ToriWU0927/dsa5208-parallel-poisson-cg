#!/usr/bin/env python3
"""Figures and tables for the report from the CSV files in results/.

    python3 scripts/plot_results.py [--results results] [--out figures]

Writes figures/F*.pdf (for the report) and F*.png (for a quick look), and
figures/tables.md (numbers quoted in the report: fitted slopes, speedup and
efficiency, weak efficiency, network parameters, model vs measured halo time).

Every point is the MEDIAN over the repetitions of one configuration. All
performance data come from Vanda CPU nodes (Intel Xeon Platinum 8452Y, 72
cores per node, foss/2023b); the script checks that each figure uses one CPU
type. Needs numpy and matplotlib only.
"""
import argparse
import collections
import csv
import glob
import math
import os
import re
import statistics

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt                      # noqa: E402
from matplotlib.ticker import FixedLocator, NullLocator, FuncFormatter  # noqa: E402

# ---------------------------------------------------------------- style ----
# Categorical slots in fixed order (validated: adjacent CVD dE >= 9.1 on a light
# surface). Slots 3 to 5 are below 3:1 contrast on white, so every series
# also has its own marker and a legend, and the numbers are in tables.md.
# Slot 5 is used only by the five-part breakdown (F7).
C = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]
MK = ["o", "s", "^", "D", "v"]
INK, INK2, MUTED, GRID, REF = "#0b0b0b", "#52514e", "#8a8984", "#e4e3df", "#7d7c78"
CORES_PER_NODE = 72

plt.rcParams.update({
    "font.size": 8.5, "axes.titlesize": 9, "axes.labelsize": 8.5,
    "legend.fontsize": 7.5, "xtick.labelsize": 7.5, "ytick.labelsize": 7.5,
    "axes.edgecolor": MUTED, "axes.labelcolor": INK2, "text.color": INK,
    "xtick.color": INK2, "ytick.color": INK2, "axes.titlecolor": INK,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "grid.linestyle": "-",
    "axes.axisbelow": True, "lines.linewidth": 1.6, "lines.markersize": 4.5,
    "legend.frameon": False, "figure.dpi": 110, "savefig.bbox": "tight",
    "pdf.fonttype": 42,
})


def style_line(i):
    return dict(color=C[i], marker=MK[i], markeredgecolor="white", markeredgewidth=0.6)


def ref_line(ax, x, y, label, where=-1, off=(0, 0)):
    ax.plot(x, y, color=REF, lw=0.9, ls=(0, (4, 3)), zorder=1)
    ax.annotate(label, (x[where], y[where]), color=INK2, fontsize=7,
                ha="left" if where == -1 else "right", va="center",
                xytext=((3 if where == -1 else -3) + off[0], off[1]), textcoords="offset points")


def p_axis(ax, ps, node_line=True, ticks=None):
    ax.set_xscale("log", base=2)
    # a tick is left out when the next process count is within 25 % (64 vs 72,
    # 128 vs 144, 256 vs 288): the full-node counts keep their labels
    if ticks is None:
        ticks = [p for p in ps if not any(p < q < 1.25 * p for q in ps)]
    ax.xaxis.set_major_locator(FixedLocator(ticks))
    ax.xaxis.set_minor_locator(NullLocator())
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{int(round(v))}"))
    if node_line and max(ps) > CORES_PER_NODE:
        ax.axvline(CORES_PER_NODE, color=MUTED, lw=0.8, zorder=1)
        ax.annotate("> 1 node", (CORES_PER_NODE, 1), xycoords=("data", "axes fraction"),
                    xytext=(3, -3), textcoords="offset points", ha="left", va="top",
                    fontsize=7, color=INK2)
    for lab in ax.get_xticklabels():
        lab.set_rotation(0)


def save(fig, out, name):
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(out, f"{name}.{ext}"), dpi=200)
    plt.close(fig)
    print("wrote", os.path.join(out, name + ".{pdf,png}"))


# ----------------------------------------------------------------- data ----
NUM = {"dim", "radius", "nx", "ny", "nz", "N", "nnz", "nodes", "ranks", "threads", "iters",
       "repeat_id", "converged", "n", "bytes", "gpus", "gpu_aware"}


def load(pattern):
    """rows of all files matching pattern (re-runs carry a job id in the file name,
    e.g. e2_vanda_n1_1541234.csv, and are not matched by the n[0-9] patterns)"""
    rows = []
    for f in sorted(glob.glob(pattern)):
        with open(f) as fh:
            file_rows = list(csv.DictReader(fh))
        for r in file_rows:
            for k, v in r.items():
                if k in NUM:
                    r[k] = int(float(v))
                elif k and k.startswith(("t_", "time", "rel", "true", "err", "h", "order", "bandwidth")):
                    try:
                        r[k] = float(v)
                    except (TypeError, ValueError):
                        pass
            rows.append(r)
    if not rows:
        raise SystemExit(f"no data: {pattern}")
    return rows


def groups(rows, *keys):
    g = collections.defaultdict(list)
    for r in rows:
        g[tuple(r[k] for k in keys)].append(r)
    return g


def med(rs, key, per_iter=False):
    v = [r[key] / (r["iters"] if per_iter else 1) for r in rs]
    return statistics.median(v)


def fit_slope(x, y):
    """least-squares slope and intercept of log y = a log x + b"""
    a, b = np.polyfit(np.log(x), np.log(y), 1)
    return a, b


def one_cpu_type(rows, what):
    # the CSV host column is rank 0's node only; the full node list is in the job
    # output ("nodes:" line). Vanda GPU nodes (GN) have the same CPU (Xeon 8452Y).
    hosts = {r.get("host", "")[:2].upper() for r in rows}
    if not hosts <= {"CN", "GN"}:
        print(f"WARNING {what}: rows from unexpected hosts {hosts}")


SERIES = [(2, 1), (2, 2), (3, 1), (3, 2)]  # (dim, radius): slots 1..4
SNAME = {(2, 1): "2D, r = 1", (2, 2): "2D, r = 2", (3, 1): "3D, r = 1", (3, 2): "3D, r = 2"}


# ------------------------------------------------------------- figures ----
def fig_e1(R, out, T):
    e1 = load(f"{R}/e1_serial_vanda.csv")
    one_cpu_type(e1, "E1")
    fx = groups([r for r in e1 if r["mode"] == "fixed"], "dim", "radius", "nx")
    sv = groups([r for r in e1 if r["mode"] == "solve"], "dim", "radius", "nx")

    # F1: time per iteration vs N and vs nnz
    fig, axs = plt.subplots(1, 2, figsize=(6.6, 2.7), sharey=True)
    T.append("## E1 — serial scaling (Vanda, 1 core of an exclusive node)\n")
    T.append("| series | slope of t_iter vs N | slope vs nnz | ns / point / iteration (range) |")
    T.append("|---|---|---|---|")
    for i, s in enumerate(SERIES):
        ks = sorted(k for k in fx if k[:2] == s)
        N = np.array([fx[k][0]["N"] for k in ks], float)
        nnz = np.array([fx[k][0]["nnz"] for k in ks], float)
        t = np.array([med(fx[k], "t_iter") for k in ks])
        axs[0].plot(N, t, label=SNAME[s], **style_line(i))
        axs[1].plot(nnz, t, label=SNAME[s], **style_line(i))
        a1, _ = fit_slope(N, t)
        a2, _ = fit_slope(nnz, t)
        npp = t / N * 1e9
        T.append(f"| {SNAME[s]} | {a1:.2f} | {a2:.2f} | {npp.min():.2f} – {npp.max():.2f} |")
    for ax, xs, lab in ((axs[0], "N", "unknowns N"), (axs[1], "nnz", "non-zeros of A (nnz)")):
        allx = np.array(sorted({r[xs] for r in e1 if r["mode"] == "fixed"}), float)
        x = np.array([allx.min(), allx.max()])
        y0 = 3.5e-9 * (x if xs == "N" else x / 7)
        ref_line(ax, x, y0 * 0.45, "O(N)" if xs == "N" else "O(nnz)")
        ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlabel(lab)
    axs[0].set_ylabel("time per CG iteration (s)")
    axs[0].set_title("Time per iteration vs N")
    axs[1].set_title("Time per iteration vs nnz")
    axs[0].legend(loc="upper left")
    save(fig, out, "F1_serial_time_per_iteration")

    # F2: time to solution vs N, and iterations vs n
    fig, axs = plt.subplots(1, 2, figsize=(6.6, 2.7))
    T.append("\n| series | iterations / n | slope of time-to-solution vs N (theory 2D 1.5, 3D 1.33) |")
    T.append("|---|---|---|")
    for i, s in enumerate(SERIES):
        ks = sorted(k for k in sv if k[:2] == s)
        N = np.array([sv[k][0]["N"] for k in ks], float)
        n = np.array([k[2] for k in ks], float)
        t = np.array([med(sv[k], "t_solve") for k in ks])
        it = np.array([sv[k][0]["iters"] for k in ks], float)
        assert all(sv[k][0]["converged"] == 1 for k in ks)
        axs[0].plot(N, t, label=SNAME[s], **style_line(i))
        axs[1].plot(n, it, label=SNAME[s], **style_line(i))
        big = N >= 1e5   # leave out the cache-resident sizes
        a, _ = fit_slope(N[big], t[big])
        T.append(f"| {SNAME[s]} | {np.median(it / n):.2f} | {a:.2f} |")
    # theory lines anchored 2.5x below the largest point of each dimension
    for d, expo, lab in ((2, 1.5, r"$O(N^{1.5})$"), (3, 4 / 3, r"$O(N^{4/3})$")):
        ks = [k for k in sv if k[0] == d and k[1] == 1]
        Nmax = max(sv[k][0]["N"] for k in ks)
        tmax = med(sv[max(ks, key=lambda k: k[2])], "t_solve")
        x = np.array([1e5, Nmax], float)
        # in the legend (direct labels collide with the curves at the right end)
        axs[0].plot(x, tmax / 2.5 * (x / Nmax) ** expo, color=REF, lw=0.9, zorder=1,
                    ls=(0, (4, 3)) if d == 2 else (0, (1, 2)), label=f"{lab} ({d}D reference)")
    x = np.array([30, 5000]); ref_line(axs[1], x, 1.1 * x, "O(n)")
    axs[0].set_xscale("log"); axs[0].set_yscale("log")
    axs[0].set_xlabel("unknowns N"); axs[0].set_ylabel("time to solution (s), tol 1e-8")
    axs[0].set_title("Time to solution vs N")
    axs[1].set_xscale("log"); axs[1].set_yscale("log")
    axs[1].set_xlabel("grid points per dimension n"); axs[1].set_ylabel("CG iterations")
    axs[1].set_title("Iterations vs n")
    axs[0].legend(loc="upper center", bbox_to_anchor=(1.1, -0.2), ncol=3, handlelength=2.2)
    save(fig, out, "F2_serial_time_to_solution")


def fig_v1(R, out, T):
    v1 = load(f"{R}/v1_convergence_vanda.csv")
    fig, ax = plt.subplots(figsize=(3.4, 2.7))
    T.append("\n## V1 — discretisation error (Vanda)\n")
    T.append("| series | observed order (finest pair) |")
    T.append("|---|---|")
    for i, s in enumerate(SERIES):
        rs = sorted((r for r in v1 if (r["dim"], r["radius"]) == s), key=lambda r: -r["h"])
        h = np.array([r["h"] for r in rs]); e = np.array([r["err_inf"] for r in rs])
        ax.plot(h, e, label=SNAME[s], **style_line(i))
        T.append(f"| {SNAME[s]} | {rs[-1]['order']:.2f} |")
    h = np.array([0.006, 0.06])
    ref_line(ax, h, 0.9 * h ** 2, "O(h²)", where=0)
    ref_line(ax, h, 0.2 * h ** 4, "O(h⁴)", where=0)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("grid spacing h"); ax.set_ylabel("max error ‖u_h − u‖∞")
    ax.set_title("V1: error vs h")
    ax.legend(loc="lower right")
    save(fig, out, "F3_v1_convergence")


def tables_validation(R, T):
    """T1: V2 - V4 (parallel vs serial) and V5 (true vs recursive residual)."""
    T.append("\n## T1 — V2 – V4: parallel solution vs serial solution (mms problem, solve to tol)\n")
    T.append("Pass: same iteration count (±1) and ‖x_par − x_ser‖₂ / ‖x_ser‖₂ < 1e-10.\n")
    T.append("| file | test | runs | passed | ranks | variants | max |Δ iterations| | max ‖Δx‖ / ‖x‖ |")
    T.append("|---|---|---|---|---|---|---|---|")
    for f in sorted(glob.glob(f"{R}/v2_mpi_*.csv")) + sorted(glob.glob(f"{R}/v3_cuda_*.csv")):
        rs = load(f)
        test = "V3 (GPU)" if "v3_" in f else "V2 / V4"
        ranks = sorted({r["ranks"] for r in rs})
        T.append(f"| {os.path.basename(f)} | {test} | {len(rs)} | {sum(int(r['pass']) for r in rs)} | "
                 f"{ranks[0]} – {ranks[-1]} | {len({r['variant'] for r in rs})} | "
                 f"{max(abs(r['iters'] - int(r['iters_serial'])) for r in rs)} | "
                 f"{max(float(r['rel_diff_x']) for r in rs):.1e} |")
    # V5: the CG recursion updates r instead of recomputing b - Ax; rounding can make the
    # two drift apart. true_rel_res is recomputed after the solve.
    T.append("\n## T1 — V5: true residual ‖b − Ax‖ / ‖b‖ vs CG recursive residual\n")
    T.append("| source | implementation | mode | runs | true / recursive (min – max) | max true_rel_res |")
    T.append("|---|---|---|---|---|---|")
    # solve-mode runs only: fixed-iteration runs on small grids continue past convergence,
    # where the recursive residual keeps shrinking below rounding level and the ratio is meaningless
    srcs = [("E1", f"{R}/e1_serial_vanda.csv")] + \
           [(os.path.basename(f)[:-4], f) for f in sorted(glob.glob(f"{R}/v[23]_*.csv"))]
    for name, pat in srcs:
        rs = [r for r in load(pat) if r.get("mode", "solve") == "solve"
              and isinstance(r.get("rel_res"), float) and r["rel_res"] > 0]
        if not rs:
            continue
        for impl, g in sorted(groups(rs, "impl").items()):
            q = [r["true_rel_res"] / r["rel_res"] for r in g]
            T.append(f"| {name} | {impl[0]} | solve | {len(g)} | {min(q):.2f} – {max(q):.2f} | "
                     f"{max(r['true_rel_res'] for r in g):.2e} |")
    T.append("\nIn solve mode CG stops when the recursive residual drops below tol = 1e-8; the true "
             "residual can end slightly above tol (same order of magnitude), as expected from rounding "
             "in the recursion.")


def amdahl(p, f):
    return 1.0 / (f + (1 - f) / p)


def fit_amdahl(p, S):
    fs = np.linspace(0, 0.2, 20001)
    err = [np.sum((np.log(amdahl(p, f)) - np.log(S)) ** 2) for f in fs]
    return fs[int(np.argmin(err))]


def fig_e2(R, out, T):
    e2 = load(f"{R}/e2_vanda_n[0-9].csv")
    one_cpu_type(e2, "E2")
    g = groups(e2, "impl", "dim", "ranks")
    dims = [(3, "3D, n = 512 (N = 1.3·10⁸)"), (2, "2D, n = 8192 (N = 6.7·10⁷)")]
    ps = sorted({k[2] for k in g if k[0] == "mpi"})
    data = {}
    for d, _ in dims:
        Ts = med(g[("serial", d, 1)], "t_iter")
        tp = np.array([med(g[("mpi", d, p)], "t_iter") for p in ps])
        data[d] = (Ts, tp)

    # F4: time vs p with ideal
    fig, ax = plt.subplots(figsize=(3.4, 2.8))
    for i, (d, lab) in enumerate(dims):
        Ts, tp = data[d]
        ax.plot(ps, tp, label=lab, **style_line(i))
        ax.plot(ps, Ts / np.array(ps, float), color=C[i], lw=0.8, ls=(0, (4, 3)), zorder=1)
    ax.plot([], [], color=REF, lw=0.8, ls=(0, (4, 3)), label="ideal T_s / p")
    ax.set_yscale("log"); p_axis(ax, ps)
    ax.set_xlabel("MPI processes p"); ax.set_ylabel("time per CG iteration (s)")
    ax.set_title("Strong scaling: time")
    ax.legend(loc="lower left")
    save(fig, out, "F4_strong_time")

    # F5: speedup and efficiency (two separate axes, not a dual axis)
    fig, axs = plt.subplots(1, 2, figsize=(6.6, 2.8))
    T.append("\n## T2 — E2 strong scaling (r = 1, 200 iterations, median of 3)\n")
    T.append("S(p) = T_s / T_p with T_s the serial code on the same node type; E(p) = S(p) / p.\n")
    T.append("| p | nodes | 3D t_iter (ms) | 3D S | 3D E | 2D t_iter (ms) | 2D S | 2D E |")
    T.append("|---|---|---|---|---|---|---|---|")
    pa = np.array(ps, float)
    for j, p in enumerate(ps):
        cells = []
        for d, _ in dims:
            Ts, tp = data[d]
            S = Ts / tp[j]
            cells += [f"{tp[j]*1e3:.2f}", f"{S:.2f}" if S < 10 else f"{S:.1f}", f"{S/p:.2f}"]
        T.append(f"| {p} | {math.ceil(p/CORES_PER_NODE)} | " + " | ".join(cells) + " |")
    fits = []
    for i, (d, lab) in enumerate(dims):
        Ts, tp = data[d]
        S = Ts / tp
        f = fit_amdahl(pa, S)
        fits.append((lab, f))
        axs[0].plot(ps, S, label=lab, **style_line(i))
        pf = np.geomspace(1, max(ps), 100)
        axs[0].plot(pf, amdahl(pf, f), color=C[i], lw=0.8, ls=(0, (1, 2)), zorder=1)
        axs[1].plot(ps, S / pa, label=lab, **style_line(i))
    axs[0].plot([1, max(ps)], [1, max(ps)], color=REF, lw=0.8, ls=(0, (4, 3)), zorder=1, label="ideal S = p")
    axs[0].plot([], [], color=REF, lw=0.8, ls=(0, (1, 2)), label="Amdahl fit")
    axs[0].set_yscale("log", base=2)
    axs[0].yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    for ax in axs:
        p_axis(ax, ps)
        ax.set_xlabel("MPI processes p")
    axs[0].set_ylabel("speedup S(p)"); axs[0].set_title("Strong scaling: speedup")
    axs[1].set_ylabel("efficiency E(p)"); axs[1].set_ylim(0, 1.05)
    axs[1].set_title("Strong scaling: efficiency")
    axs[0].legend(loc="upper left")
    save(fig, out, "F5_strong_speedup_efficiency")
    T.append("\nAmdahl fit (least squares on log S over all p): " +
             "; ".join(f"{lab.split(',')[0]} serial fraction f = {f:.4f}" for lab, f in fits) +
             ". The fit is poor (the measured speedup leaves the curve for p ≥ 144): inside a node "
             "the limit is memory bandwidth, not a serial part, so f must not be read as a serial "
             "fraction; across nodes use the speedup relative to 72 ranks below.")

    # speedup relative to one full node (separates node-internal bandwidth from scaling across nodes)
    T.append("\n| p | 3D S rel. to 72 ranks | 2D S rel. to 72 ranks |")
    T.append("|---|---|---|")
    j72 = ps.index(CORES_PER_NODE)
    for j, p in enumerate(ps):
        if p >= CORES_PER_NODE:
            T.append(f"| {p} | " + " | ".join(f"{data[d][1][j72]/data[d][1][j]:.2f}" for d, _ in dims) + " |")
    return data, ps


def fig_e3(R, out, T):
    e3 = load(f"{R}/e3_vanda_n[0-9].csv")
    one_cpu_type(e3, "E3")
    g = groups(e3, "dim", "ranks")
    dims = [(3, "3D, 192³ per process"), (2, "2D, 2048² per process")]
    ps = sorted({k[1] for k in g})
    pa = np.array(ps, float)
    fig, axs = plt.subplots(1, 2, figsize=(6.6, 2.8))
    T.append("\n## E3 — weak scaling (fixed points per process)\n")
    T.append("E_w(p) = t_iter(1) / t_iter(p); E_w,72(p) = t_iter(72) / t_iter(p) for p ≥ 72 "
             "(one full node as the base: separates memory-bandwidth saturation inside a node "
             "from scaling across nodes).\n")
    T.append("| p | 3D t_iter (ms) | 3D E_w | 3D E_w,72 | 2D t_iter (ms) | 2D E_w | 2D E_w,72 |")
    T.append("|---|---|---|---|---|---|---|")
    res = {}
    for i, (d, lab) in enumerate(dims):
        t = np.array([med(g[(d, p)], "t_iter") for p in ps])
        res[d] = t
        axs[0].plot(ps, t, label=lab, **style_line(i))
        Ew = t[0] / t
        axs[1].plot(ps, Ew, label=lab, **style_line(i))
    j72 = ps.index(CORES_PER_NODE)
    for j, p in enumerate(ps):
        T.append(f"| {p} | " + " | ".join(
            f"{res[d][j]*1e3:.1f} | {res[d][0]/res[d][j]:.2f} | "
            + (f"{res[d][j72]/res[d][j]:.2f}" if p >= CORES_PER_NODE else "–") for d, _ in dims) + " |")
    # Gustafson: scaled speedup S_w = p E_w = p - f (p - 1)  ->  fit f
    T.append("\nThe Gustafson fit below is dominated by the memory-bandwidth saturation inside a node "
             "(E_w drops between p = 1 and 72, then stays almost flat); the fitted f is not a serial "
             "fraction of the code.\n")
    for d, lab in dims:
        Sw = pa * res[d][0] / res[d]
        f = np.clip(np.sum((pa - Sw) * (pa - 1)) / np.sum((pa - 1) ** 2), 0, 1)
        T.append(f"Gustafson fit {lab.split(',')[0]}: serial fraction f = {f:.3f} (scaled speedup at p = {ps[-1]}: {Sw[-1]:.1f})")
    for ax in axs:
        p_axis(ax, ps); ax.set_xlabel("MPI processes p")
    axs[0].set_yscale("log")
    axs[0].yaxis.set_major_locator(FixedLocator([0.01, 0.02, 0.05, 0.1, 0.2]))
    axs[0].yaxis.set_minor_locator(NullLocator())
    axs[0].yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    axs[0].set_ylabel("time per CG iteration (s)")
    axs[0].set_title("Weak scaling: time")
    axs[1].set_ylabel("weak efficiency E_w(p)"); axs[1].set_ylim(0, 1.1)
    axs[1].set_title("Weak scaling: efficiency")
    axs[1].legend(loc="upper right")
    save(fig, out, "F6_weak_scaling")


def net_params(R, T):
    net = load(f"{R}/net_vanda.csv")
    pp = [r for r in net if r["test"] == "pingpong"]
    out = {}
    T.append("\n## Network (ping-pong, Open MPI 4.1.6, TCP between nodes)\n")
    T.append("| link | latency α (µs, 8 B) | bandwidth β (GB/s, 8 MB) |")
    T.append("|---|---|---|")
    for name, same in (("intra-node", True), ("inter-node", False)):
        rs = [r for r in pp if (r["host0"] == r["host1"]) == same]
        a = min(r["time_s"] for r in rs if r["bytes"] == 8)
        big = max(rs, key=lambda r: r["bytes"])
        b = big["bytes"] / big["time_s"]
        out[name] = (a, b)
        # one-way time vs message size (bytes 8 .. 8 MB), used by halo_model
        curve = sorted({r["bytes"]: r["time_s"] for r in rs}.items())
        out[name + " curve"] = (np.array([c[0] for c in curve], float),
                                np.array([c[1] for c in curve], float))
        T.append(f"| {name} | {a*1e6:.2f} | {b/1e9:.2f} |")
    # file order (scripts/pbs/vanda_net.pbs): 2 ranks on one node, then 2, 4, 16, 72, 144 ranks over 2 nodes
    ar = [r for r in net if r["test"] == "allreduce" and r["bytes"] == 8]
    T.append("\n8-byte MPI_Allreduce: " + ", ".join(
        f"{r['ranks']} ranks {'on 1 node' if i == 0 else 'over 2 nodes'} {r['time_s']*1e6:.0f} µs"
        for i, r in enumerate(ar)))
    return out


def local_box(decomp, n, dim):
    pd = [int(x) for x in decomp.split(":")[1].split("x")]   # processes along x, y (, z)
    return [math.ceil(n / q) for q in pd], pd


def pingpong_time(curve, m):
    """one-way message time for m bytes: log-log interpolation of the measured ping-pong"""
    b, t = curve
    return float(np.exp(np.interp(np.log(m), np.log(b), np.log(t))))


def halo_model(decomp, n, dim, r, curve):
    """one halo exchange (= one CG iteration) of the busiest rank, no contention.
    halo.cpp exchanges dimension by dimension with two blocking MPI_Sendrecv calls
    (send up / receive from below, then send down / receive from above). A rank sends
    and receives inside one call at the same time, so every decomposed dimension costs
    two message times, also when it has only 2 processes (the two directions are then
    in different calls). Message time from the measured ping-pong at the face size
    (r layers x local face area x 8 B), so the protocol cost of medium messages is
    included."""
    box, pd = local_box(decomp, n, dim)
    t = 0.0
    for k, q in enumerate(pd):
        if q == 1:
            continue
        area = math.prod(b for j, b in enumerate(box) if j != k)
        t += 2 * pingpong_time(curve, r * area * 8)
    return t


def breakdown_data(R):
    """rows and parts for F7 / F7b. Preferred: the diag runs (E2 configurations with
    --sync-timers 1): a timed barrier before every Allreduce separates the waiting for
    the slowest rank (t_wait) from the Allreduce itself, and t_bc separates the
    physical boundary fill from the halo exchange. Without them: the E2 runs."""
    if glob.glob(f"{R}/diag_vanda_n[0-9].csv"):
        rows = [r for r in load(f"{R}/diag_vanda_n[0-9].csv") if r["impl"] == "mpi"]
        for r in rows:
            r["t_xchg"] = r["t_halo"] - r["t_bc"]
        parts = [("compute", "t_compute"), ("halo exchange", "t_xchg"),
                 ("boundary fill", "t_bc"), ("Allreduce", "t_allreduce"),
                 ("barrier (waiting + sync)", "t_wait")]
        return rows, parts, True
    rows = [r for r in load(f"{R}/e2_vanda_n[0-9].csv") if r["impl"] == "mpi"]
    for r in rows:
        r["t_xchg"] = r["t_halo"]
    # t_allreduce then also holds the waiting time (load imbalance), t_halo the boundary fill
    parts = [("compute", "t_compute"), ("halo (incl. boundary fill)", "t_halo"),
             ("Allreduce (incl. waiting)", "t_allreduce")]
    return rows, parts, False


def fig_breakdown(R, out, T, net):
    rows, parts, diag = breakdown_data(R)
    g = groups(rows, "dim", "ranks")
    ps = sorted({k[1] for k in g})
    # The components of a rank add up to its t_solve, so "t_iter - sum of parts" is ~0
    # and is not shown; shares are of the sum of the (rank-averaged) parts.
    fig, axs = plt.subplots(1, 2, figsize=(6.6, 2.9), sharey=True)
    for ax, (d, title) in zip(axs, ((3, "3D, n = 512"), (2, "2D, n = 8192"))):
        x = np.arange(len(ps))
        bottom = np.zeros(len(ps))
        tot = np.array([med(g[(d, p)], "t_iter") for p in ps])
        vals = np.array([[med(g[(d, p)], k, per_iter=True) for p in ps] for _, k in parts])
        for i, (lab, _) in enumerate(parts):
            share = 100 * vals[i] / vals.sum(axis=0)
            ax.bar(x, share, bottom=bottom, width=0.72, color=C[i], label=lab,
                   edgecolor="white", linewidth=0.8)
            bottom += share
        for xi, p, t in zip(x, ps, tot):   # selective labels; all values are in tables.md
            if p in (1, CORES_PER_NODE, max(ps)):
                ax.annotate(f"{t*1e3:.3g} ms", (xi, 100), xytext=(0, 2), textcoords="offset points",
                            ha="center", va="bottom", fontsize=6.5, color=INK2)
        ax.set_xticks(x, [str(p) for p in ps], fontsize=6.5, rotation=45)
        if max(ps) > CORES_PER_NODE:
            ax.axvline(ps.index(CORES_PER_NODE) + 0.5, color=MUTED, lw=0.8)
        ax.set_title(f"{title} (labels: time per iteration)", pad=11)
        ax.set_xlabel("MPI processes p (categorical axis)")
        ax.set_ylim(0, 100); ax.grid(axis="x", visible=False)
    src = ("diag runs: E2 configurations, 100 iterations, --sync-timers 1" if diag
           else "E2 runs")
    T.append(f"\n## E4 — time per iteration split into parts ({src}; ms, averages over ranks)\n")
    T.append("| dim | p | t_iter (slowest rank) | " + " | ".join(l for l, _ in parts) + " |")
    T.append("|---|---|---|" + "---|" * len(parts))
    for d in (3, 2):
        for p in ps:
            rs = g[(d, p)]
            T.append(f"| {d}D | {p} | {med(rs, 't_iter')*1e3:.2f} | " +
                     " | ".join(f"{med(rs, k, per_iter=True)*1e3:.3f}" for _, k in parts) + " |")
    axs[0].set_ylabel("share of time per iteration (%)")
    axs[0].legend(loc="upper center", bbox_to_anchor=(1.0, -0.22), ncol=3 if len(parts) <= 3 else 5,
                  fontsize=7 if len(parts) > 3 else 7.5, columnspacing=1.2, handlelength=1.4)
    if diag:
        T.append("\nRuns: E2 configurations with --sync-timers 1 and 100 fixed iterations (E2: 200), "
                 "so the labels (t_iter of these runs) differ from F4 / T2. t_wait is a timed "
                 "MPI_Barrier before each of the 2 Allreduces per iteration (standard CG). Inside "
                 "one node a barrier costs µs, so t_wait is the waiting for the slowest rank (load "
                 "imbalance); the Allreduce itself then costs 2 – 40 µs per iteration. Across nodes "
                 "the barrier itself costs about as much as an Allreduce (0.2 – 0.4 ms per "
                 "iteration at p = 256 / 288), so there t_wait is waiting plus synchronisation. "
                 "The barriers make t_iter ≤ 2 % longer on 1 node (about 0.5 % of it is the "
                 "shorter run: the initial residual is spread over 100 instead of 200 "
                 "iterations); on 4 nodes 3D −1 / +2 %, 2D +6 / +13 % (+0.2 / +0.4 ms). These "
                 "runs are used for the breakdown only, not for speedups. Boundary fill = "
                 "physical (Dirichlet) ghost layers, also present at p = 1; it shrinks with the "
                 "local box and is done only by ranks at the domain boundary.")
    else:
        T.append("\nt_allreduce includes the time a rank waits in MPI_Allreduce for the slowest rank "
                 "(load imbalance): an 8-byte Allreduce alone costs ~1 µs inside a node (network "
                 "table), so values of 0.3 – 2.6 ms per iteration at p ≤ 72 are mostly waiting. "
                 "t_halo includes the physical boundary fill (also present at p = 1).")
    save(fig, out, "F7_communication_breakdown")

    # F7b: measured halo exchange per iteration vs alpha-beta model (bounds: all faces
    # inside a node / all faces across nodes)
    ci, ce = net["intra-node curve"], net["inter-node curve"]
    fig, axs = plt.subplots(1, 2, figsize=(6.6, 2.7), sharey=True)
    T.append("\n## E4 — halo exchange: measured vs message-time model (per CG iteration)\n")
    meas_lab = ("measured exchange (incl. packing)" if diag
                else "measured (incl. packing and boundary fill)")
    T.append("Model (a lower estimate): per decomposed dimension two message times (two blocking "
             "Sendrecv calls, also when the dimension has only 2 processes), each the measured "
             "one-way ping-pong time at the face size r·(face area)·8 B, largest local box; no "
             "contention for the network card or memory bandwidth (64 – 72 ranks per node share "
             "them). 'inter-node' assumes that every face crosses nodes: with 72 ranks per node "
             "only part of them do, so it is not an upper bound. " +
             ("The measured value is t_halo − t_bc (boundary fill removed); it still contains "
              "the packing of non-contiguous faces (MPI derived datatypes) and waiting for the "
              "neighbours. " if diag else
              "The measured t_halo also contains the physical boundary fill and the packing of "
              "non-contiguous faces (MPI derived datatypes). ") +
             f"At p = 1 (no messages) it is {med(g[(3, 1)], 't_xchg', per_iter=True)*1e3:.3f} ms (3D) "
             f"and {med(g[(2, 1)], 't_xchg', per_iter=True)*1e3:.3f} ms (2D) per iteration.\n")
    T.append("| dim | p | decomposition | measured (µs) | model intra-node (µs) | model inter-node (µs) |")
    T.append("|---|---|---|---|---|---|")
    for ax, (d, n, title) in zip(axs, ((3, 512, "3D, n = 512"), (2, 8192, "2D, n = 8192"))):
        pp = [p for p in ps if p > 1]
        meas = np.array([med(g[(d, p)], "t_xchg", per_iter=True) for p in pp])
        dec = [g[(d, p)][0]["decomp"] for p in pp]
        mi = np.array([halo_model(dc, n, d, 1, ci) for dc in dec])
        me = np.array([halo_model(dc, n, d, 1, ce) for dc in dec])
        ax.plot(pp, meas, label=meas_lab, **style_line(0))
        ax.plot(pp, mi, label="model, intra-node ping-pong", **style_line(1))
        ax.plot(pp, me, label="model, all faces inter-node", **style_line(2))
        for p, a, b, c, dc in zip(pp, meas, mi, me, dec):
            T.append(f"| {d}D | {p} | {dc} | {a*1e6:.0f} | {b*1e6:.0f} | {c*1e6:.0f} |")
        ax.set_yscale("log"); p_axis(ax, pp)
        ax.set_title(title); ax.set_xlabel("MPI processes p")
    axs[0].set_ylabel("halo time per iteration (s)")
    axs[0].legend(loc="lower left")
    save(fig, out, "F7b_halo_model")


def fig_e5(R, out, T):
    T.append("\n## E5 — communication variants (one full node / four full nodes)\n")
    T.append("Change of time per iteration relative to block halo + auto (Cartesian) decomposition + standard CG.\n")
    fig, axs = plt.subplots(2, 2, figsize=(6.6, 4.6), sharey=False)
    for row, nodes in enumerate((1, 4)):
        e5 = load(f"{R}/e5_vanda_n{nodes}.csv")
        for r in e5:
            r["decomp_kind"] = r["decomp"].split(":")[0]
        g = groups(e5, "dim", "radius", "halo", "decomp_kind", "cg")
        p = e5[0]["ranks"]
        for col, d in enumerate((3, 2)):
            ax = axs[row, col]
            cats = [(rr, dk, h) for rr in (1, 2) for dk in ("auto", "slab") for h in ("block", "overlap")]
            x = np.arange(len(cats))
            T.append(f"\n{d}D, p = {p}: base t_iter " +
                     ", ".join(f"r = {rr}: {med(g[(d, rr, 'block', 'auto', 'std')], 't_iter')*1e3:.2f} ms" for rr in (1, 2)))
            T.append("\n| r | decomposition | halo | std CG | single-reduction CG |")
            T.append("|---|---|---|---|---|")
            for i, cg in enumerate(("std", "single")):
                vals = []
                for rr, dk, h in cats:
                    key = (d, rr, h, dk, cg)
                    base = med(g[(d, rr, "block", "auto", "std")], "t_iter")
                    vals.append(100 * (med(g[key], "t_iter") / base - 1) if key in g else np.nan)
                ax.bar(x + (i - 0.5) * 0.38, np.nan_to_num(vals), width=0.36, color=C[i],
                       label="standard CG" if cg == "std" else "single-reduction CG",
                       edgecolor="white", linewidth=0.6)
                for xi, v in zip(x, vals):
                    if np.isnan(v) and cg == "std":   # one label per missing pair
                        ax.annotate("n/a", (xi, 0), xytext=(0, 2),
                                    textcoords="offset points", ha="center", fontsize=6, color=INK2)
                if cg == "std":
                    std_vals = vals
            for (rr, dk, h), vs, vg in zip(cats, std_vals, vals):
                f = lambda v: "n/a" if np.isnan(v) else f"{v:+.1f}%"
                T.append(f"| {rr} | {dk} | {h} | {f(vs)} | {f(vg)} |")
            ax.axhline(0, color=MUTED, lw=0.8)
            ax.set_xticks(x, [f"r{rr}\n{dk}\n{'ovlp' if h == 'overlap' else h}" for rr, dk, h in cats], fontsize=6)
            ax.set_title(f"{d}D, p = {p} ({nodes} node{'s' if nodes > 1 else ''})")
            ax.grid(axis="x", visible=False)
            if col == 0:
                ax.set_ylabel("time vs baseline (%)")
    axs[0, 0].legend(loc="upper left")
    fig.tight_layout(h_pad=1.2)
    save(fig, out, "F8_communication_variants")


# 2D rows of F9 from the e6_2d re-runs (after the 2D first-touch fix in fill_all) for
# these node counts. The 4-node re-run (job 1541476, 12/11 04:47) is too noisy: the
# slowest of 3 repetitions is up to 1.97x the fastest (the pure-MPI baseline 6.0 /
# 4.2 / 3.1 ms), so its median ratios would show a 25 % gain from threads that the
# best-of-3 values do not have. 4 nodes keep the 2D rows of the first E6 run
# (spread <= 1.32x); the fix changes 2D only for ranks whose threads span both sockets.
E6_2D_RERUN_NODES = (1,)


def fig_e6(R, out, T):
    T.append("\n## E6 — MPI × OpenMP (ranks per node × threads per rank = 72)\n")
    T.append("| nodes | dim | " + " | ".join(f"{t} thr" for t in (1, 2, 3, 4, 6, 9, 12, 18, 24, 36, 72)) + " |")
    T.append("|---|---|" + "---|" * 11)
    fig, axs = plt.subplots(1, 2, figsize=(6.6, 2.7), sharey=True)
    for i, nodes in enumerate((1, 4)):
        e6 = load(f"{R}/e6_vanda_n{nodes}.csv")
        if nodes in E6_2D_RERUN_NODES:
            e6 = [r for r in e6 if r["dim"] == 3] + load(f"{R}/e6_2d_vanda_n{nodes}.csv")
        g = groups(e6, "dim", "threads")
        for ax, d in zip(axs, (3, 2)):
            th = sorted(k[1] for k in g if k[0] == d)
            t = np.array([med(g[(d, k)], "t_iter") for k in th])
            ax.plot(th, t / t[0], label=f"{nodes} node{'s' if nodes > 1 else ''} ({nodes*72} cores)", **style_line(i))
            T.append(f"| {nodes} | {d}D | " + " | ".join(f"{v*1e3:.2f}" for v in t) + " |")
    for ax, d in zip(axs, (3, 2)):
        p_axis(ax, [1, 2, 3, 4, 6, 9, 12, 18, 24, 36, 72], node_line=False,
               ticks=[1, 2, 4, 6, 9, 12, 18, 36, 72])
        ax.axhline(1, color=MUTED, lw=0.8)
        ax.set_xlabel("OpenMP threads per MPI rank")
        ax.set_title("3D, n = 512" if d == 3 else "2D, n = 8192")
    f2 = f"{R}/e6_2d_vanda_n4.csv"
    if os.path.exists(f2):   # not used for the line: see E6_2D_RERUN_NODES
        g4 = groups(load(f2), "threads")
        th = sorted(k[0] for k in g4)
        b = min(r["t_iter"] for r in g4[(1,)])
        axs[1].plot(th, [min(r["t_iter"] for r in g4[(t,)]) / b for t in th], ls="none",
                    marker="s", markerfacecolor="none", markeredgecolor=C[1], markersize=4.5,
                    markeredgewidth=0.9, label="4 nodes, re-run after\nfirst-touch fix (best of 3)")
        axs[1].legend(loc="upper left")
    axs[0].set_ylabel("time per iteration / pure MPI")
    axs[0].legend(loc="upper left")
    T.append("\n(times in ms per iteration, median of 3. 2D on 1 node: re-run after the "
             "first-touch fix; 2D on 4 nodes: first E6 run, because the 4-node re-run is too "
             "noisy (slowest / fastest repetition up to 1.97×).)")
    # the 4-node 2D re-run, for the record: median and best of 3
    f2 = f"{R}/e6_2d_vanda_n4.csv"
    if os.path.exists(f2):
        g = groups(load(f2), "threads")
        th = sorted(k[0] for k in g)
        T.append("\n4-node 2D re-run after the fix (not plotted): time per iteration / pure MPI")
        T.append("\n| statistic | " + " | ".join(f"{t} thr" for t in th) + " |")
        T.append("|---|" + "---|" * len(th))
        for name, fn in (("median", statistics.median), ("best of 3", min), ("max / min", None)):
            if fn is None:
                vals = [max(r["t_iter"] for r in g[(t,)]) / min(r["t_iter"] for r in g[(t,)]) for t in th]
            else:
                b = fn(r["t_iter"] for r in g[(1,)])
                vals = [fn(r["t_iter"] for r in g[(t,)]) / b for t in th]
            T.append(f"| {name} | " + " | ".join(f"{v:.2f}" for v in vals) + " |")
    save(fig, out, "F9_hybrid_mpi_openmp")


# Minimum memory traffic of one standard fused CG iteration, in vectors of N
# doubles: stencil q = A p (read p, write q) + x, r update (read x, r, p, q;
# write x, r) + p = r + beta p (read r, p; write p) = 11. Neighbour re-reads of
# the stencil are not counted, so the effective bandwidth is a lower bound.
CG_VECTORS = 11
A40_PEAK_GBS = 696.0      # NVIDIA A40, GDDR6
CPU_NODE_PEAK_GBS = 614.4  # 2 sockets x 8 channels x DDR5-4800 x 8 B (Xeon 8452Y node)


def eff_bw(N, t_iter):
    return CG_VECTORS * 8.0 * N / t_iter / 1e9


def fig_e7(R, out, T):
    # one 1-GPU job only (the 2-GPU job repeats the 1-GPU runs on another host; mixing
    # the two would put 6 runs from two machines into one median)
    f1 = sorted(glob.glob(f"{R}/e7_vanda_g1_*.csv"))[-1]
    e7 = [r for r in load(f1) if r["ranks"] == 1]
    gpu = groups(e7, "dim", "N")
    e1 = [r for r in load(f"{R}/e1_serial_vanda.csv") if r["mode"] == "fixed" and r["radius"] == 1]
    ser = groups(e1, "dim", "N")
    node = groups([r for r in load(f"{R}/e2_vanda_n1.csv") if r["ranks"] == CORES_PER_NODE
                   and r["threads"] == 1 and r["radius"] == 1], "dim", "N")

    # F10: one A40 vs N, compared with one CPU core (E1) and one CPU node (E2, 72 ranks)
    fig, axs = plt.subplots(1, 2, figsize=(6.6, 2.7))
    T.append(f"\n## E7 — one NVIDIA A40 (MPI + CUDA, 1 rank), 200 iterations, r = 1 ({os.path.basename(f1)})\n")
    T.append(f"Effective bandwidth = {CG_VECTORS} × 8 B × N / t_iter (minimum traffic of one CG "
             f"iteration); A40 peak {A40_PEAK_GBS:.0f} GB/s, CPU node peak {CPU_NODE_PEAK_GBS:.0f} GB/s.\n")
    T.append("| dim | n | N | GPU ms / iter | GPU GB/s (% of peak) | 1 core ms / iter | 72-core node ms / iter "
             "| GPU vs 1 core | GPU vs node |")
    T.append("|---|---|---|---|---|---|---|---|---|")
    for i, d in enumerate((2, 3)):
        Ns = sorted(k[1] for k in gpu if k[0] == d)
        t = np.array([med(gpu[(d, N)], "t_iter") for N in Ns])
        axs[0].plot(Ns, t, label=f"1 A40, {d}D", **style_line(i))
        bw = eff_bw(np.array(Ns, float), t)
        axs[1].plot(Ns, bw, label=f"1 A40, {d}D", **style_line(i))
        Ss = sorted(k[1] for k in ser if k[0] == d)
        ts = np.array([med(ser[(d, N)], "t_iter") for N in Ss])
        axs[0].plot(Ss, ts, label=f"1 CPU core, {d}D", **style_line(i + 2))
        for N, tg in zip(Ns, t):
            n = gpu[(d, N)][0]["nx"]
            t1 = med(ser[(d, N)], "t_iter") if (d, N) in ser else None
            tn = med(node[(d, N)], "t_iter") if (d, N) in node else None
            T.append(f"| {d}D | {n} | {N:.3g} | {tg*1e3:.3f} | {eff_bw(N, tg):.0f} "
                     f"({eff_bw(N, tg)/A40_PEAK_GBS*100:.0f} %) | "
                     + (f"{t1*1e3:.2f}" if t1 else "–") + " | " + (f"{tn*1e3:.2f}" if tn else "–") + " | "
                     + (f"{t1/tg:.1f}×" if t1 and t1 / tg < 10 else f"{t1/tg:.0f}×" if t1 else "–") + " | " + (f"{tn/tg:.2f}×" if tn else "–") + " |")
    # one CPU node (72 ranks, E2): a single size per dimension
    for j, d in enumerate((2, 3)):
        for (dd, N), rs in node.items():
            if dd != d:
                continue
            tn = med(rs, "t_iter")
            kw = dict(marker="*", ms=9, color=INK, mec="white", mew=0.6, ls="none", zorder=5)
            axs[0].plot([N], [tn], label="72-core CPU node" if j == 0 else None, **kw)
            axs[1].plot([N], [eff_bw(N, tn)], label="72-core CPU node" if j == 0 else None, **kw)
    for ax in axs:
        ax.set_xscale("log"); ax.set_xlabel("unknowns N")
    axs[0].set_yscale("log")
    axs[0].set_ylabel("time per CG iteration (s)")
    axs[0].set_title("Time per iteration vs N")
    h, l = axs[0].get_legend_handles_labels()
    order = sorted(range(len(l)), key=lambda k: ("node" in l[k], "CPU core" in l[k], l[k]))
    axs[0].legend([h[k] for k in order], [l[k] for k in order], loc="upper left", fontsize=6.5)
    axs[1].axhline(A40_PEAK_GBS, color=REF, lw=0.9, ls=(0, (4, 3)))
    axs[1].annotate(f"A40 peak {A40_PEAK_GBS:.0f} GB/s", (0.02, A40_PEAK_GBS), xycoords=("axes fraction", "data"),
                    xytext=(0, 3), textcoords="offset points", fontsize=7, color=INK2)
    axs[1].axhline(CPU_NODE_PEAK_GBS, color=REF, lw=0.9, ls=(0, (1, 2)))
    axs[1].annotate(f"CPU node peak {CPU_NODE_PEAK_GBS:.0f} GB/s", (0.02, CPU_NODE_PEAK_GBS),
                    xycoords=("axes fraction", "data"), xytext=(0, -3), textcoords="offset points",
                    va="top", fontsize=7, color=INK2)
    axs[1].set_ylim(0, 780)
    axs[1].set_ylabel("effective bandwidth (GB/s)")
    axs[1].set_title("Effective memory bandwidth")
    axs[1].legend(loc="lower right")
    fig.tight_layout()
    save(fig, out, "F10_gpu_vs_size")


def fig_e7_multi(R, out, T):
    """F11: 1 vs 2 GPUs of one node (strong and weak scaling), host staging vs CUDA-aware MPI,
    time per iteration split into parts. All runs of one job (same host)."""
    f2 = sorted(glob.glob(f"{R}/e7_vanda_g2_*.csv"))[-1]
    rows = load(f2)
    host = sorted({r["host"] for r in rows})
    g = groups(rows, "dim", "N", "ranks", "gpu_aware")
    # (panel, label, dim, N on 1 GPU, N on 2 GPUs)
    probs = [("strong", "3D 512³", 3, 512 ** 3, 512 ** 3),
             ("strong", "2D 16384²", 2, 16384 ** 2, 16384 ** 2),
             ("strong", "2D 8192²", 2, 8192 ** 2, 8192 ** 2),
             ("weak", "3D 256³ / GPU", 3, 256 ** 3, 2 * 256 ** 3),
             ("weak", "2D 8192² / GPU", 2, 8192 ** 2, 2 * 8192 ** 2)]
    cases = [("1 GPU", 1, 1), ("2 GPUs, host staging", 2, 0), ("2 GPUs, CUDA-aware", 2, 1)]

    def parts(rs):
        it = rs[0]["iters"]
        m = {k: med(rs, k) / it for k in ("t_compute", "t_halo", "t_bc", "t_allreduce")}
        return [m["t_compute"], m["t_halo"] - m["t_bc"], m["t_bc"], m["t_allreduce"]]

    pnames = ["compute", "halo exchange (pack, copies, MPI)", "boundary fill", "Allreduce"]
    T.append(f"\n## E7 — 1 vs 2 A40 of one node ({os.path.basename(f2)}, host {', '.join(host)}; "
             "the two GPUs sit on different CPU sockets, nvidia-smi topo: SYS, no P2P / NVLink)\n")
    T.append("t_iter in ms (median of 3, 200 iterations). Strong: S = t(1 GPU) / t(2 GPUs), E = S / 2. "
             "Weak (fixed size per GPU): E_w = t(1 GPU) / t(2 GPUs). Halo = t_halo per iteration "
             "(incl. boundary fill).\n")
    T.append("| problem | 1 GPU | 2 GPUs staging | 2 GPUs CUDA-aware | S or E_w staging | S or E_w aware "
             "| halo staging (ms) | halo aware (ms) |")
    T.append("|---|---|---|---|---|---|---|---|")
    fig, axs = plt.subplots(1, 2, figsize=(6.6, 2.9), sharey=True,
                            gridspec_kw={"width_ratios": [3, 2]})
    for ax, panel in zip(axs, ("strong", "weak")):
        pp = [p for p in probs if p[0] == panel]
        xt, xl = [], []
        for j, (_, lab, d, n1, n2) in enumerate(pp):
            base = med(g[(d, n1, 1, 1)], "t_iter")
            tt = []
            for c, (clab, p, aw) in enumerate(cases):
                rs = g[(d, n1 if p == 1 else n2, p, aw)]
                t = med(rs, "t_iter")
                tt.append((t, med(rs, "t_halo") / rs[0]["iters"]))
                x = j * 4 + c
                bottom = 0.0
                for i, v in enumerate(parts(rs)):
                    ax.bar(x, v / base, bottom=bottom, width=0.8, color=C[i], edgecolor="white",
                           linewidth=0.6, label=pnames[i] if (j == 0 and c == 0 and panel == "strong") else None)
                    bottom += v / base
            xt += [j * 4 + c for c in range(3)]
            # speedup (strong) or weak efficiency under each 2-GPU bar
            xl += ["1"] + [f"{name}\n{base / tt[c][0]:.2f}" for c, name in ((1, "2 st"), (2, "2 aw"))]
            ax.annotate(lab, (j * 4 + 1, 0), xycoords=("data", "axes fraction"), xytext=(0, -34),
                        textcoords="offset points", ha="center", va="top", fontsize=7, color=INK)
            T.append(f"| {lab} | {tt[0][0]*1e3:.2f} | {tt[1][0]*1e3:.2f} | {tt[2][0]*1e3:.2f} | "
                     f"{base/tt[1][0]:.2f} | {base/tt[2][0]:.2f} | {tt[1][1]*1e3:.3f} | {tt[2][1]*1e3:.3f} |")
        ax.set_xticks(xt, xl, fontsize=6.5)
        ax.grid(axis="x", visible=False)
        ideal = 0.5 if panel == "strong" else 1.0
        ax.axhline(ideal, color=REF, lw=0.9, ls=(0, (4, 3)), zorder=0,
                   label="ideal" if panel == "strong" else None)
        ax.set_title("Strong scaling: speedup S under bars" if panel == "strong"
                     else "Weak scaling: efficiency E_w under bars")
    axs[0].set_ylim(0, 1.2)
    axs[0].set_ylabel("time per iteration / 1 GPU")
    axs[0].legend(loc="upper center", bbox_to_anchor=(0.85, -0.33), ncol=5, fontsize=6.5)
    T.append("\n(GPU count: 1 = one A40; '2 st' = 2 GPUs with host staging (--gpu-aware 0), "
             "'2 aw' = 2 GPUs with CUDA-aware MPI (--gpu-aware 1).)")
    save(fig, out, "F11_gpu_scaling")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--results", default="results")
    ap.add_argument("--out", default="figures")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    T = ["# Tables for the report (generated by scripts/plot_results.py — do not edit)\n"]
    fig_e1(a.results, a.out, T)
    fig_v1(a.results, a.out, T)
    tables_validation(a.results, T)
    fig_e2(a.results, a.out, T)
    fig_e3(a.results, a.out, T)
    net = net_params(a.results, T)
    fig_breakdown(a.results, a.out, T, net)
    fig_e5(a.results, a.out, T)
    fig_e6(a.results, a.out, T)
    if glob.glob(f"{a.results}/e7_vanda_g1_*.csv"):
        fig_e7(a.results, a.out, T)
    if glob.glob(f"{a.results}/e7_vanda_g2_*.csv"):
        fig_e7_multi(a.results, a.out, T)
    with open(os.path.join(a.out, "tables.md"), "w") as f:
        f.write("\n".join(T) + "\n")
    print("wrote", os.path.join(a.out, "tables.md"))


if __name__ == "__main__":
    main()
