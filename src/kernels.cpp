#include "kernels.hpp"

#include <algorithm>
#include <cmath>

#include "problem.hpp"

namespace poisson {

// Interior loops: k and j are distributed over OpenMP threads (collapse(2)),
// i is the contiguous, vectorised direction.
#define ROW_LOOP_HEAD                                \
    for (int k = 0; k < g.n[2]; ++k)                 \
        for (int j = 0; j < g.n[1]; ++j) {           \
            const long base = g.idx(0, j, k);        \
            const int ni = g.n[0];

void fill_all(const Grid& g, double* x, double v) {
    // First touch: the planes (3D) or rows (2D) go to the threads that later
    // work on them. 2D has a single "plane" (s[2] = 1), so it is split by rows.
    if (g.s[2] > 1) {
#pragma omp parallel for schedule(static)
        for (int k = 0; k < g.s[2]; ++k)
            for (int j = 0; j < g.s[1]; ++j) {
                double* row = x + (long)k * g.sz + (long)j * g.sy;
                for (int i = 0; i < g.s[0]; ++i) row[i] = v;
            }
    } else {
#pragma omp parallel for schedule(static)
        for (int j = 0; j < g.s[1]; ++j) {
            double* row = x + (long)j * g.sy;
            for (int i = 0; i < g.s[0]; ++i) row[i] = v;
        }
    }
}

void apply_physical_bc(const Grid& g, double* x) {
    for (int d = 0; d < g.dim; ++d) {
        // the two dimensions tangential to faces normal to d
        const int t1 = (d == 0) ? 1 : 0;
        const int t2 = (d == 2) ? 1 : 2;
        for (int side = 0; side < 2; ++side) {
            if (side == 0 ? !g.phys_lo[d] : !g.phys_hi[d]) continue;
            const int ghost1 = side == 0 ? -1 : g.n[d];         // on the boundary
            const int ghost2 = side == 0 ? -2 : g.n[d] + 1;     // outside (r = 2)
            const int inner = side == 0 ? 0 : g.n[d] - 1;       // first interior point
            // threaded: with few ranks and many threads per rank a serial face
            // fill would dominate the "halo" time (Vanda smoke run, 1 x 72)
#pragma omp parallel for collapse(2) schedule(static)
            for (int b = 0; b < g.n[t2]; ++b)
                for (int a = 0; a < g.n[t1]; ++a) {
                    int c[3];
                    c[t1] = a;
                    c[t2] = b;
                    c[d] = ghost1;
                    x[g.idx(c[0], c[1], c[2])] = 0.0;
                    if (g.r == 2) {
                        c[d] = inner;
                        const double v = x[g.idx(c[0], c[1], c[2])];
                        c[d] = ghost2;
                        x[g.idx(c[0], c[1], c[2])] = -v;
                    }
                }
        }
    }
}

template <int DIM, int R, int NDOT>
static void stencil_impl(const Grid& g, const double* __restrict x, double* __restrict y,
                         const Box& b, double* sums) {
    // 1D coefficients of -d^2/dx^2 (times h^2)
    constexpr double c0 = (R == 1) ? 2.0 : 30.0 / 12.0;
    constexpr double c1 = (R == 1) ? -1.0 : -16.0 / 12.0;
    constexpr double c2 = (R == 1) ? 0.0 : 1.0 / 12.0;
    const double ax = g.inv_h2[0], ay = g.inv_h2[1], az = g.inv_h2[2];
    const double diag = c0 * (ax + ay + (DIM == 3 ? az : 0.0));
    const double ax1 = c1 * ax, ay1 = c1 * ay, az1 = c1 * az;
    const double ax2 = c2 * ax, ay2 = c2 * ay, az2 = c2 * az;
    const long sy = g.sy, sz = g.sz;
    double s0 = 0.0, s1 = 0.0;

#pragma omp parallel for collapse(2) schedule(static) reduction(+ : s0, s1)
    for (int k = b.k0; k < b.k1; ++k)
        for (int j = b.j0; j < b.j1; ++j) {
            const long base = g.idx(0, j, k);
            const double* __restrict xc = x + base;
            double* __restrict yc = y + base;
#pragma omp simd reduction(+ : s0, s1)
            for (int i = b.i0; i < b.i1; ++i) {
                double v = diag * xc[i] + ax1 * (xc[i - 1] + xc[i + 1]) +
                           ay1 * (xc[i - sy] + xc[i + sy]);
                if constexpr (DIM == 3) v += az1 * (xc[i - sz] + xc[i + sz]);
                if constexpr (R == 2) {
                    v += ax2 * (xc[i - 2] + xc[i + 2]) + ay2 * (xc[i - 2 * sy] + xc[i + 2 * sy]);
                    if constexpr (DIM == 3) v += az2 * (xc[i - 2 * sz] + xc[i + 2 * sz]);
                }
                yc[i] = v;
                if constexpr (NDOT >= 1) s0 += xc[i] * v;
                if constexpr (NDOT >= 2) s1 += xc[i] * xc[i];
            }
        }
    if constexpr (NDOT >= 1) sums[0] += s0;
    if constexpr (NDOT >= 2) sums[1] += s1;
}

template <int DIM, int R>
static void stencil_dispatch_ndot(const Grid& g, const double* x, double* y, const Box& b,
                                  int ndot, double* sums) {
    switch (ndot) {
        case 0: stencil_impl<DIM, R, 0>(g, x, y, b, sums); break;
        case 1: stencil_impl<DIM, R, 1>(g, x, y, b, sums); break;
        default: stencil_impl<DIM, R, 2>(g, x, y, b, sums); break;
    }
}

void stencil(const Grid& g, const double* x, double* y, const Box& b, int ndot, double* sums) {
    if (b.i0 >= b.i1 || b.j0 >= b.j1 || b.k0 >= b.k1) return;
    if (g.dim == 2) {
        if (g.r == 1) stencil_dispatch_ndot<2, 1>(g, x, y, b, ndot, sums);
        else stencil_dispatch_ndot<2, 2>(g, x, y, b, ndot, sums);
    } else {
        if (g.r == 1) stencil_dispatch_ndot<3, 1>(g, x, y, b, ndot, sums);
        else stencil_dispatch_ndot<3, 2>(g, x, y, b, ndot, sums);
    }
}

double dot(const Grid& g, const double* __restrict a, const double* __restrict b) {
    double s = 0.0;
    #pragma omp parallel for collapse(2) schedule(static) reduction(+ : s)
    ROW_LOOP_HEAD
#pragma omp simd reduction(+ : s)
    for (int i = 0; i < ni; ++i) s += a[base + i] * b[base + i];
    }
    return s;
}

void copy(const Grid& g, double* __restrict dst, const double* __restrict src) {
    #pragma omp parallel for collapse(2) schedule(static)
    ROW_LOOP_HEAD
    std::copy(src + base, src + base + ni, dst + base);
    }
}

void axpy(const Grid& g, double* __restrict y, double a, const double* __restrict x) {
    #pragma omp parallel for collapse(2) schedule(static)
    ROW_LOOP_HEAD
#pragma omp simd
    for (int i = 0; i < ni; ++i) y[base + i] += a * x[base + i];
    }
}

void xpay(const Grid& g, double* __restrict y, double b, const double* __restrict x) {
    #pragma omp parallel for collapse(2) schedule(static)
    ROW_LOOP_HEAD
#pragma omp simd
    for (int i = 0; i < ni; ++i) y[base + i] = x[base + i] + b * y[base + i];
    }
}

void sub(const Grid& g, double* __restrict z, const double* __restrict a,
         const double* __restrict b) {
    #pragma omp parallel for collapse(2) schedule(static)
    ROW_LOOP_HEAD
#pragma omp simd
    for (int i = 0; i < ni; ++i) z[base + i] = a[base + i] - b[base + i];
    }
}

double cg_update_xr(const Grid& g, double* __restrict x, double* __restrict r,
                    const double* __restrict p, const double* __restrict q, double a) {
    double s = 0.0;
    #pragma omp parallel for collapse(2) schedule(static) reduction(+ : s)
    ROW_LOOP_HEAD
#pragma omp simd reduction(+ : s)
    for (int i = 0; i < ni; ++i) {
        const long m = base + i;
        x[m] += a * p[m];
        const double rv = r[m] - a * q[m];
        r[m] = rv;
        s += rv * rv;
    }
    }
    return s;
}

void cg1_update(const Grid& g, double* __restrict x, double* __restrict r, double* __restrict p,
                double* __restrict s, const double* __restrict w, double a, double b) {
    #pragma omp parallel for collapse(2) schedule(static)
    ROW_LOOP_HEAD
#pragma omp simd
    for (int i = 0; i < ni; ++i) {
        const long m = base + i;
        const double pv = r[m] + b * p[m];
        const double sv = w[m] + b * s[m];
        p[m] = pv;
        s[m] = sv;
        x[m] += a * pv;
        r[m] -= a * sv;
    }
    }
}

double max_error(const Grid& g, const double* x) {
    double e = 0.0;
#pragma omp parallel for collapse(2) schedule(static) reduction(max : e)
    for (int k = 0; k < g.n[2]; ++k)
        for (int j = 0; j < g.n[1]; ++j)
            for (int i = 0; i < g.n[0]; ++i)
                e = std::max(e, std::fabs(x[g.idx(i, j, k)] - exact_solution(g, i, j, k)));
    return e;
}

}  // namespace poisson
