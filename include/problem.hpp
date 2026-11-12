// problem.hpp — test problems (synthetic data, generated from formulas).
//
// Problem "mms" (method of manufactured solutions), used for validation:
//   u(x) = prod_d g(x_d),   g(t) = sin(pi t) / (2 - cos(pi t))
//   f(x) = -Laplace(u) = sum_d m(x_d) prod_{e != d} g(x_e),
//   m(t) = -g''(t) = 2 pi^2 sin(pi t) (1 + cos(pi t)) / (2 - cos(pi t))^3
// g vanishes at t = 0 and t = 1 and is odd about both points, so the odd
// reflection used by the 4th-order stencil (r = 2) is exact. Unlike
// sin(pi x) sin(pi y), u is NOT an eigenvector of the discrete operator
// (its sine series is infinite), so CG needs a non-trivial number of iterations.
//
// Problem "ones": f = 1 (uniform source). All Fourier modes are present, so the
// CG iteration count grows like O(n); used for the performance/scaling runs.
// There is no closed-form solution, so no error against an exact solution.
//
// Grid points: x_d = (global index + 1) * h_d.
#pragma once

#include <cmath>
#include <string>

#include "grid.hpp"

namespace poisson {

enum class Problem { MMS, Ones };

inline const char* problem_name(Problem p) { return p == Problem::MMS ? "mms" : "ones"; }

namespace detail {
constexpr double kPi = 3.14159265358979323846;
inline double g1(double t) { return std::sin(kPi * t) / (2.0 - std::cos(kPi * t)); }
inline double m1(double t) {
    const double c = std::cos(kPi * t), d = 2.0 - c;
    return 2.0 * kPi * kPi * std::sin(kPi * t) * (1.0 + c) / (d * d * d);
}
inline void coords(const Grid& g, int i, int j, int k, double x[3]) {
    x[0] = (g.off[0] + i + 1) * g.h[0];
    x[1] = (g.off[1] + j + 1) * g.h[1];
    x[2] = (g.off[2] + k + 1) * g.h[2];
}
}  // namespace detail

// Exact solution of problem "mms" at local interior point (i, j, k).
inline double exact_solution(const Grid& g, int i, int j, int k) {
    double x[3];
    detail::coords(g, i, j, k, x);
    double u = detail::g1(x[0]) * detail::g1(x[1]);
    if (g.dim == 3) u *= detail::g1(x[2]);
    return u;
}

// b = f on the local interior (halos untouched).
inline void fill_rhs(const Grid& g, Problem p, double* b) {
#pragma omp parallel for collapse(2) schedule(static)
    for (int k = 0; k < g.n[2]; ++k)
        for (int j = 0; j < g.n[1]; ++j)
            for (int i = 0; i < g.n[0]; ++i) {
                double f = 1.0;
                if (p == Problem::MMS) {
                    double x[3], gv[3], mv[3];
                    detail::coords(g, i, j, k, x);
                    for (int d = 0; d < g.dim; ++d) {
                        gv[d] = detail::g1(x[d]);
                        mv[d] = detail::m1(x[d]);
                    }
                    if (g.dim == 2) f = mv[0] * gv[1] + gv[0] * mv[1];
                    else f = mv[0] * gv[1] * gv[2] + gv[0] * mv[1] * gv[2] + gv[0] * gv[1] * mv[2];
                }
                b[g.idx(i, j, k)] = f;
            }
}

}  // namespace poisson
