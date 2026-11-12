// cg.hpp — Conjugate Gradient for A x = b, generic over
//   * the "operator" Op that provides the matrix-vector product and the global
//     reductions, and
//   * the "vector backend" V that provides the vector operations on the local
//     interior and the type of the work vectors.
// The serial, MPI(+OpenMP) and MPI+CUDA programs share this code. The serial
// and MPI programs use HostVec (the CPU kernels of kernels.hpp); the CUDA
// program uses a GPU backend whose vectors live in device memory, so the CG
// loop never copies vectors between host and device (only scalars).
//
// V must provide (pointers refer to V::FieldT storage; dot products are LOCAL):
//   using FieldT;  fill_all(g, x, v); sub(g, z, a, b); copy(g, dst, src);
//   double dot(g, a, b); axpy(g, y, a, x); xpay(g, y, b, x);
//   double cg_update_xr(g, x, r, p, q, a); cg1_update(g, x, r, p, s, w, a, b)
//
// Op must provide:
//   void apply(double* x, double* y, int ndot, double* sums, Timers& t);
//       update the halo of x, then y = A x on the interior; if ndot >= 1 add
//       the LOCAL (x,y) to sums[0], if ndot >= 2 add the LOCAL (x,x) to sums[1]
//   void allreduce_sum(double* v, int n, Timers& t);   // in-place global sum
#pragma once

#include <cmath>
#include <initializer_list>

#include "grid.hpp"
#include "kernels.hpp"
#include "timer.hpp"

namespace poisson {

enum class CGVariant { Standard, SingleReduction };

struct CGParams {
    CGVariant variant = CGVariant::Standard;
    double tol = 1e-8;     // stop when ||r|| / ||b|| < tol (solve mode)
    int maxit = 100000;
    int fixed_iters = 0;   // > 0: run exactly this many iterations, ignore tol
    bool fuse = true;      // fused kernels (fewer passes over memory)
};

struct CGResult {
    int iters = 0;
    bool converged = false;
    double rel_res = 0.0;  // recursive residual ||r_k|| / ||b||
    double bnorm = 0.0;
};

// CPU vector backend: forwards to the kernels of kernels.hpp.
struct HostVec {
    using FieldT = Field;
    void fill_all(const Grid& g, double* x, double v) { poisson::fill_all(g, x, v); }
    void sub(const Grid& g, double* z, const double* a, const double* b) { poisson::sub(g, z, a, b); }
    void copy(const Grid& g, double* d, const double* s) { poisson::copy(g, d, s); }
    double dot(const Grid& g, const double* a, const double* b) { return poisson::dot(g, a, b); }
    void axpy(const Grid& g, double* y, double a, const double* x) { poisson::axpy(g, y, a, x); }
    void xpay(const Grid& g, double* y, double b, const double* x) { poisson::xpay(g, y, b, x); }
    double cg_update_xr(const Grid& g, double* x, double* r, const double* p, const double* q,
                        double a) {
        return poisson::cg_update_xr(g, x, r, p, q, a);
    }
    void cg1_update(const Grid& g, double* x, double* r, double* p, double* s, const double* w,
                    double a, double b) {
        poisson::cg1_update(g, x, r, p, s, w, a, b);
    }
};

namespace detail {
// Stopping test at the top of iteration k. Returns true to stop.
inline bool cg_stop(const CGParams& prm, int k, double gamma, double bnorm, CGResult& res) {
    res.rel_res = std::sqrt(gamma) / bnorm;
    res.converged = res.rel_res < prm.tol;
    if (gamma == 0.0) return true;                 // exact solution reached
    if (prm.fixed_iters > 0) return k >= prm.fixed_iters;
    return res.converged || k >= prm.maxit;
}
}  // namespace detail

// Work vectors are allocated by the caller so that their allocation and first
// touch are excluded from the solve time.
template <class F>
struct CGWorkT {
    F r, p, q, w, s;  // w, s only used by the single-reduction variant
    explicit CGWorkT(const Grid& g, CGVariant v) : r(g), p(g), q(g) {
        if (v == CGVariant::SingleReduction) {
            w = F(g);
            s = F(g);
        }
    }
};
using CGWork = CGWorkT<Field>;

// Initialise all work vectors (and their halos) to zero.
template <class V, class F>
void cg_reset(V& v, const Grid& g, CGWorkT<F>& wk) {
    for (F* f : {&wk.r, &wk.p, &wk.q, &wk.w, &wk.s})
        if (f->size() > 0) v.fill_all(g, f->data(), 0.0);
}
inline void cg_reset(const Grid& g, CGWork& wk) {
    HostVec v;
    cg_reset(v, g, wk);
}

// Standard CG: per iteration 1 apply + 2 allreduces.
template <class Op, class V>
CGResult cg_standard(Op& op, V& v, const Grid& g, double* x, const double* b,
                     CGWorkT<typename V::FieldT>& wk, const CGParams& prm, Timers& t) {
    double *r = wk.r.data(), *p = wk.p.data(), *q = wk.q.data();
    CGResult res;
    double sums[2] = {0.0, 0.0};

    // r = b - A x, p = r
    op.apply(x, q, 0, sums, t);
    double t0 = wtime();
    v.sub(g, r, b, q);
    v.copy(g, p, r);
    sums[0] = v.dot(g, r, r);
    sums[1] = v.dot(g, b, b);
    t.compute += wtime() - t0;
    op.allreduce_sum(sums, 2, t);
    double gamma = sums[0];
    res.bnorm = std::sqrt(sums[1]);
    if (res.bnorm == 0.0) res.bnorm = 1.0;

    int k = 0;
    while (!detail::cg_stop(prm, k, gamma, res.bnorm, res)) {
        // q = A p, (p, q)
        double pq = 0.0;
        op.apply(p, q, prm.fuse ? 1 : 0, &pq, t);
        if (!prm.fuse) {
            t0 = wtime();
            pq = v.dot(g, p, q);
            t.compute += wtime() - t0;
        }
        op.allreduce_sum(&pq, 1, t);
        const double alpha = gamma / pq;

        // x += alpha p, r -= alpha q, (r, r)
        double rr;
        t0 = wtime();
        if (prm.fuse) {
            rr = v.cg_update_xr(g, x, r, p, q, alpha);
        } else {
            v.axpy(g, x, alpha, p);
            v.axpy(g, r, -alpha, q);
            rr = v.dot(g, r, r);
        }
        t.compute += wtime() - t0;
        op.allreduce_sum(&rr, 1, t);

        const double beta = rr / gamma;
        gamma = rr;
        t0 = wtime();
        v.xpay(g, p, beta, r);  // p = r + beta p
        t.compute += wtime() - t0;
        ++k;
    }
    res.iters = k;
    return res;
}

// Single-reduction CG (Chronopoulos & Gear, 1989): the two inner products of
// one iteration, (r, r) and (r, A r), are computed together and reduced with a
// single allreduce. Mathematically equivalent to standard CG in exact
// arithmetic; needs two extra vectors (w = A r, s = A p).
template <class Op, class V>
CGResult cg_single_reduction(Op& op, V& v, const Grid& g, double* x, const double* b,
                             CGWorkT<typename V::FieldT>& wk, const CGParams& prm, Timers& t) {
    double *r = wk.r.data(), *p = wk.p.data(), *q = wk.q.data();
    double *w = wk.w.data(), *s = wk.s.data();
    CGResult res;

    // r = b - A x
    double dummy[2] = {0.0, 0.0};
    op.apply(x, q, 0, dummy, t);
    double t0 = wtime();
    v.sub(g, r, b, q);
    t.compute += wtime() - t0;

    // w = A r, (r, w), (r, r), (b, b)
    double sums[3] = {0.0, 0.0, 0.0};
    op.apply(r, w, prm.fuse ? 2 : 0, sums, t);
    t0 = wtime();
    if (!prm.fuse) {
        sums[0] = v.dot(g, r, w);
        sums[1] = v.dot(g, r, r);
    }
    sums[2] = v.dot(g, b, b);
    t.compute += wtime() - t0;
    op.allreduce_sum(sums, 3, t);
    double delta = sums[0], gamma = sums[1];
    res.bnorm = std::sqrt(sums[2]);
    if (res.bnorm == 0.0) res.bnorm = 1.0;

    double alpha = gamma / delta, beta = 0.0;  // p and s start at zero (cg_reset)
    int k = 0;
    while (!detail::cg_stop(prm, k, gamma, res.bnorm, res)) {
        // p = r + beta p; s = w + beta s; x += alpha p; r -= alpha s
        t0 = wtime();
        if (prm.fuse) {
            v.cg1_update(g, x, r, p, s, w, alpha, beta);
        } else {
            v.xpay(g, p, beta, r);
            v.xpay(g, s, beta, w);
            v.axpy(g, x, alpha, p);
            v.axpy(g, r, -alpha, s);
        }
        t.compute += wtime() - t0;

        // w = A r; (r, w) and (r, r) in ONE allreduce
        double d[2] = {0.0, 0.0};
        op.apply(r, w, prm.fuse ? 2 : 0, d, t);
        if (!prm.fuse) {
            t0 = wtime();
            d[0] = v.dot(g, r, w);
            d[1] = v.dot(g, r, r);
            t.compute += wtime() - t0;
        }
        op.allreduce_sum(d, 2, t);

        const double gamma_new = d[1];
        delta = d[0];
        beta = gamma_new / gamma;
        alpha = gamma_new / (delta - beta * gamma_new / alpha);
        gamma = gamma_new;
        ++k;
    }
    res.iters = k;
    return res;
}

template <class Op, class V>
CGResult cg_solve(Op& op, V& v, const Grid& g, double* x, const double* b,
                  CGWorkT<typename V::FieldT>& wk, const CGParams& prm, Timers& t) {
    if (prm.variant == CGVariant::SingleReduction)
        return cg_single_reduction(op, v, g, x, b, wk, prm, t);
    return cg_standard(op, v, g, x, b, wk, prm, t);
}
template <class Op>
CGResult cg_solve(Op& op, const Grid& g, double* x, const double* b, CGWork& wk,
                  const CGParams& prm, Timers& t) {
    HostVec v;
    return cg_solve(op, v, g, x, b, wk, prm, t);
}

// ||b - A x|| / ||b|| computed from scratch (validation V5). Overwrites wk.p, wk.q.
template <class Op, class V>
double true_relative_residual(Op& op, V& v, const Grid& g, double* x, const double* b,
                              CGWorkT<typename V::FieldT>& wk, double bnorm) {
    Timers t;
    double dummy[2] = {0.0, 0.0};
    double *q = wk.q.data(), *res = wk.p.data();
    op.apply(x, q, 0, dummy, t);
    v.sub(g, res, b, q);
    double rr = v.dot(g, res, res);
    op.allreduce_sum(&rr, 1, t);
    return std::sqrt(rr) / bnorm;
}
template <class Op>
double true_relative_residual(Op& op, const Grid& g, double* x, const double* b, CGWork& wk,
                              double bnorm) {
    HostVec v;
    return true_relative_residual(op, v, g, x, b, wk, bnorm);
}

}  // namespace poisson
