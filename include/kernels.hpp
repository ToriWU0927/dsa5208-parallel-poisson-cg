// kernels.hpp — CPU kernels on halo-padded arrays (serial and MPI+OpenMP).
//
// All loops run over the local interior only; halos are written only by
// apply_physical_bc() and by the halo exchange. OpenMP pragmas are ignored
// when the code is compiled without -fopenmp (serial build).
#pragma once

#include <memory>

#include "grid.hpp"

namespace poisson {

// Padded array. Memory is allocated without being touched so that the first
// write (fill_all, done inside an OpenMP region) places pages on the NUMA node
// of the thread that will later use them ("first touch").
class Field {
   public:
    Field() = default;
    explicit Field(const Grid& g) : n_(g.ntot), p_(new double[g.ntot]) {}
    double* data() { return p_.get(); }
    const double* data() const { return p_.get(); }
    long size() const { return n_; }

   private:
    long n_ = 0;
    std::unique_ptr<double[]> p_;
};

// x = v everywhere, including halos (parallel, first touch).
void fill_all(const Grid& g, double* x, double v);

// Fill halo cells that lie outside the physical domain:
//   distance 1 from the boundary face -> 0 (Dirichlet value)
//   distance 2 (r = 2 only)           -> -x(first interior point)  (odd reflection)
void apply_physical_bc(const Grid& g, double* x);

// y = A x on the index box b (x must have valid halos). Optionally also
// accumulates local dot products in the same pass:
//   ndot >= 1: sums[0] += (x, y)      ndot >= 2: sums[1] += (x, x)
void stencil(const Grid& g, const double* x, double* y, const Box& b, int ndot, double* sums);

// ---- vector operations on the local interior --------------------------------
double dot(const Grid& g, const double* a, const double* b);
void copy(const Grid& g, double* dst, const double* src);
void axpy(const Grid& g, double* y, double a, const double* x);   // y += a x
void xpay(const Grid& g, double* y, double b, const double* x);   // y = x + b y
void sub(const Grid& g, double* z, const double* a, const double* b);  // z = a - b

// Standard CG, fused: x += a p; r -= a q; returns local (r, r).
double cg_update_xr(const Grid& g, double* x, double* r, const double* p, const double* q,
                    double a);

// Single-reduction CG, fused: p = r + b p; s = w + b s; x += a p; r -= a s.
void cg1_update(const Grid& g, double* x, double* r, double* p, double* s, const double* w,
                double a, double b);

// max |x - u_exact| over the local interior.
double max_error(const Grid& g, const double* x);

}  // namespace poisson
