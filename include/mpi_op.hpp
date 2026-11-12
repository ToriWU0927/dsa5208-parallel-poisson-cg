// mpi_op.hpp — the CG operator for MPI(+OpenMP): matrix-vector product with
// halo exchange, and global sums with MPI_Allreduce. Plugs into cg.hpp.
//
// Halo modes:
//   block    exchange all halos (blocking), then apply the stencil everywhere
//   overlap  start the exchange, apply the stencil on the INNER points (those
//            at distance >= r from every face, which need no halo data), wait
//            for the exchange, then apply the stencil on the boundary shell
#pragma once

#include <mpi.h>

#include <algorithm>
#include <vector>

#include "decomp.hpp"
#include "grid.hpp"
#include "halo.hpp"
#include "kernels.hpp"
#include "timer.hpp"

namespace poisson {

enum class HaloMode { Blocking, Overlap };

class MpiOperator {
   public:
    MpiOperator(const Grid& g, const Decomposition& dc, HaloMode mode, bool sync_timers = false)
        : g_(g), dc_(dc), mode_(mode), sync_(sync_timers), halo_(g, dc) {
        split_boxes();
    }

    void apply(double* x, double* y, int ndot, double* sums, Timers& t) {
        if (mode_ == HaloMode::Blocking) {
            double t0 = wtime();
            apply_physical_bc(g_, x);
            const double tb = wtime();
            halo_.exchange(x);
            double t1 = wtime();
            stencil(g_, x, y, g_.interior(), ndot, sums);
            t.halo += t1 - t0;
            t.bc += tb - t0;
            t.compute += wtime() - t1;
        } else {
            double t0 = wtime();
            apply_physical_bc(g_, x);
            t.bc += wtime() - t0;
            halo_.begin(x);
            double t1 = wtime();
            stencil(g_, x, y, inner_, ndot, sums);
            double t2 = wtime();
            halo_.end();
            double t3 = wtime();
            for (const Box& b : shell_) stencil(g_, x, y, b, ndot, sums);
            double t4 = wtime();
            t.halo += (t1 - t0) + (t3 - t2);
            t.compute += (t2 - t1) + (t4 - t3);
        }
    }

    void allreduce_sum(double* v, int n, Timers& t) {
        if (sync_) {  // diagnostics: wait for the slowest rank first, timed separately
            const double tw = wtime();
            MPI_Barrier(dc_.comm);
            t.wait += wtime() - tw;
        }
        double t0 = wtime();
        MPI_Allreduce(MPI_IN_PLACE, v, n, MPI_DOUBLE, MPI_SUM, dc_.comm);
        t.allreduce += wtime() - t0;
    }

   private:
    // inner_ = points at distance >= r from every face; shell_ = the rest,
    // as at most 2*dim disjoint boxes.
    void split_boxes() {
        int a[3], b[3];  // inner range [a, b) per dimension
        for (int d = 0; d < 3; ++d) {
            const int n = g_.n[d], r = (d < g_.dim) ? g_.r : 0;
            a[d] = std::min(r, n);
            b[d] = std::max(a[d], n - r);
        }
        inner_ = Box{a[0], b[0], a[1], b[1], a[2], b[2]};
        for (int d = 0; d < g_.dim; ++d) {
            for (int side = 0; side < 2; ++side) {
                int lo[3], hi[3];
                for (int e = 0; e < 3; ++e) {
                    if (e < d) { lo[e] = a[e]; hi[e] = b[e]; }        // already covered outside
                    else { lo[e] = 0; hi[e] = g_.n[e]; }
                }
                if (side == 0) { lo[d] = 0; hi[d] = a[d]; }
                else { lo[d] = b[d]; hi[d] = g_.n[d]; }
                Box bx{lo[0], hi[0], lo[1], hi[1], lo[2], hi[2]};
                if (bx.i0 < bx.i1 && bx.j0 < bx.j1 && bx.k0 < bx.k1) shell_.push_back(bx);
            }
        }
    }

    const Grid& g_;
    const Decomposition& dc_;
    HaloMode mode_;
    bool sync_;
    HaloExchanger halo_;
    Box inner_{};
    std::vector<Box> shell_;
};

}  // namespace poisson
