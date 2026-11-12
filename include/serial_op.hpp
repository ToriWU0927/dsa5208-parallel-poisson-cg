// serial_op.hpp — the CG operator for a single process: the "halo update" is
// only the physical boundary fill, and reductions are no-ops.
#pragma once

#include "grid.hpp"
#include "kernels.hpp"
#include "timer.hpp"

namespace poisson {

class SerialOperator {
   public:
    explicit SerialOperator(const Grid& g) : g_(g) {}

    void apply(double* x, double* y, int ndot, double* sums, Timers& t) {
        double t0 = wtime();
        apply_physical_bc(g_, x);
        double t1 = wtime();
        stencil(g_, x, y, g_.interior(), ndot, sums);
        double t2 = wtime();
        t.halo += t1 - t0;
        t.bc += t1 - t0;
        t.compute += t2 - t1;
    }

    void allreduce_sum(double*, int, Timers&) {}

   private:
    const Grid& g_;
};

}  // namespace poisson
