#pragma once

#ifdef USE_MPI
#include <mpi.h>
#else
#include <chrono>
#endif

namespace poisson {

inline double wtime() {
#ifdef USE_MPI
    return MPI_Wtime();
#else
    using clk = std::chrono::steady_clock;
    return std::chrono::duration<double>(clk::now().time_since_epoch()).count();
#endif
}

// Accumulated time per category during a solve (seconds, this process).
struct Timers {
    double halo = 0.0;       // halo exchange incl. physical BC fill and pack/unpack
    double allreduce = 0.0;  // global reductions (incl. waiting for slower ranks, see wait)
    double compute = 0.0;    // stencil + vector operations
    // diagnostics (not separate categories: not added to the three above twice)
    double bc = 0.0;         // physical boundary fill, a part of halo
    double wait = 0.0;       // --sync-timers 1 only: MPI_Barrier before each Allreduce
                             // (= waiting for the slowest rank); allreduce is then the
                             // reduction itself. Part of t_solve, not of the three above.
};

}  // namespace poisson
