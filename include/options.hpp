// options.hpp — command-line options shared by all executables.
#pragma once

#include <string>

#include "cg.hpp"
#include "problem.hpp"

namespace poisson {

struct Options {
    int dim = 3;
    Problem problem = Problem::MMS;
    int ng[3] = {0, 0, 0};   // --n sets all, --nx/--ny/--nz individually
    int local_n = 0;         // MPI only: > 0 -> global n = local_n * processes per dimension (weak scaling)
    int radius = 1;
    double tol = 1e-8;
    int maxit = 0;           // 0 -> 10 * max(ng)
    int fixed_iters = 0;     // > 0 -> fixed-iteration mode
    CGVariant cg = CGVariant::Standard;
    bool fuse = true;
    int repeat = 3;
    bool verify = false;     // compute max error against the exact solution
    bool header = false;     // print the CSV header line first
    bool verbose = false;    // human-readable summary on stderr
    std::string save;        // write the final solution to this file (validation only)
    // used by the parallel executables
    std::string halo = "block";   // block | overlap
    std::string decomp = "auto";  // auto | slab
    int gpu_aware = 1;
    bool sync_timers = false;  // MPI: timed barrier before each Allreduce (diagnostics)

    CGParams cg_params() const;
    const char* cg_name() const { return cg == CGVariant::Standard ? "std" : "single"; }
    const char* mode_name() const { return fixed_iters > 0 ? "fixed" : "solve"; }
};

// Parses argv; prints usage and exits on --help or on an error.
Options parse_options(int argc, char** argv, const char* program);

}  // namespace poisson
