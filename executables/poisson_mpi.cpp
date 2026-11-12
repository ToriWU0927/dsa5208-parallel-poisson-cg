// poisson_mpi — MPI + OpenMP implementation.
//
// Same problem, discretisation and CG as poisson_serial; the domain is split
// over a Cartesian process grid (decomp.hpp), halos are exchanged between face
// neighbours (halo.hpp) and inner products are summed with MPI_Allreduce.
// Inside each rank the kernels are parallelised with OpenMP; the number of
// threads is set with OMP_NUM_THREADS (1 = pure MPI). Only the master thread
// calls MPI (MPI_THREAD_FUNNELED).
//
// Rank 0 prints one CSV line per repetition (see record.hpp). Times:
//   t_solve      maximum over ranks (the slowest rank determines the run time)
//   t_halo, t_allreduce, t_compute   average over ranks
//
// Example:
//   mpirun -np 8 ./build/poisson_mpi --dim 3 --n 128 --problem mms --verify --header

#include <mpi.h>
#ifdef _OPENMP
#include <omp.h>
#endif

#include <cmath>
#include <cstdio>
#include <limits>
#include <stdexcept>

#include "cg.hpp"
#include "decomp.hpp"
#include "grid.hpp"
#include "io.hpp"
#include "kernels.hpp"
#include "mpi_op.hpp"
#include "options.hpp"
#include "problem.hpp"
#include "record.hpp"
#include "timer.hpp"

using namespace poisson;

static int run(int argc, char** argv) {
    Options opt = parse_options(argc, argv, "poisson_mpi");
    int world_rank = 0, world_size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    if (opt.local_n > 0) {  // weak scaling: fixed work per process
        int pd[3];
        process_grid(world_size, opt.dim, opt.decomp, pd);
        for (int d = 0; d < opt.dim; ++d) opt.ng[d] = opt.local_n * pd[d];
    }
    const CGParams prm = opt.cg_params();

    // ---- setup (timed separately, not part of the solve time) ----
    MPI_Barrier(MPI_COMM_WORLD);
    const double ts0 = wtime();
    Grid g;
    Decomposition dc;
    try {
        dc = make_decomposition(MPI_COMM_WORLD, opt.dim, opt.ng, opt.radius, opt.decomp, g);
    } catch (const std::exception& e) {
        if (world_rank == 0) std::fprintf(stderr, "error: %s\n", e.what());
        return 2;
    }
    const bool root = dc.rank == 0;
    Field x(g), b(g);
    CGWork wk(g, opt.cg);
    fill_all(g, x.data(), 0.0);
    fill_all(g, b.data(), 0.0);
    cg_reset(g, wk);
    fill_rhs(g, opt.problem, b.data());
    MpiOperator op(g, dc, opt.halo == "overlap" ? HaloMode::Overlap : HaloMode::Blocking,
                   opt.sync_timers);
    double t_setup = wtime() - ts0;
    MPI_Allreduce(MPI_IN_PLACE, &t_setup, 1, MPI_DOUBLE, MPI_MAX, dc.comm);

    int threads = 1;
#ifdef _OPENMP
    threads = omp_get_max_threads();
#endif

    RunRecord rec;
    rec.impl = "mpi";
    rec.problem = problem_name(opt.problem);
    rec.host = hostname();
    rec.dim = g.dim;
    rec.radius = g.r;
    rec.nx = g.ng[0];
    rec.ny = g.ng[1];
    rec.nz = g.ng[2];
    rec.N = g.global_N();
    rec.nnz = global_nnz(g.dim, g.r, g.ng);
    rec.nodes = count_nodes(dc.comm);
    rec.ranks = dc.size;
    rec.threads = threads;
    rec.decomp = dc.name;
    rec.halo = opt.halo;
    rec.cg = opt.cg_name();
    rec.mode = opt.mode_name();
    rec.fuse = prm.fuse ? 1 : 0;
    rec.tol = prm.tol;
    rec.t_setup = t_setup;

    if (root && opt.header) print_csv_header(stdout);

    for (int rep = 0; rep < opt.repeat; ++rep) {
        fill_all(g, x.data(), 0.0);  // initial guess x0 = 0
        cg_reset(g, wk);
        Timers t;
        MPI_Barrier(dc.comm);
        const double t0 = wtime();
        const CGResult res = cg_solve(op, g, x.data(), b.data(), wk, prm, t);
        double t_solve = wtime() - t0;

        double tmax = 0.0, tsum[5] = {t.halo, t.allreduce, t.compute, t.bc, t.wait};
        MPI_Reduce(&t_solve, &tmax, 1, MPI_DOUBLE, MPI_MAX, 0, dc.comm);
        MPI_Reduce(root ? MPI_IN_PLACE : tsum, tsum, 5, MPI_DOUBLE, MPI_SUM, 0, dc.comm);

        const double true_rel = true_relative_residual(op, g, x.data(), b.data(), wk, res.bnorm);
        double err = std::numeric_limits<double>::quiet_NaN();
        if (opt.verify) {
            err = max_error(g, x.data());
            MPI_Allreduce(MPI_IN_PLACE, &err, 1, MPI_DOUBLE, MPI_MAX, dc.comm);
        }

        if (root) {
            rec.iters = res.iters;
            rec.converged = res.converged ? 1 : 0;
            rec.t_solve = tmax;
            rec.t_iter = res.iters > 0 ? tmax / res.iters : 0.0;
            rec.t_halo = tsum[0] / dc.size;
            rec.t_allreduce = tsum[1] / dc.size;
            rec.t_compute = tsum[2] / dc.size;
            rec.t_bc = tsum[3] / dc.size;
            rec.t_wait = tsum[4] / dc.size;
            rec.rel_res = res.rel_res;
            rec.true_rel_res = true_rel;
            rec.err_inf = err;
            rec.repeat_id = rep;
            print_csv_row(stdout, rec);
            if (opt.verbose)
                std::fprintf(stderr,
                             "[mpi] %s ranks=%d threads=%d dim=%d r=%d N=%lld iters=%d conv=%d "
                             "t_solve=%.4fs t_iter=%.3es (halo %.3e, allreduce %.3e, compute %.3e) "
                             "rel_res=%.3e true_rel_res=%.3e err_inf=%.3e\n",
                             rec.decomp.c_str(), rec.ranks, rec.threads, rec.dim, rec.radius, rec.N,
                             rec.iters, rec.converged, rec.t_solve, rec.t_iter, rec.t_halo,
                             rec.t_allreduce, rec.t_compute, rec.rel_res, rec.true_rel_res,
                             rec.err_inf);
        }
    }
    if (!opt.save.empty()) write_solution_mpi(g, x.data(), opt.save, dc.comm);
    MPI_Comm_free(&dc.comm);
    return 0;
}

int main(int argc, char** argv) {
    int provided = 0;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
    if (provided < MPI_THREAD_FUNNELED) {
        std::fprintf(stderr, "error: MPI library does not provide MPI_THREAD_FUNNELED\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    const int rc = run(argc, argv);
    MPI_Finalize();
    return rc;
}
