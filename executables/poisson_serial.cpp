// poisson_serial — serial reference implementation (no MPI, no OpenMP).
//
// Solves -Laplace(u) = f on [0,1]^d (d = 2, 3) with homogeneous Dirichlet
// boundary conditions using a matrix-free finite-difference stencil and CG.
// Prints one CSV line per repetition to stdout (see record.hpp).
//
// Example:
//   ./build/poisson_serial --dim 3 --n 128 --radius 1 --problem mms --verify --header

#include <cmath>
#include <cstdio>
#include <limits>

#include "cg.hpp"
#include "grid.hpp"
#include "io.hpp"
#include "kernels.hpp"
#include "options.hpp"
#include "problem.hpp"
#include "record.hpp"
#include "serial_op.hpp"
#include "timer.hpp"

using namespace poisson;

int main(int argc, char** argv) {
    const Options opt = parse_options(argc, argv, "poisson_serial");
    const CGParams prm = opt.cg_params();

    // ---- setup (timed separately, not part of the solve time) ----
    const double ts0 = wtime();
    const Grid g = make_serial_grid(opt.dim, opt.radius, opt.ng);
    Field x(g), b(g);
    CGWork wk(g, opt.cg);
    fill_all(g, x.data(), 0.0);
    fill_all(g, b.data(), 0.0);
    cg_reset(g, wk);
    fill_rhs(g, opt.problem, b.data());
    const double t_setup = wtime() - ts0;

    SerialOperator op(g);

    RunRecord rec;
    rec.impl = "serial";
    rec.problem = problem_name(opt.problem);
    rec.host = hostname();
    rec.dim = g.dim;
    rec.radius = g.r;
    rec.nx = g.ng[0];
    rec.ny = g.ng[1];
    rec.nz = g.ng[2];
    rec.N = g.global_N();
    rec.nnz = global_nnz(g.dim, g.r, g.ng);
    rec.cg = opt.cg_name();
    rec.mode = opt.mode_name();
    rec.fuse = prm.fuse ? 1 : 0;
    rec.tol = prm.tol;
    rec.t_setup = t_setup;

    if (opt.header) print_csv_header(stdout);

    for (int rep = 0; rep < opt.repeat; ++rep) {
        fill_all(g, x.data(), 0.0);  // initial guess x0 = 0
        cg_reset(g, wk);
        Timers t;
        const double t0 = wtime();
        const CGResult res = cg_solve(op, g, x.data(), b.data(), wk, prm, t);
        const double t_solve = wtime() - t0;

        rec.iters = res.iters;
        rec.converged = res.converged ? 1 : 0;
        rec.t_solve = t_solve;
        rec.t_iter = res.iters > 0 ? t_solve / res.iters : 0.0;
        rec.t_halo = t.halo;
        rec.t_allreduce = t.allreduce;
        rec.t_compute = t.compute;
        rec.t_bc = t.bc;
        rec.rel_res = res.rel_res;
        rec.true_rel_res = true_relative_residual(op, g, x.data(), b.data(), wk, res.bnorm);
        rec.err_inf = opt.verify ? max_error(g, x.data())
                                 : std::numeric_limits<double>::quiet_NaN();
        rec.repeat_id = rep;
        print_csv_row(stdout, rec);

        if (opt.verbose)
            std::fprintf(stderr,
                         "[serial] dim=%d r=%d N=%lld iters=%d conv=%d t_solve=%.4fs "
                         "t_iter=%.3es rel_res=%.3e true_rel_res=%.3e err_inf=%.3e\n",
                         rec.dim, rec.radius, rec.N, rec.iters, rec.converged, rec.t_solve,
                         rec.t_iter, rec.rel_res, rec.true_rel_res, rec.err_inf);
    }
    if (!opt.save.empty()) write_solution(g, x.data(), opt.save);
    return 0;
}
