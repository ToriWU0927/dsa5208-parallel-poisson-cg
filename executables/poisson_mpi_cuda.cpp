// poisson_mpi_cuda — MPI + CUDA implementation (one GPU per MPI rank).
//
// Same problem, discretisation, domain decomposition and CG as poisson_mpi;
// all CG vectors live in GPU memory (gpu.hpp), the kernels run on the GPU,
// and halos are exchanged between face neighbours either with CUDA-aware MPI
// (--gpu-aware 1, device pointers passed to MPI) or through host buffers
// (--gpu-aware 0, "host staging"); see cuda_op.hpp.
//
// Each rank uses GPU (node-local rank mod #visible GPUs). Several ranks may
// share a GPU (correctness tests only).
//
// Rank 0 prints one CSV line per repetition (record.hpp), impl = mpi_cuda,
// gpus = number of GPUs in use. Times as in poisson_mpi (t_solve: max over ranks;
// components: average over ranks).
//
// Example:
//   mpirun -np 2 ./build/poisson_mpi_cuda --dim 3 --n 256 --problem ones --fixed-iters 200 --header

#include <mpi.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <limits>
#include <stdexcept>

#include "cg.hpp"
#include "cuda_op.hpp"
#include "decomp.hpp"
#include "gpu.hpp"
#include "grid.hpp"
#include "io.hpp"
#include "kernels.hpp"
#include "options.hpp"
#include "problem.hpp"
#include "record.hpp"
#include "timer.hpp"

using namespace poisson;

static int run(int argc, char** argv) {
    Options opt = parse_options(argc, argv, "poisson_mpi_cuda");
    int world_rank = 0, world_size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    if (opt.halo != "block") {
        if (world_rank == 0)
            std::fprintf(stderr, "error: poisson_mpi_cuda implements --halo block only\n");
        return 2;
    }
    if (opt.local_n > 0) {  // weak scaling: fixed work per GPU
        int pd[3];
        process_grid(world_size, opt.dim, opt.decomp, pd);
        for (int d = 0; d < opt.dim; ++d) opt.ng[d] = opt.local_n * pd[d];
    }
    const CGParams prm = opt.cg_params();

    // one GPU per rank: node-local rank -> device
    MPI_Comm node;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, world_rank, MPI_INFO_NULL, &node);
    int local_rank = 0, local_size = 1;
    MPI_Comm_rank(node, &local_rank);
    MPI_Comm_size(node, &local_size);
    MPI_Comm_free(&node);
    int ndev = 0;
    try {
        ndev = gpu::select_device(local_rank);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "error (rank %d): %s\n", world_rank, e.what());
        return 2;
    }
    // GPUs in use: on every node min(ranks on the node, visible devices); several ranks
    // share a GPU only in the correctness tests (V3)
    int gpus_used = local_rank == 0 ? std::min(local_size, ndev) : 0;
    MPI_Allreduce(MPI_IN_PLACE, &gpus_used, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);

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
    gpu::GpuVec vec(g);
    gpu::DeviceField x(g), b(g);
    CGWorkT<gpu::DeviceField> wk(g, opt.cg);
    {   // right-hand side: computed on the host, copied once
        Field hb(g);
        fill_all(g, hb.data(), 0.0);
        fill_rhs(g, opt.problem, hb.data());
        gpu::to_device(b.data(), hb.data(), g.ntot);
    }
    vec.fill_all(g, x.data(), 0.0);
    cg_reset(vec, g, wk);
    CudaOperator op(g, dc, vec, opt.gpu_aware != 0, opt.sync_timers);
    {   // warm-up: first kernel launches and first CUDA-aware MPI transfers
        Timers tw;
        double dummy[2] = {0.0, 0.0};
        op.apply(b.data(), wk.q.data(), 1, dummy, tw);
        op.allreduce_sum(dummy, 1, tw);
    }
    double t_setup = wtime() - ts0;
    MPI_Allreduce(MPI_IN_PLACE, &t_setup, 1, MPI_DOUBLE, MPI_MAX, dc.comm);

    RunRecord rec;
    rec.impl = "mpi_cuda";
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
    rec.threads = 1;
    rec.gpus = gpus_used;
    rec.gpu_aware = opt.gpu_aware != 0 ? 1 : 0;
    rec.decomp = dc.name;
    rec.halo = opt.halo;
    rec.cg = opt.cg_name();
    rec.mode = opt.mode_name();
    rec.fuse = prm.fuse ? 1 : 0;
    rec.tol = prm.tol;
    rec.t_setup = t_setup;

    if (root && opt.verbose)
        std::fprintf(stderr, "[mpi_cuda] %d rank(s), %d visible GPU(s) per node, device 0 of rank 0: %s\n",
                     dc.size, ndev, gpu::device_name());
    if (root && opt.header) print_csv_header(stdout);

    for (int rep = 0; rep < opt.repeat; ++rep) {
        vec.fill_all(g, x.data(), 0.0);  // initial guess x0 = 0
        cg_reset(vec, g, wk);
        Timers t;
        MPI_Barrier(dc.comm);
        const double t0 = wtime();
        const CGResult res = cg_solve(op, vec, g, x.data(), b.data(), wk, prm, t);
        double t_solve = wtime() - t0;

        double tmax = 0.0, tsum[5] = {t.halo, t.allreduce, t.compute, t.bc, t.wait};
        MPI_Reduce(&t_solve, &tmax, 1, MPI_DOUBLE, MPI_MAX, 0, dc.comm);
        MPI_Reduce(root ? MPI_IN_PLACE : tsum, tsum, 5, MPI_DOUBLE, MPI_SUM, 0, dc.comm);

        const double true_rel =
            true_relative_residual(op, vec, g, x.data(), b.data(), wk, res.bnorm);
        double err = std::numeric_limits<double>::quiet_NaN();
        if (opt.verify) {
            Field hx(g);
            gpu::to_host(hx.data(), x.data(), g.ntot);
            err = max_error(g, hx.data());
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
                             "[mpi_cuda] %s gpus=%d aware=%d dim=%d r=%d N=%lld iters=%d conv=%d "
                             "t_solve=%.4fs t_iter=%.3es (halo %.3e, allreduce %.3e, compute %.3e) "
                             "rel_res=%.3e true_rel_res=%.3e err_inf=%.3e\n",
                             rec.decomp.c_str(), rec.gpus, rec.gpu_aware, rec.dim, rec.radius,
                             rec.N, rec.iters, rec.converged, rec.t_solve, rec.t_iter, rec.t_halo,
                             rec.t_allreduce, rec.t_compute, rec.rel_res, rec.true_rel_res,
                             rec.err_inf);
        }
    }
    if (!opt.save.empty()) {
        Field hx(g);
        gpu::to_host(hx.data(), x.data(), g.ntot);
        write_solution_mpi(g, hx.data(), opt.save, dc.comm);
    }
    MPI_Comm_free(&dc.comm);
    return 0;
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rc = 0;
    try {
        rc = run(argc, argv);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    MPI_Finalize();
    return rc;
}
