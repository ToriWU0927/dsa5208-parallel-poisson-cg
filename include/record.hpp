// record.hpp — one CSV line per solve, collected by the scripts in scripts/.
#pragma once

#include <cstdio>
#include <string>

#include <unistd.h>

namespace poisson {

struct RunRecord {
    std::string impl;   // serial | mpi | mpi_cuda
    std::string problem;  // mms | ones
    int dim = 0, radius = 0, nx = 0, ny = 0, nz = 0;
    long long N = 0, nnz = 0;
    int nodes = 1, ranks = 1, threads = 1, gpus = 0;
    std::string decomp = "none", halo = "none", cg, mode;
    int fuse = 1, gpu_aware = 0;
    double tol = 0.0;
    int iters = 0, converged = 0;
    double t_setup = 0, t_solve = 0, t_iter = 0, t_halo = 0, t_allreduce = 0, t_compute = 0;
    double rel_res = 0, true_rel_res = 0, err_inf = 0;  // err_inf is NaN unless --verify
    int repeat_id = 0;
    std::string host;   // machine the run was measured on
    double t_bc = 0, t_wait = 0;  // diagnostics: boundary fill (part of t_halo), barrier wait
};

inline std::string hostname() {
    char buf[256] = {0};
    if (gethostname(buf, sizeof(buf) - 1) != 0) return "unknown";
    return buf;
}

inline void print_csv_header(std::FILE* f) {
    std::fprintf(f,
                 "impl,problem,dim,radius,nx,ny,nz,N,nnz,nodes,ranks,threads,gpus,decomp,halo,cg,fuse,"
                 "gpu_aware,mode,tol,iters,converged,t_setup,t_solve,t_iter,t_halo,t_allreduce,"
                 "t_compute,rel_res,true_rel_res,err_inf,repeat_id,host,t_bc,t_wait\n");
}

inline void print_csv_row(std::FILE* f, const RunRecord& r) {
    std::fprintf(f,
                 "%s,%s,%d,%d,%d,%d,%d,%lld,%lld,%d,%d,%d,%d,%s,%s,%s,%d,%d,%s,%.3e,%d,%d,"
                 "%.6e,%.6e,%.6e,%.6e,%.6e,%.6e,%.6e,%.6e,%.6e,%d,%s,%.6e,%.6e\n",
                 r.impl.c_str(), r.problem.c_str(), r.dim, r.radius, r.nx, r.ny, r.nz, r.N, r.nnz, r.nodes, r.ranks,
                 r.threads, r.gpus, r.decomp.c_str(), r.halo.c_str(), r.cg.c_str(), r.fuse,
                 r.gpu_aware, r.mode.c_str(), r.tol, r.iters, r.converged, r.t_setup, r.t_solve,
                 r.t_iter, r.t_halo, r.t_allreduce, r.t_compute, r.rel_res, r.true_rel_res,
                 r.err_inf, r.repeat_id, r.host.c_str(), r.t_bc, r.t_wait);
    std::fflush(f);
}

}  // namespace poisson
