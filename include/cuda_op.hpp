// cuda_op.hpp — the CG operator for MPI + CUDA (one GPU per MPI rank).
// Plugs into cg.hpp together with the GPU vector backend gpu::GpuVec.
//
// apply(x, y): physical boundary fill on the GPU, halo exchange, stencil on
// the GPU. Halo exchange (face neighbours only, as on the CPU):
//   1. pack every face that has a neighbour into a contiguous device buffer
//      (one small kernel per face; x faces are strided in memory)
//   2. exchange the buffers with MPI_Sendrecv, per dimension and direction
//        gpu_aware = 1: device pointers are passed to MPI (CUDA-aware MPI;
//                       the MPI library moves GPU data itself)
//        gpu_aware = 0: host staging — cudaMemcpy device -> pinned host
//                       buffer, MPI on host memory, cudaMemcpy back
//   3. unpack the received buffers into the halo of x
// Only scalars (the local dot products) leave the GPU in the CG loop; they
// are summed with MPI_Allreduce on the host.
// The halo time t.halo includes boundary fill, pack, copies, MPI and unpack.
#pragma once

#include <mpi.h>

#include "decomp.hpp"
#include "gpu.hpp"
#include "grid.hpp"
#include "timer.hpp"

namespace poisson {

class CudaOperator {
   public:
    CudaOperator(const Grid& g, const Decomposition& dc, gpu::GpuVec& v, bool gpu_aware,
                 bool sync_timers = false)
        : g_(g), dc_(dc), v_(v), aware_(gpu_aware), sync_(sync_timers) {
        for (int d = 0; d < g.dim; ++d) {
            const long n = gpu::face_size(g, d);
            for (int s = 0; s < 2; ++s) {
                dsend_[d][s] = gpu::alloc_device(n);
                drecv_[d][s] = gpu::alloc_device(n);
                if (!aware_) {
                    hsend_[d][s] = gpu::alloc_pinned(n);
                    hrecv_[d][s] = gpu::alloc_pinned(n);
                }
            }
        }
    }
    ~CudaOperator() {
        for (int d = 0; d < 3; ++d)
            for (int s = 0; s < 2; ++s) {
                gpu::free_device(dsend_[d][s]);
                gpu::free_device(drecv_[d][s]);
                gpu::free_pinned(hsend_[d][s]);
                gpu::free_pinned(hrecv_[d][s]);
            }
    }
    CudaOperator(const CudaOperator&) = delete;
    CudaOperator& operator=(const CudaOperator&) = delete;

    void apply(double* x, double* y, int ndot, double* sums, Timers& t) {
        const double t0 = wtime();
        v_.apply_physical_bc(g_, x);   // returns after the kernels have finished
        const double tb = wtime();
        exchange(x);
        const double t1 = wtime();
        v_.stencil(g_, x, y, g_.interior(), ndot, sums);
        t.halo += t1 - t0;
        t.bc += tb - t0;
        t.compute += wtime() - t1;
    }

    void allreduce_sum(double* v, int n, Timers& t) {
        if (sync_) {  // diagnostics: wait for the slowest rank first, timed separately
            const double tw = wtime();
            MPI_Barrier(dc_.comm);
            t.wait += wtime() - tw;
        }
        const double t0 = wtime();
        MPI_Allreduce(MPI_IN_PLACE, v, n, MPI_DOUBLE, MPI_SUM, dc_.comm);
        t.allreduce += wtime() - t0;
    }

   private:
    bool has(int d, int s) const {
        return (s == 0 ? dc_.nbr_lo[d] : dc_.nbr_hi[d]) != MPI_PROC_NULL;
    }
    static int tag(int d, int towards_hi) { return 2 * d + towards_hi; }

    void exchange(double* x) {
        // 1. pack the first / last r interior layers (local coordinates 0 and n - r)
        bool any = false;
        for (int d = 0; d < g_.dim; ++d)
            for (int s = 0; s < 2; ++s)
                if (has(d, s)) {
                    v_.pack(g_, x, d, s == 0 ? 0 : g_.n[d] - g_.r, dsend_[d][s]);
                    any = true;
                }
        if (!any) return;  // single process: nothing to exchange
        gpu::sync();
        if (!aware_)
            for (int d = 0; d < g_.dim; ++d)
                for (int s = 0; s < 2; ++s)
                    if (has(d, s)) gpu::to_host(hsend_[d][s], dsend_[d][s], gpu::face_size(g_, d));

        // 2. same message pattern as the CPU halo exchange (halo.cpp)
        for (int d = 0; d < g_.dim; ++d) {
            const int n = (int)gpu::face_size(g_, d);
            double* const* sb = aware_ ? dsend_[d] : hsend_[d];
            double* const* rb = aware_ ? drecv_[d] : hrecv_[d];
            // send high face up, receive low halo from below
            MPI_Sendrecv(sb[1], n, MPI_DOUBLE, dc_.nbr_hi[d], tag(d, 1),
                         rb[0], n, MPI_DOUBLE, dc_.nbr_lo[d], tag(d, 1), dc_.comm, MPI_STATUS_IGNORE);
            // send low face down, receive high halo from above
            MPI_Sendrecv(sb[0], n, MPI_DOUBLE, dc_.nbr_lo[d], tag(d, 0),
                         rb[1], n, MPI_DOUBLE, dc_.nbr_hi[d], tag(d, 0), dc_.comm, MPI_STATUS_IGNORE);
        }

        // 3. unpack into the halos (local coordinates -r and n)
        for (int d = 0; d < g_.dim; ++d)
            for (int s = 0; s < 2; ++s)
                if (has(d, s)) {
                    if (!aware_) gpu::to_device(drecv_[d][s], hrecv_[d][s], gpu::face_size(g_, d));
                    v_.unpack(g_, x, d, s == 0 ? -g_.r : g_.n[d], drecv_[d][s]);
                }
        gpu::sync();
    }

    const Grid& g_;
    const Decomposition& dc_;
    gpu::GpuVec& v_;
    bool aware_;
    bool sync_;
    double* dsend_[3][2] = {};
    double* drecv_[3][2] = {};
    double* hsend_[3][2] = {};
    double* hrecv_[3][2] = {};
};

}  // namespace poisson
