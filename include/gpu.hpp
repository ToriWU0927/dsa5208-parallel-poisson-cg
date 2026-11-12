// gpu.hpp — CUDA backend: device memory, kernels and the GPU vector backend
// for cg.hpp. Implemented in src/gpu.cu (compiled by nvcc); this header uses
// plain C++ types only, so host code compiled by the MPI C++ compiler can
// include it without CUDA headers.
//
// Layout and semantics are those of the CPU code (grid.hpp, kernels.hpp):
// halo-padded arrays, x index fastest, all vector operations on the local
// interior only. Every kernel runs on the default stream and every wrapper
// returns after the kernel has finished (cudaDeviceSynchronize), so that the
// CPU-side timers (Timers in timer.hpp) measure GPU time correctly.
//
// Reductions (dot products, fused stencil + dot) are deterministic: each
// thread block writes one partial sum, and a second kernel adds the partial
// sums in a fixed order. The result for a given grid and GPU does not depend
// on thread scheduling (unlike atomicAdd accumulation).
#pragma once

#include "grid.hpp"

namespace poisson {
namespace gpu {

// Select the GPU of this process: device (local_rank mod #devices). Returns
// the number of visible devices; throws if there is none.
int select_device(int local_rank);
// Name of the current device (for the log).
const char* device_name();

// ---- memory -----------------------------------------------------------------
double* alloc_device(long n);
void free_device(double* p);
double* alloc_pinned(long n);          // page-locked host memory (fast staging copies)
void free_pinned(double* p);
void to_device(double* dst, const double* src, long n);
void to_host(double* dst, const double* src, long n);
void sync();

// Halo-padded array in device memory (the device counterpart of Field).
class DeviceField {
   public:
    DeviceField() = default;
    explicit DeviceField(const Grid& g) : n_(g.ntot), p_(alloc_device(g.ntot)) {}
    ~DeviceField() { free_device(p_); }
    DeviceField(DeviceField&& o) noexcept : n_(o.n_), p_(o.p_) { o.n_ = 0; o.p_ = nullptr; }
    DeviceField& operator=(DeviceField&& o) noexcept {
        if (this != &o) {
            free_device(p_);
            n_ = o.n_; p_ = o.p_;
            o.n_ = 0; o.p_ = nullptr;
        }
        return *this;
    }
    DeviceField(const DeviceField&) = delete;
    DeviceField& operator=(const DeviceField&) = delete;
    double* data() { return p_; }
    const double* data() const { return p_; }
    long size() const { return n_; }

   private:
    long n_ = 0;
    double* p_ = nullptr;
};

// ---- kernels ----------------------------------------------------------------
// GPU vector backend for cg.hpp (see HostVec there) plus the stencil, the
// physical boundary fill and the halo pack / unpack used by CudaOperator.
// Holds the scratch buffers of the two-stage reductions.
class GpuVec {
   public:
    using FieldT = DeviceField;
    explicit GpuVec(const Grid& g);
    ~GpuVec();
    GpuVec(const GpuVec&) = delete;
    GpuVec& operator=(const GpuVec&) = delete;

    void fill_all(const Grid& g, double* x, double v);
    void sub(const Grid& g, double* z, const double* a, const double* b);
    void copy(const Grid& g, double* dst, const double* src);
    double dot(const Grid& g, const double* a, const double* b);
    void axpy(const Grid& g, double* y, double a, const double* x);
    void xpay(const Grid& g, double* y, double b, const double* x);
    double cg_update_xr(const Grid& g, double* x, double* r, const double* p, const double* q,
                        double a);
    void cg1_update(const Grid& g, double* x, double* r, double* p, double* s, const double* w,
                    double a, double b);

    // y = A x on box b; ndot as in kernels.hpp stencil() (sums are added to).
    void stencil(const Grid& g, const double* x, double* y, const Box& b, int ndot, double* sums);
    // apply_physical_bc() of kernels.hpp on the device.
    void apply_physical_bc(const Grid& g, double* x);
    // Copy the r layers of x starting at index `start` along dimension d
    // (tangential directions: local interior) to / from a contiguous buffer
    // of face_size(g, d) doubles. Does NOT synchronise (several faces are
    // packed back to back; call sync() before using the buffers).
    void pack(const Grid& g, const double* x, int d, int start, double* buf);
    void unpack(const Grid& g, double* x, int d, int start, const double* buf);

   private:
    double* partial_ = nullptr;   // per-block partial sums (2 per block)
    double* result_ = nullptr;    // final sums on the device
    double* host_result_ = nullptr;  // pinned copy of result_
    long max_blocks_ = 0;
    void reduce(long nblocks, int nsum, double* out);
};

// Number of doubles in one face message along dimension d.
long face_size(const Grid& g, int d);

}  // namespace gpu
}  // namespace poisson
