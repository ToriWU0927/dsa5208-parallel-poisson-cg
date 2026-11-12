// gpu.cu — CUDA kernels and device memory management (see gpu.hpp).
//
// Thread mapping: one thread per grid point. Blocks of BX x BY = 128 x 2
// threads cover rows along x (the contiguous direction, so global memory
// accesses of a warp are coalesced); gridDim.z runs over the k planes.
// The stencil reads its neighbours directly from global memory; the reuse
// between neighbouring threads is served by the L1 / L2 caches.
#include "gpu.hpp"

#include <cuda_runtime.h>

#include <stdexcept>
#include <string>

namespace poisson {
namespace gpu {

namespace {

constexpr int BX = 128, BY = 2, BT = BX * BY;  // threads per block (box kernels)
constexpr int RT = 1024;                         // threads of the final reduction kernel

void check(cudaError_t e, const char* what) {
    if (e != cudaSuccess)
        throw std::runtime_error(std::string("CUDA error in ") + what + ": " + cudaGetErrorString(e));
}
#define CK(call) check((call), #call)
#define CK_LAUNCH(name) check(cudaGetLastError(), name)

// The part of Grid the kernels need (Grid itself has host-only member functions).
struct DG {
    int dim, r;
    int n0, n1, n2;
    int hw0, hw1, hw2;
    long sy, sz;
    double ax, ay, az;  // 1/h^2
    __device__ __forceinline__ long idx(int i, int j, int k) const {
        return (long)(k + hw2) * sz + (long)(j + hw1) * sy + (i + hw0);
    }
};

DG make_dg(const Grid& g) {
    return DG{g.dim, g.r, g.n[0], g.n[1], g.n[2], g.hw[0], g.hw[1], g.hw[2], g.sy, g.sz,
              g.inv_h2[0], g.inv_h2[1], g.inv_h2[2]};
}

struct Launch {
    dim3 grid, block;
    long nblocks;
};

Launch box_launch(const Box& b) {
    const int ni = b.i1 - b.i0, nj = b.j1 - b.j0, nk = b.k1 - b.k0;
    Launch l;
    l.block = dim3(BX, BY, 1);
    l.grid = dim3((ni + BX - 1) / BX, (nj + BY - 1) / BY, nk);
    l.nblocks = (long)l.grid.x * l.grid.y * l.grid.z;
    return l;
}

// Thread -> grid point of box b; `in` is false for the padding threads of the
// last block in a row (they must still take part in block reductions).
#define BOX_POINT(b)                                                   \
    const int i = (b).i0 + blockIdx.x * BX + threadIdx.x;             \
    const int j = (b).j0 + blockIdx.y * BY + threadIdx.y;             \
    const int k = (b).k0 + blockIdx.z;                                \
    const bool in = i < (b).i1 && j < (b).j1 && k < (b).k1;

__device__ __forceinline__ double warp_sum(double v) {
    for (int o = 16; o > 0; o >>= 1) v += __shfl_down_sync(0xffffffffu, v, o);
    return v;
}

// Sum s0, s1 over the BT threads of the block; thread 0 writes the block's
// partial sums to partial[2*block], partial[2*block+1]. Fixed order -> deterministic.
template <int NS>
__device__ __forceinline__ void block_sums(double s0, double s1, double* partial) {
    __shared__ double sh[2][BT / 32];
    const int t = threadIdx.y * BX + threadIdx.x, lane = t & 31, warp = t >> 5;
    s0 = warp_sum(s0);
    if (NS > 1) s1 = warp_sum(s1);
    if (lane == 0) { sh[0][warp] = s0; sh[1][warp] = s1; }
    __syncthreads();
    if (warp == 0) {
        s0 = lane < BT / 32 ? sh[0][lane] : 0.0;
        s1 = lane < BT / 32 ? sh[1][lane] : 0.0;
        s0 = warp_sum(s0);
        if (NS > 1) s1 = warp_sum(s1);
        if (lane == 0) {
            const long blk = ((long)blockIdx.z * gridDim.y + blockIdx.y) * gridDim.x + blockIdx.x;
            partial[2 * blk] = s0;
            partial[2 * blk + 1] = s1;
        }
    }
}

// Second stage: one block adds the nblocks partial sums (fixed order).
__global__ void k_reduce(const double* __restrict__ partial, long nblocks, double* out) {
    __shared__ double sh[2][RT];
    double s0 = 0.0, s1 = 0.0;
    for (long b = threadIdx.x; b < nblocks; b += RT) {
        s0 += partial[2 * b];
        s1 += partial[2 * b + 1];
    }
    sh[0][threadIdx.x] = s0;
    sh[1][threadIdx.x] = s1;
    __syncthreads();
    for (int w = RT / 2; w > 0; w >>= 1) {
        if ((int)threadIdx.x < w) {
            sh[0][threadIdx.x] += sh[0][threadIdx.x + w];
            sh[1][threadIdx.x] += sh[1][threadIdx.x + w];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) { out[0] = sh[0][0]; out[1] = sh[1][0]; }
}

// ---- stencil ------------------------------------------------------------------
template <int DIM, int R, int NDOT>
__global__ void k_stencil(DG g, const double* __restrict__ x, double* __restrict__ y, Box b,
                          double* partial) {
    BOX_POINT(b)
    constexpr double c0 = (R == 1) ? 2.0 : 30.0 / 12.0;  // as in kernels.cpp
    constexpr double c1 = (R == 1) ? -1.0 : -16.0 / 12.0;
    constexpr double c2 = (R == 1) ? 0.0 : 1.0 / 12.0;
    double s0 = 0.0, s1 = 0.0;
    if (in) {
        const double diag = c0 * (g.ax + g.ay + (DIM == 3 ? g.az : 0.0));
        const long m = g.idx(i, j, k), sy = g.sy, sz = g.sz;
        const double xc = x[m];
        double v = diag * xc + c1 * g.ax * (x[m - 1] + x[m + 1]) + c1 * g.ay * (x[m - sy] + x[m + sy]);
        if (DIM == 3) v += c1 * g.az * (x[m - sz] + x[m + sz]);
        if (R == 2) {
            v += c2 * g.ax * (x[m - 2] + x[m + 2]) + c2 * g.ay * (x[m - 2 * sy] + x[m + 2 * sy]);
            if (DIM == 3) v += c2 * g.az * (x[m - 2 * sz] + x[m + 2 * sz]);
        }
        y[m] = v;
        s0 = xc * v;
        s1 = xc * xc;
    }
    if (NDOT > 0) block_sums<NDOT>(s0, s1, partial);
}

template <int DIM, int R>
void stencil_launch(const Launch& l, const DG& g, const double* x, double* y, const Box& b,
                    int ndot, double* partial) {
    switch (ndot) {
        case 0: k_stencil<DIM, R, 0><<<l.grid, l.block>>>(g, x, y, b, partial); break;
        case 1: k_stencil<DIM, R, 1><<<l.grid, l.block>>>(g, x, y, b, partial); break;
        default: k_stencil<DIM, R, 2><<<l.grid, l.block>>>(g, x, y, b, partial); break;
    }
}

// ---- vector operations (local interior) ------------------------------------------
__global__ void k_fill(double* x, long n, double v) {
    const long m = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (m < n) x[m] = v;
}

__global__ void k_sub(DG g, Box b, double* __restrict__ z, const double* __restrict__ a,
                      const double* __restrict__ c) {
    BOX_POINT(b)
    if (in) { const long m = g.idx(i, j, k); z[m] = a[m] - c[m]; }
}
__global__ void k_copy(DG g, Box b, double* __restrict__ d, const double* __restrict__ s) {
    BOX_POINT(b)
    if (in) { const long m = g.idx(i, j, k); d[m] = s[m]; }
}
__global__ void k_axpy(DG g, Box b, double* __restrict__ y, double a, const double* __restrict__ x) {
    BOX_POINT(b)
    if (in) { const long m = g.idx(i, j, k); y[m] += a * x[m]; }
}
__global__ void k_xpay(DG g, Box b, double* __restrict__ y, double c, const double* __restrict__ x) {
    BOX_POINT(b)
    if (in) { const long m = g.idx(i, j, k); y[m] = x[m] + c * y[m]; }
}
__global__ void k_dot(DG g, Box b, const double* __restrict__ a, const double* __restrict__ c,
                      double* partial) {
    BOX_POINT(b)
    double s = 0.0;
    if (in) { const long m = g.idx(i, j, k); s = a[m] * c[m]; }
    block_sums<1>(s, 0.0, partial);
}
__global__ void k_cg_update_xr(DG g, Box b, double* __restrict__ x, double* __restrict__ r,
                               const double* __restrict__ p, const double* __restrict__ q,
                               double a, double* partial) {
    BOX_POINT(b)
    double s = 0.0;
    if (in) {
        const long m = g.idx(i, j, k);
        x[m] += a * p[m];
        const double rv = r[m] - a * q[m];
        r[m] = rv;
        s = rv * rv;
    }
    block_sums<1>(s, 0.0, partial);
}
__global__ void k_cg1_update(DG g, Box b, double* __restrict__ x, double* __restrict__ r,
                             double* __restrict__ p, double* __restrict__ s,
                             const double* __restrict__ w, double a, double c) {
    BOX_POINT(b)
    if (in) {
        const long m = g.idx(i, j, k);
        const double pv = r[m] + c * p[m];
        const double sv = w[m] + c * s[m];
        p[m] = pv;
        s[m] = sv;
        x[m] += a * pv;
        r[m] -= a * sv;
    }
}

// ---- boundary fill and halo faces ---------------------------------------------------
// Point (layer l, tangential a, b) of a face normal to d -> local coordinates.
__device__ __forceinline__ void face_point(int d, int l, int a, int bb, int c[3]) {
    const int t1 = (d == 0) ? 1 : 0, t2 = (d == 2) ? 1 : 2;
    c[t1] = a;
    c[t2] = bb;
    c[d] = l;
}

// One physical face: ghost layer 1 -> 0, ghost layer 2 (r = 2) -> -(first interior point).
__global__ void k_bc(DG g, int d, int ghost1, int ghost2, int inner, int nt1, int nt2, double* x) {
    const long t = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= (long)nt1 * nt2) return;
    const int a = (int)(t % nt1), bb = (int)(t / nt1);
    int c[3];
    face_point(d, ghost1, a, bb, c);
    x[g.idx(c[0], c[1], c[2])] = 0.0;
    if (g.r == 2) {
        face_point(d, inner, a, bb, c);
        const double v = x[g.idx(c[0], c[1], c[2])];
        face_point(d, ghost2, a, bb, c);
        x[g.idx(c[0], c[1], c[2])] = -v;
    }
}

// buf[(l * nt2 + b) * nt1 + a] <-> x(face point (start + l, a, b))
template <bool PACK>
__global__ void k_face(DG g, int d, int start, int nt1, int nt2, double* x, double* buf) {
    const long t = (long)blockIdx.x * blockDim.x + threadIdx.x;
    const long per_layer = (long)nt1 * nt2;
    if (t >= per_layer * g.r) return;
    const int l = (int)(t / per_layer);
    const long rest = t - l * per_layer;
    const int a = (int)(rest % nt1), bb = (int)(rest / nt1);
    int c[3];
    face_point(d, start + l, a, bb, c);
    const long m = g.idx(c[0], c[1], c[2]);
    if (PACK) buf[t] = x[m];
    else x[m] = buf[t];
}

void tangential(const Grid& g, int d, int& nt1, int& nt2) {
    const int t1 = (d == 0) ? 1 : 0, t2 = (d == 2) ? 1 : 2;
    nt1 = g.n[t1];
    nt2 = g.n[t2];
}

inline unsigned blocks_for(long n, int bs) { return (unsigned)((n + bs - 1) / bs); }

}  // namespace

// ---- public API -----------------------------------------------------------------------
int select_device(int local_rank) {
    int n = 0;
    CK(cudaGetDeviceCount(&n));
    if (n == 0) throw std::runtime_error("no CUDA device visible");
    CK(cudaSetDevice(local_rank % n));
    CK(cudaFree(nullptr));  // create the context now (not inside a timed region)
    return n;
}

const char* device_name() {
    static cudaDeviceProp prop;
    int dev = 0;
    CK(cudaGetDevice(&dev));
    CK(cudaGetDeviceProperties(&prop, dev));
    return prop.name;
}

double* alloc_device(long n) {
    double* p = nullptr;
    if (n > 0) CK(cudaMalloc(&p, sizeof(double) * n));
    return p;
}
void free_device(double* p) {
    if (p) cudaFree(p);
}
double* alloc_pinned(long n) {
    double* p = nullptr;
    if (n > 0) CK(cudaMallocHost(&p, sizeof(double) * n));
    return p;
}
void free_pinned(double* p) {
    if (p) cudaFreeHost(p);
}
void to_device(double* dst, const double* src, long n) {
    CK(cudaMemcpy(dst, src, sizeof(double) * n, cudaMemcpyHostToDevice));
}
void to_host(double* dst, const double* src, long n) {
    CK(cudaMemcpy(dst, src, sizeof(double) * n, cudaMemcpyDeviceToHost));
}
void sync() { CK(cudaDeviceSynchronize()); }

long face_size(const Grid& g, int d) {
    int nt1, nt2;
    tangential(g, d, nt1, nt2);
    return (long)g.r * nt1 * nt2;
}

GpuVec::GpuVec(const Grid& g) {
    max_blocks_ = box_launch(g.interior()).nblocks;
    partial_ = alloc_device(2 * max_blocks_);
    result_ = alloc_device(2);
    host_result_ = alloc_pinned(2);
}

GpuVec::~GpuVec() {
    free_device(partial_);
    free_device(result_);
    free_pinned(host_result_);
}

void GpuVec::reduce(long nblocks, int nsum, double* out) {
    k_reduce<<<1, RT>>>(partial_, nblocks, result_);
    CK_LAUNCH("k_reduce");
    CK(cudaMemcpy(host_result_, result_, 2 * sizeof(double), cudaMemcpyDeviceToHost));
    for (int c = 0; c < nsum; ++c) out[c] = host_result_[c];
}

void GpuVec::fill_all(const Grid& g, double* x, double v) {
    k_fill<<<blocks_for(g.ntot, 256), 256>>>(x, g.ntot, v);
    CK_LAUNCH("k_fill");
    sync();
}

#define INTERIOR_LAUNCH                    \
    const DG d = make_dg(g);               \
    const Box b = g.interior();            \
    const Launch l = box_launch(b);

void GpuVec::sub(const Grid& g, double* z, const double* a, const double* c) {
    INTERIOR_LAUNCH
    k_sub<<<l.grid, l.block>>>(d, b, z, a, c);
    CK_LAUNCH("k_sub");
    sync();
}
void GpuVec::copy(const Grid& g, double* dst, const double* src) {
    INTERIOR_LAUNCH
    k_copy<<<l.grid, l.block>>>(d, b, dst, src);
    CK_LAUNCH("k_copy");
    sync();
}
double GpuVec::dot(const Grid& g, const double* a, const double* c) {
    INTERIOR_LAUNCH
    k_dot<<<l.grid, l.block>>>(d, b, a, c, partial_);
    CK_LAUNCH("k_dot");
    double s = 0.0;
    reduce(l.nblocks, 1, &s);
    return s;
}
void GpuVec::axpy(const Grid& g, double* y, double a, const double* x) {
    INTERIOR_LAUNCH
    k_axpy<<<l.grid, l.block>>>(d, b, y, a, x);
    CK_LAUNCH("k_axpy");
    sync();
}
void GpuVec::xpay(const Grid& g, double* y, double c, const double* x) {
    INTERIOR_LAUNCH
    k_xpay<<<l.grid, l.block>>>(d, b, y, c, x);
    CK_LAUNCH("k_xpay");
    sync();
}
double GpuVec::cg_update_xr(const Grid& g, double* x, double* r, const double* p, const double* q,
                            double a) {
    INTERIOR_LAUNCH
    k_cg_update_xr<<<l.grid, l.block>>>(d, b, x, r, p, q, a, partial_);
    CK_LAUNCH("k_cg_update_xr");
    double s = 0.0;
    reduce(l.nblocks, 1, &s);
    return s;
}
void GpuVec::cg1_update(const Grid& g, double* x, double* r, double* p, double* s,
                        const double* w, double a, double c) {
    INTERIOR_LAUNCH
    k_cg1_update<<<l.grid, l.block>>>(d, b, x, r, p, s, w, a, c);
    CK_LAUNCH("k_cg1_update");
    sync();
}

void GpuVec::stencil(const Grid& g, const double* x, double* y, const Box& b, int ndot,
                     double* sums) {
    if (b.i0 >= b.i1 || b.j0 >= b.j1 || b.k0 >= b.k1) return;
    const DG d = make_dg(g);
    const Launch l = box_launch(b);
    if (g.dim == 2) {
        if (g.r == 1) stencil_launch<2, 1>(l, d, x, y, b, ndot, partial_);
        else stencil_launch<2, 2>(l, d, x, y, b, ndot, partial_);
    } else {
        if (g.r == 1) stencil_launch<3, 1>(l, d, x, y, b, ndot, partial_);
        else stencil_launch<3, 2>(l, d, x, y, b, ndot, partial_);
    }
    CK_LAUNCH("k_stencil");
    if (ndot > 0) {
        double s[2] = {0.0, 0.0};
        reduce(l.nblocks, ndot, s);
        for (int c = 0; c < ndot; ++c) sums[c] += s[c];
    } else {
        sync();
    }
}

void GpuVec::apply_physical_bc(const Grid& g, double* x) {
    const DG dg = make_dg(g);
    for (int d = 0; d < g.dim; ++d) {
        int nt1, nt2;
        tangential(g, d, nt1, nt2);
        const long nf = (long)nt1 * nt2;
        for (int side = 0; side < 2; ++side) {
            if (side == 0 ? !g.phys_lo[d] : !g.phys_hi[d]) continue;
            const int ghost1 = side == 0 ? -1 : g.n[d];
            const int ghost2 = side == 0 ? -2 : g.n[d] + 1;
            const int inner = side == 0 ? 0 : g.n[d] - 1;
            k_bc<<<blocks_for(nf, 256), 256>>>(dg, d, ghost1, ghost2, inner, nt1, nt2, x);
            CK_LAUNCH("k_bc");
        }
    }
    sync();
}

void GpuVec::pack(const Grid& g, const double* x, int d, int start, double* buf) {
    int nt1, nt2;
    tangential(g, d, nt1, nt2);
    const long n = face_size(g, d);
    k_face<true><<<blocks_for(n, 256), 256>>>(make_dg(g), d, start, nt1, nt2,
                                              const_cast<double*>(x), buf);
    CK_LAUNCH("k_face<pack>");
}

void GpuVec::unpack(const Grid& g, double* x, int d, int start, const double* buf) {
    int nt1, nt2;
    tangential(g, d, nt1, nt2);
    const long n = face_size(g, d);
    k_face<false><<<blocks_for(n, 256), 256>>>(make_dg(g), d, start, nt1, nt2, x,
                                               const_cast<double*>(buf));
    CK_LAUNCH("k_face<unpack>");
}

}  // namespace gpu
}  // namespace poisson
