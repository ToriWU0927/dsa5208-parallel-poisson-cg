// grid.hpp — description of the (local) structured grid owned by one process.
//
// Global problem: -Laplace(u) = f on [0,1]^d with u = 0 on the boundary,
// discretised on ng[0] x ng[1] (x ng[2]) interior points, spacing h = 1/(ng+1).
//
// Each process owns a rectangular box of n[0] x n[1] x n[2] interior points
// (n[2] = 1 in 2D). Arrays are stored with a halo (ghost layer) of width
// hw[d] = r around the box, x index fastest:
//
//     idx(i,j,k) = (k+hw2)*sz + (j+hw1)*sy + (i+hw0),   -hw <= i < n+hw
//
// The serial program uses a single box covering the whole domain; the MPI
// program uses one box per rank. The same kernels operate on both.
#pragma once

#include <cstddef>

namespace poisson {

struct Box {  // half-open index ranges in local interior coordinates
    int i0, i1, j0, j1, k0, k1;
};

struct Grid {
    int dim = 3;       // 2 or 3
    int r = 1;         // stencil radius (1: 2nd order, 2: 4th order)
    int ng[3] = {1, 1, 1};   // global interior points per dimension
    int n[3] = {1, 1, 1};    // local interior points per dimension
    int off[3] = {0, 0, 0};  // global index of local point 0 (0-based)
    int hw[3] = {0, 0, 0};   // halo width per dimension
    int s[3] = {1, 1, 1};    // padded extents n + 2*hw
    long sy = 1, sz = 1;     // strides of j and k
    long ntot = 1;           // number of doubles in a padded array
    double h[3] = {1, 1, 1};       // grid spacing
    double inv_h2[3] = {0, 0, 0};  // 1/h^2 (0 for an unused dimension)
    bool phys_lo[3] = {true, true, true};  // face lies on the physical boundary
    bool phys_hi[3] = {true, true, true};

    long idx(int i, int j, int k) const {
        return (long)(k + hw[2]) * sz + (long)(j + hw[1]) * sy + (i + hw[0]);
    }
    Box interior() const { return Box{0, n[0], 0, n[1], 0, n[2]}; }
    long local_N() const { return (long)n[0] * n[1] * n[2]; }
    long global_N() const { return (long)ng[0] * ng[1] * ng[2]; }
};

// Grid for one box with global sizes ng, local sizes n and offset off.
// phys_lo/phys_hi say which faces touch the physical boundary.
Grid make_grid(int dim, int r, const int ng[3], const int n[3], const int off[3],
               const bool phys_lo[3], const bool phys_hi[3]);

// Grid covering the whole domain (serial program).
Grid make_serial_grid(int dim, int r, const int ng[3]);

// Number of nonzeros of the (implicit) global matrix A.
long long global_nnz(int dim, int r, const int ng[3]);

}  // namespace poisson
