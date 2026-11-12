// decomp.hpp — Cartesian domain decomposition over MPI processes.
//
// The global grid ng[0] x ng[1] (x ng[2]) is split into a process grid
// pd[0] x pd[1] (x pd[2]). Along each dimension the points are distributed in
// contiguous blocks; when ng is not divisible by pd the first (ng % pd)
// processes get one extra point, so local sizes differ by at most 1.
//
//   auto: pd from MPI_Dims_create (as close to a cube as possible); the largest
//         factor goes to the slowest-varying dimension (z in 3D, y in 2D)
//   slab: all processes along the slowest dimension (1D decomposition)
//
// The MPI Cartesian communicator is created with dimension order
// (slowest, ..., x), so consecutive ranks are neighbours along x.
#pragma once

#include <mpi.h>

#include <string>

#include "grid.hpp"

namespace poisson {

struct Decomposition {
    MPI_Comm comm = MPI_COMM_NULL;  // Cartesian communicator
    int rank = 0, size = 1;
    int pd[3] = {1, 1, 1};          // processes per physical dimension (x, y, z)
    int coord[3] = {0, 0, 0};       // this process's coordinate per physical dimension
    int nbr_lo[3] = {MPI_PROC_NULL, MPI_PROC_NULL, MPI_PROC_NULL};  // neighbour at -d
    int nbr_hi[3] = {MPI_PROC_NULL, MPI_PROC_NULL, MPI_PROC_NULL};  // neighbour at +d
    std::string name;               // e.g. "auto:4x4x2" (x by y by z)
};

// Processes per physical dimension (pd[0] = x) for p processes and the given
// mode ("auto" or "slab"); unused dimensions get 1. Same rule as
// make_decomposition, usable before the global grid size is known (weak scaling).
void process_grid(int p, int dim, const std::string& mode, int pd[3]);

// Builds the process grid and the local Grid of this rank. Collective over
// world. Throws std::runtime_error (on every rank) if a local box would have
// fewer than r points in some dimension.
Decomposition make_decomposition(MPI_Comm world, int dim, const int ng[3], int r,
                                 const std::string& mode, Grid& g);

// Number of distinct shared-memory nodes in comm.
int count_nodes(MPI_Comm comm);

}  // namespace poisson
