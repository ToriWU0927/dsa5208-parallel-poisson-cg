#include "decomp.hpp"

#include <algorithm>
#include <stdexcept>
#include <string>

namespace poisson {

// cdims[c]: processes along Cartesian dimension c; c = 0 is the slowest
// physical dimension (z in 3D, y in 2D), c = dim-1 is x.
static void cart_dims(int p, int dim, const std::string& mode, int cdims[3]) {
    cdims[0] = cdims[1] = cdims[2] = 0;
    if (mode == "slab") {
        for (int c = 0; c < dim; ++c) cdims[c] = 1;
        cdims[0] = p;
    } else {
        MPI_Dims_create(p, dim, cdims);  // non-increasing order
    }
}

void process_grid(int p, int dim, const std::string& mode, int pd[3]) {
    int cdims[3];
    cart_dims(p, dim, mode, cdims);
    pd[0] = pd[1] = pd[2] = 1;
    for (int d = 0; d < dim; ++d) pd[d] = cdims[dim - 1 - d];
}

Decomposition make_decomposition(MPI_Comm world, int dim, const int ng[3], int r,
                                 const std::string& mode, Grid& g) {
    Decomposition dc;
    int p = 1;
    MPI_Comm_size(world, &p);

    int cdims[3];
    cart_dims(p, dim, mode, cdims);
    for (int d = 0; d < dim; ++d) dc.pd[d] = cdims[dim - 1 - d];

    // A process grid finer than the global grid along some dimension cannot
    // work; checked before creating the communicator so every rank agrees.
    for (int d = 0; d < dim; ++d)
        if (ng[d] / dc.pd[d] < r)
            throw std::runtime_error("decomposition " + std::to_string(dc.pd[d]) +
                                     " processes along dimension " + std::to_string(d) +
                                     " leaves fewer than r = " + std::to_string(r) +
                                     " points per process");

    int periods[3] = {0, 0, 0};
    MPI_Cart_create(world, dim, cdims, periods, /*reorder=*/1, &dc.comm);
    MPI_Comm_rank(dc.comm, &dc.rank);
    MPI_Comm_size(dc.comm, &dc.size);

    int ccoord[3] = {0, 0, 0};
    MPI_Cart_coords(dc.comm, dc.rank, dim, ccoord);
    for (int d = 0; d < dim; ++d) {
        dc.coord[d] = ccoord[dim - 1 - d];
        MPI_Cart_shift(dc.comm, dim - 1 - d, 1, &dc.nbr_lo[d], &dc.nbr_hi[d]);
    }

    int n[3] = {1, 1, 1}, off[3] = {0, 0, 0};
    bool lo[3] = {true, true, true}, hi[3] = {true, true, true};
    for (int d = 0; d < dim; ++d) {
        const int q = ng[d] / dc.pd[d], rem = ng[d] % dc.pd[d], c = dc.coord[d];
        n[d] = q + (c < rem ? 1 : 0);
        off[d] = c * q + std::min(c, rem);
        lo[d] = (c == 0);
        hi[d] = (c == dc.pd[d] - 1);
    }
    g = make_grid(dim, r, ng, n, off, lo, hi);

    dc.name = mode + ":" + std::to_string(dc.pd[0]) + "x" + std::to_string(dc.pd[1]);
    if (dim == 3) dc.name += "x" + std::to_string(dc.pd[2]);
    return dc;
}

int count_nodes(MPI_Comm comm) {
    MPI_Comm node;
    MPI_Comm_split_type(comm, MPI_COMM_TYPE_SHARED, 0, MPI_INFO_NULL, &node);
    int node_rank = 0;
    MPI_Comm_rank(node, &node_rank);
    int leader = node_rank == 0 ? 1 : 0, nodes = 0;
    MPI_Allreduce(&leader, &nodes, 1, MPI_INT, MPI_SUM, comm);
    MPI_Comm_free(&node);
    return nodes;
}

}  // namespace poisson
