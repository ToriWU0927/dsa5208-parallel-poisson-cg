// halo.hpp — halo (ghost layer) exchange between neighbouring MPI processes.
//
// The stencil is star-shaped (it only reaches along the coordinate axes), so
// only the 2*dim FACE neighbours are needed — no edge or corner exchange.
// Each face message is the layer of width r next to the face, restricted to
// the local interior in the tangential directions. Messages are described by
// MPI derived datatypes (MPI_Type_create_subarray) directly on the padded
// array, so MPI does the packing and no extra buffers are needed.
//
//   exchange(x)          blocking: one MPI_Sendrecv per direction and side
//   begin(x) ... end()   non-blocking: MPI_Irecv/MPI_Isend for all faces in
//                        begin(), MPI_Waitall in end(); used to overlap the
//                        exchange with the stencil on the inner points
#pragma once

#include <mpi.h>

#include "decomp.hpp"
#include "grid.hpp"

namespace poisson {

class HaloExchanger {
   public:
    HaloExchanger(const Grid& g, const Decomposition& dc);
    ~HaloExchanger();
    HaloExchanger(const HaloExchanger&) = delete;
    HaloExchanger& operator=(const HaloExchanger&) = delete;

    void exchange(double* x);
    void begin(double* x);
    void end();

   private:
    const Grid& g_;
    const Decomposition& dc_;
    // [d][side]: side 0 = low face, 1 = high face
    MPI_Datatype send_[3][2];
    MPI_Datatype recv_[3][2];
    MPI_Request req_[12];
    int nreq_ = 0;
};

}  // namespace poisson
