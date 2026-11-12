#include "halo.hpp"

namespace poisson {

namespace {
// Subarray of the padded local array. Physical dimension d maps to C-order
// array dimension 2-d (z slowest, x fastest).
MPI_Datatype face_type(const Grid& g, int d, int start_d) {
    int sizes[3], subs[3], starts[3];
    for (int e = 0; e < 3; ++e) {
        sizes[2 - e] = g.s[e];
        subs[2 - e] = (e == d) ? g.r : g.n[e];
        starts[2 - e] = (e == d) ? start_d : g.hw[e];
    }
    MPI_Datatype t;
    MPI_Type_create_subarray(3, sizes, subs, starts, MPI_ORDER_C, MPI_DOUBLE, &t);
    MPI_Type_commit(&t);
    return t;
}
// Tags identify the direction of travel so that the two messages exchanged
// between the same pair of processes cannot be confused.
inline int tag(int d, int towards_hi) { return 2 * d + towards_hi; }
}  // namespace

HaloExchanger::HaloExchanger(const Grid& g, const Decomposition& dc) : g_(g), dc_(dc) {
    for (int d = 0; d < 3; ++d)
        for (int s = 0; s < 2; ++s) send_[d][s] = recv_[d][s] = MPI_DATATYPE_NULL;
    for (int d = 0; d < g.dim; ++d) {
        const int r = g.r, hw = g.hw[d], n = g.n[d];
        send_[d][0] = face_type(g, d, hw);           // first r interior layers
        send_[d][1] = face_type(g, d, hw + n - r);   // last r interior layers
        recv_[d][0] = face_type(g, d, hw - r);       // low halo
        recv_[d][1] = face_type(g, d, hw + n);       // high halo
    }
}

HaloExchanger::~HaloExchanger() {
    for (int d = 0; d < 3; ++d)
        for (int s = 0; s < 2; ++s) {
            if (send_[d][s] != MPI_DATATYPE_NULL) MPI_Type_free(&send_[d][s]);
            if (recv_[d][s] != MPI_DATATYPE_NULL) MPI_Type_free(&recv_[d][s]);
        }
}

void HaloExchanger::exchange(double* x) {
    for (int d = 0; d < g_.dim; ++d) {
        // send high face up, receive low halo from below
        MPI_Sendrecv(x, 1, send_[d][1], dc_.nbr_hi[d], tag(d, 1),
                     x, 1, recv_[d][0], dc_.nbr_lo[d], tag(d, 1), dc_.comm, MPI_STATUS_IGNORE);
        // send low face down, receive high halo from above
        MPI_Sendrecv(x, 1, send_[d][0], dc_.nbr_lo[d], tag(d, 0),
                     x, 1, recv_[d][1], dc_.nbr_hi[d], tag(d, 0), dc_.comm, MPI_STATUS_IGNORE);
    }
}

void HaloExchanger::begin(double* x) {
    nreq_ = 0;
    for (int d = 0; d < g_.dim; ++d) {
        MPI_Irecv(x, 1, recv_[d][0], dc_.nbr_lo[d], tag(d, 1), dc_.comm, &req_[nreq_++]);
        MPI_Irecv(x, 1, recv_[d][1], dc_.nbr_hi[d], tag(d, 0), dc_.comm, &req_[nreq_++]);
    }
    for (int d = 0; d < g_.dim; ++d) {
        MPI_Isend(x, 1, send_[d][1], dc_.nbr_hi[d], tag(d, 1), dc_.comm, &req_[nreq_++]);
        MPI_Isend(x, 1, send_[d][0], dc_.nbr_lo[d], tag(d, 0), dc_.comm, &req_[nreq_++]);
    }
}

void HaloExchanger::end() {
    MPI_Waitall(nreq_, req_, MPI_STATUSES_IGNORE);
    nreq_ = 0;
}

}  // namespace poisson
