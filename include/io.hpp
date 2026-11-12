// io.hpp — write the solution on the GLOBAL interior grid to a raw binary
// file (float64, native byte order, x fastest: index = (k*ny + j)*nx + i).
// Used only by validation V2/V4 (--save) to compare solutions element-wise
// between the serial and the parallel programs; not used in timed runs.
#pragma once

#include <cstdio>
#include <stdexcept>
#include <string>

#ifdef USE_MPI
#include <mpi.h>
#endif

#include "grid.hpp"

namespace poisson {

// Serial program: the local box is the whole domain.
inline void write_solution(const Grid& g, const double* x, const std::string& path) {
    std::FILE* f = std::fopen(path.c_str(), "wb");
    if (!f) throw std::runtime_error("cannot open " + path);
    for (int k = 0; k < g.n[2]; ++k)
        for (int j = 0; j < g.n[1]; ++j)
            std::fwrite(x + g.idx(0, j, k), sizeof(double), g.n[0], f);
    std::fclose(f);
}

#ifdef USE_MPI
// MPI program: collective MPI-IO write; every rank writes its own box into
// the global array (file view = global subarray, memory = local interior).
inline void write_solution_mpi(const Grid& g, const double* x, const std::string& path,
                               MPI_Comm comm) {
    int gsz[3], lsz[3], goff[3], psz[3], pst[3];
    for (int e = 0; e < 3; ++e) {  // physical dimension e -> C-order index 2-e
        gsz[2 - e] = g.ng[e];
        lsz[2 - e] = g.n[e];
        goff[2 - e] = g.off[e];
        psz[2 - e] = g.s[e];
        pst[2 - e] = g.hw[e];
    }
    MPI_Datatype ftype, mtype;
    MPI_Type_create_subarray(3, gsz, lsz, goff, MPI_ORDER_C, MPI_DOUBLE, &ftype);
    MPI_Type_create_subarray(3, psz, lsz, pst, MPI_ORDER_C, MPI_DOUBLE, &mtype);
    MPI_Type_commit(&ftype);
    MPI_Type_commit(&mtype);
    MPI_File fh;
    if (MPI_File_open(comm, path.c_str(), MPI_MODE_CREATE | MPI_MODE_WRONLY, MPI_INFO_NULL,
                      &fh) != MPI_SUCCESS)
        throw std::runtime_error("cannot open " + path);
    MPI_File_set_size(fh, 0);
    MPI_File_set_view(fh, 0, MPI_DOUBLE, ftype, "native", MPI_INFO_NULL);
    MPI_File_write_all(fh, x, 1, mtype, MPI_STATUS_IGNORE);
    MPI_File_close(&fh);
    MPI_Type_free(&ftype);
    MPI_Type_free(&mtype);
}
#endif

}  // namespace poisson
