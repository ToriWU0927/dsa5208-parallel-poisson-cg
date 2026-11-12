// pingpong — measures the network parameters of the communication model
// (E4):  T(m) ≈ alpha + m / beta  for one point-to-point message of m
// bytes, and the latency of a small MPI_Allreduce over all ranks (what CG does
// twice per iteration).
//
//   mpirun -np P ./build/pingpong [max_bytes]      (P >= 2)
//
// Ping-pong runs between rank 0 and rank P-1 (place them on different nodes
// to measure the network, e.g. --map-by node). All ranks join the allreduce.
// Output (rank 0, CSV): test,bytes,ranks,time_s,bandwidth_MBps,host0,host1

#include <mpi.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank = 0, size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    if (size < 2) {
        if (rank == 0) std::fprintf(stderr, "pingpong needs at least 2 ranks\n");
        MPI_Finalize();
        return 1;
    }
    const long max_bytes = argc > 1 ? std::atol(argv[1]) : (8L << 20);
    const int peer = size - 1;

    char host[MPI_MAX_PROCESSOR_NAME] = {0}, host_peer[MPI_MAX_PROCESSOR_NAME] = {0};
    int len = 0;
    MPI_Get_processor_name(host, &len);
    if (rank == peer) MPI_Send(host, MPI_MAX_PROCESSOR_NAME, MPI_CHAR, 0, 9, MPI_COMM_WORLD);
    if (rank == 0) MPI_Recv(host_peer, MPI_MAX_PROCESSOR_NAME, MPI_CHAR, peer, 9, MPI_COMM_WORLD,
                            MPI_STATUS_IGNORE);
    if (rank == 0) std::printf("test,bytes,ranks,time_s,bandwidth_MBps,host0,host1\n");

    std::vector<char> buf(std::max(max_bytes, 1L));
    for (long m = 8; m <= max_bytes; m *= 4) {
        const int reps = m <= 65536 ? 1000 : (m <= (1L << 20) ? 100 : 20);
        for (int warm = 0; warm < 2; ++warm) {  // first pass is warm-up
            MPI_Barrier(MPI_COMM_WORLD);
            const double t0 = MPI_Wtime();
            for (int i = 0; i < reps; ++i) {
                if (rank == 0) {
                    MPI_Send(buf.data(), (int)m, MPI_CHAR, peer, 0, MPI_COMM_WORLD);
                    MPI_Recv(buf.data(), (int)m, MPI_CHAR, peer, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                } else if (rank == peer) {
                    MPI_Recv(buf.data(), (int)m, MPI_CHAR, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                    MPI_Send(buf.data(), (int)m, MPI_CHAR, 0, 0, MPI_COMM_WORLD);
                }
            }
            const double t = (MPI_Wtime() - t0) / (2.0 * reps);  // one-way time
            if (warm == 1 && rank == 0)
                std::printf("pingpong,%ld,2,%.6e,%.2f,%s,%s\n", m, t, m / t / 1e6, host, host_peer);
        }
    }

    // small allreduce, as in CG (1 or 2 doubles per call)
    for (int nd : {1, 2}) {
        double v[2] = {1.0, 1.0};
        const int reps = 2000;
        for (int warm = 0; warm < 2; ++warm) {
            MPI_Barrier(MPI_COMM_WORLD);
            const double t0 = MPI_Wtime();
            for (int i = 0; i < reps; ++i)
                MPI_Allreduce(MPI_IN_PLACE, v, nd, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
            double t = (MPI_Wtime() - t0) / reps, tmax = 0.0;
            MPI_Reduce(&t, &tmax, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
            if (warm == 1 && rank == 0)
                std::printf("allreduce,%d,%d,%.6e,,%s,\n", 8 * nd, size, tmax, host);
        }
    }
    std::fflush(stdout);
    MPI_Finalize();
    return 0;
}
