#include "grid.hpp"

#include <stdexcept>
#include <string>

namespace poisson {

Grid make_grid(int dim, int r, const int ng[3], const int n[3], const int off[3],
               const bool phys_lo[3], const bool phys_hi[3]) {
    if (dim != 2 && dim != 3) throw std::invalid_argument("dim must be 2 or 3");
    if (r != 1 && r != 2) throw std::invalid_argument("radius must be 1 or 2");
    Grid g;
    g.dim = dim;
    g.r = r;
    for (int d = 0; d < 3; ++d) {
        const bool active = d < dim;
        g.ng[d] = active ? ng[d] : 1;
        g.n[d] = active ? n[d] : 1;
        g.off[d] = active ? off[d] : 0;
        g.hw[d] = active ? r : 0;
        g.s[d] = g.n[d] + 2 * g.hw[d];
        g.h[d] = active ? 1.0 / (g.ng[d] + 1) : 1.0;
        g.inv_h2[d] = active ? 1.0 / (g.h[d] * g.h[d]) : 0.0;
        g.phys_lo[d] = active ? phys_lo[d] : true;
        g.phys_hi[d] = active ? phys_hi[d] : true;
        if (active && g.n[d] < r)
            throw std::invalid_argument("local box has fewer than r points in dimension " +
                                        std::to_string(d));
    }
    g.sy = g.s[0];
    g.sz = (long)g.s[0] * g.s[1];
    g.ntot = g.sz * g.s[2];
    return g;
}

Grid make_serial_grid(int dim, int r, const int ng[3]) {
    const int off[3] = {0, 0, 0};
    const bool t[3] = {true, true, true};
    return make_grid(dim, r, ng, ng, off, t, t);
}

long long global_nnz(int dim, int r, const int ng[3]) {
    long long N = 1;
    for (int d = 0; d < dim; ++d) N *= ng[d];
    long long nnz = N;  // diagonal
    for (int d = 0; d < dim; ++d) {
        const long long lines = N / ng[d];  // grid lines along dimension d
        for (int k = 1; k <= r; ++k) {
            const long long pairs = ng[d] > k ? ng[d] - k : 0;  // couplings at distance k
            nnz += 2 * lines * pairs;
        }
    }
    return nnz;
}

}  // namespace poisson
