#include "options.hpp"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>

namespace poisson {

CGParams Options::cg_params() const {
    CGParams p;
    p.variant = cg;
    p.tol = tol;
    p.fixed_iters = fixed_iters;
    p.fuse = fuse;
    p.maxit = maxit > 0 ? maxit : 10 * std::max({ng[0], ng[1], ng[2]});
    return p;
}

static void usage(const char* program) {
    std::fprintf(stderr,
                 "Usage: %s [options]\n"
                 "  --dim 2|3            problem dimension (default 3)\n"
                 "  --problem mms|ones   manufactured solution (validation) or f = 1 (default mms)\n"
                 "  --n N                interior points per dimension\n"
                 "  --nx/--ny/--nz N     interior points in one dimension\n"
                 "  --local-n L          MPI only, weak scaling: L points per process and dimension\n"
                 "                       (global n = L * processes along that dimension)\n"
                 "  --radius 1|2         stencil radius: 1 = 2nd order, 2 = 4th order (default 1)\n"
                 "  --tol T              relative residual tolerance (default 1e-8)\n"
                 "  --maxit K            maximum iterations (default 10*max(n))\n"
                 "  --fixed-iters K      run exactly K iterations (scaling runs)\n"
                 "  --cg std|single      standard CG or single-reduction CG (default std)\n"
                 "  --fuse 0|1           fused vector kernels (default 1)\n"
                 "  --repeat R           repetitions of the solve (default 3)\n"
                 "  --verify             compute max error against the exact solution\n"
                 "  --header             print the CSV header line first\n"
                 "  --verbose            human-readable summary on stderr\n"
                 "  --save FILE          write the final solution (raw float64, x fastest) for validation\n"
                 "  --halo block|overlap halo exchange (MPI only, default block)\n"
                 "  --decomp auto|slab   process grid (MPI only, default auto)\n"
                 "  --gpu-aware 0|1      CUDA-aware MPI (CUDA only, default 1)\n"
                 "  --sync-timers 0|1    timed MPI_Barrier before each Allreduce: separates waiting\n"
                 "                       for the slowest rank (t_wait) from the reduction (diagnostics,\n"
                 "                       adds one barrier per reduction; default 0)\n",
                 program);
}

Options parse_options(int argc, char** argv, const char* program) {
    Options o;
    auto need = [&](int& i) -> const char* {
        if (i + 1 >= argc) {
            std::fprintf(stderr, "missing value for %s\n", argv[i]);
            usage(program);
            std::exit(2);
        }
        return argv[++i];
    };
    try {
        for (int i = 1; i < argc; ++i) {
            const char* a = argv[i];
            if (!std::strcmp(a, "--help") || !std::strcmp(a, "-h")) {
                usage(program);
                std::exit(0);
            } else if (!std::strcmp(a, "--dim")) o.dim = std::stoi(need(i));
            else if (!std::strcmp(a, "--problem")) {
                const std::string v = need(i);
                if (v == "mms") o.problem = Problem::MMS;
                else if (v == "ones") o.problem = Problem::Ones;
                else throw std::invalid_argument("--problem must be mms or ones");
            }
            else if (!std::strcmp(a, "--n")) o.ng[0] = o.ng[1] = o.ng[2] = std::stoi(need(i));
            else if (!std::strcmp(a, "--nx")) o.ng[0] = std::stoi(need(i));
            else if (!std::strcmp(a, "--ny")) o.ng[1] = std::stoi(need(i));
            else if (!std::strcmp(a, "--nz")) o.ng[2] = std::stoi(need(i));
            else if (!std::strcmp(a, "--local-n")) o.local_n = std::stoi(need(i));
            else if (!std::strcmp(a, "--radius")) o.radius = std::stoi(need(i));
            else if (!std::strcmp(a, "--tol")) o.tol = std::stod(need(i));
            else if (!std::strcmp(a, "--maxit")) o.maxit = std::stoi(need(i));
            else if (!std::strcmp(a, "--fixed-iters")) o.fixed_iters = std::stoi(need(i));
            else if (!std::strcmp(a, "--cg")) {
                const std::string v = need(i);
                if (v == "std") o.cg = CGVariant::Standard;
                else if (v == "single") o.cg = CGVariant::SingleReduction;
                else throw std::invalid_argument("--cg must be std or single");
            } else if (!std::strcmp(a, "--fuse")) o.fuse = std::stoi(need(i)) != 0;
            else if (!std::strcmp(a, "--repeat")) o.repeat = std::stoi(need(i));
            else if (!std::strcmp(a, "--verify")) o.verify = true;
            else if (!std::strcmp(a, "--header")) o.header = true;
            else if (!std::strcmp(a, "--verbose")) o.verbose = true;
            else if (!std::strcmp(a, "--save")) o.save = need(i);
            else if (!std::strcmp(a, "--halo")) o.halo = need(i);
            else if (!std::strcmp(a, "--decomp")) o.decomp = need(i);
            else if (!std::strcmp(a, "--gpu-aware")) o.gpu_aware = std::stoi(need(i));
            else if (!std::strcmp(a, "--sync-timers")) o.sync_timers = std::stoi(need(i)) != 0;
            else throw std::invalid_argument(std::string("unknown option ") + a);
        }
        if (o.dim != 2 && o.dim != 3) throw std::invalid_argument("--dim must be 2 or 3");
        if (o.radius != 1 && o.radius != 2) throw std::invalid_argument("--radius must be 1 or 2");
        if (o.dim == 2) o.ng[2] = 1;
        if (o.verify && o.problem != Problem::MMS)
            throw std::invalid_argument("--verify needs --problem mms (no exact solution for ones)");
        if (o.local_n > 0) {
            if (o.local_n < o.radius) throw std::invalid_argument("--local-n must be >= radius");
            for (int d = 0; d < 3; ++d) o.ng[d] = d < o.dim ? o.local_n : 1;  // placeholder until the process grid is known
        }
        for (int d = 0; d < o.dim; ++d)
            if (o.ng[d] < 2 * o.radius && o.local_n == 0)
                throw std::invalid_argument("grid size missing or too small (use --n)");
        if (o.repeat < 1) throw std::invalid_argument("--repeat must be >= 1");
        if (o.halo != "block" && o.halo != "overlap")
            throw std::invalid_argument("--halo must be block or overlap");
        if (o.decomp != "auto" && o.decomp != "slab")
            throw std::invalid_argument("--decomp must be auto or slab");
    } catch (const std::exception& e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        usage(program);
        std::exit(2);
    }
    return o;
}

}  // namespace poisson
