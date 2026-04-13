// benchmark.cpp
// Microbenchmark for core theblas Level-1 BLAS routines.
//
// Each routine is timed over WARMUP + RUNS iterations on a vector of N
// elements.  The first WARMUP iterations are discarded to let the CPU reach
// steady state and populate caches.  The best (minimum) elapsed time across
// the measurement runs is reported, which reflects peak throughput free of
// OS scheduling noise.
//
// Usage:
//   ./example_benchmark [N]     (default N = 1 048 576)
//
// The optional argument lets you sweep vector sizes to observe cache effects:
//   ./example_benchmark 256          # fits in L1
//   ./example_benchmark 16384        # fits in L2
//   ./example_benchmark 1048576      # spills to L3 / DRAM (default)

#include "theblas/theblas.h"

#include <algorithm>
#include <chrono>
#include <complex>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>

namespace {

constexpr int warmup = 5;
constexpr int runs   = 20;

using Clock     = std::chrono::steady_clock;
using Seconds   = std::chrono::duration<double>;

// Returns the minimum elapsed time (seconds) over runs iterations after
// warmup discarded warm-up passes.
template <typename Fn>
double time_min(Fn fn) {
    for (int i = 0; i < warmup; ++i) {
        fn();
    }
    double best = std::numeric_limits<double>::max();
    for (int i = 0; i < runs; ++i) {
        auto t0 = Clock::now();
        fn();
        auto t1 = Clock::now();
        best = std::min(best, Seconds(t1 - t0).count());
    }
    return best;
}

void print_row(const char* name, int n, double seconds, std::size_t bytes) {
    double gb_s = static_cast<double>(bytes) / seconds / 1e9;
    std::printf("  %-16s  n=%9d  %.3e s  %6.2f GB/s\n", name, n, seconds, gb_s);
}

} // namespace

int main(int argc, const char* const argv[]) {
    int n = 1 << 20; // 1 048 576
    if (argc >= 2) {
        n = std::atoi(argv[1]);
        if (n <= 0) {
            std::fprintf(stderr, "N must be > 0\n");
            return 1;
        }
    }

    std::printf("theblas microbenchmark  (n=%d, warmup=%d, runs=%d)\n\n",
                n, warmup, runs);

    // -----------------------------------------------------------------
    // Allocate and initialise vectors
    // -----------------------------------------------------------------
    std::vector<float>  xs(n, 1.0F);
    std::vector<float>  ys(n, 2.0F);
    std::vector<double> xd(n, 1.0);
    std::vector<double> yd(n, 2.0);
    std::vector<std::complex<float>>  xc(n, {1.0F, 0.5F});
    std::vector<std::complex<float>>  yc(n, {2.0F, 1.0F});
    std::vector<std::complex<double>> xz(n, {1.0, 0.5});
    std::vector<std::complex<double>> yz(n, {2.0, 1.0});

    // Keep originals so mutable routines can be reset each run
    std::vector<float>  ys0 = ys;
    std::vector<double> yd0 = yd;
    std::vector<std::complex<float>>  yc0 = yc;
    std::vector<std::complex<double>> yz0 = yz;

    std::printf("  %-16s  %-14s  %-12s  %-10s\n",
                "routine", "n", "best time", "bandwidth");
    std::puts("  -------------------------------------------------------"
              "-------------------");

    // -----------------------------------------------------------------
    // snrm2 / dnrm2 — reads n floats/doubles once
    // -----------------------------------------------------------------
    print_row("snrm2", n,
              time_min([&]{ theblas::snrm2(n, xs.data(), 1); }),
              static_cast<std::size_t>(n) * sizeof(float));

    print_row("dnrm2", n,
              time_min([&]{ theblas::dnrm2(n, xd.data(), 1); }),
              static_cast<std::size_t>(n) * sizeof(double));

    // -----------------------------------------------------------------
    // sasum / dasum — reads n elements once
    // -----------------------------------------------------------------
    print_row("sasum", n,
              time_min([&]{ theblas::sasum(n, xs.data(), 1); }),
              static_cast<std::size_t>(n) * sizeof(float));

    print_row("dasum", n,
              time_min([&]{ theblas::dasum(n, xd.data(), 1); }),
              static_cast<std::size_t>(n) * sizeof(double));

    // -----------------------------------------------------------------
    // sdot / ddot — reads 2n elements
    // -----------------------------------------------------------------
    print_row("sdot", n,
              time_min([&]{ theblas::sdot(n, xs.data(), 1, ys.data(), 1); }),
              std::size_t{2} * static_cast<std::size_t>(n) * sizeof(float));

    print_row("ddot", n,
              time_min([&]{ theblas::ddot(n, xd.data(), 1, yd.data(), 1); }),
              std::size_t{2} * static_cast<std::size_t>(n) * sizeof(double));

    // -----------------------------------------------------------------
    // saxpy / daxpy — reads 2n, writes n (3n bytes total)
    // -----------------------------------------------------------------
    print_row("saxpy", n,
              time_min([&]{
                  std::copy(ys0.begin(), ys0.end(), ys.begin());
                  theblas::saxpy(n, 0.5F, xs.data(), 1, ys.data(), 1);
              }),
              std::size_t{3} * static_cast<std::size_t>(n) * sizeof(float));

    print_row("daxpy", n,
              time_min([&]{
                  std::copy(yd0.begin(), yd0.end(), yd.begin());
                  theblas::daxpy(n, 0.5, xd.data(), 1, yd.data(), 1);
              }),
              std::size_t{3} * static_cast<std::size_t>(n) * sizeof(double));

    // -----------------------------------------------------------------
    // sscal / dscal — reads and writes n elements
    // -----------------------------------------------------------------
    print_row("sscal", n,
              time_min([&]{
                  std::copy(ys0.begin(), ys0.end(), ys.begin());
                  theblas::sscal(n, 2.0F, ys.data(), 1);
              }),
              std::size_t{2} * static_cast<std::size_t>(n) * sizeof(float));

    print_row("dscal", n,
              time_min([&]{
                  std::copy(yd0.begin(), yd0.end(), yd.begin());
                  theblas::dscal(n, 2.0, yd.data(), 1);
              }),
              std::size_t{2} * static_cast<std::size_t>(n) * sizeof(double));

    // -----------------------------------------------------------------
    // scopy / dcopy — reads n, writes n
    // -----------------------------------------------------------------
    print_row("scopy", n,
              time_min([&]{ theblas::scopy(n, xs.data(), 1, ys.data(), 1); }),
              std::size_t{2} * static_cast<std::size_t>(n) * sizeof(float));

    print_row("dcopy", n,
              time_min([&]{ theblas::dcopy(n, xd.data(), 1, yd.data(), 1); }),
              std::size_t{2} * static_cast<std::size_t>(n) * sizeof(double));

    // -----------------------------------------------------------------
    // sswap / dswap — reads 2n, writes 2n
    // -----------------------------------------------------------------
    print_row("sswap", n,
              time_min([&]{ theblas::sswap(n, xs.data(), 1, ys.data(), 1); }),
              std::size_t{4} * static_cast<std::size_t>(n) * sizeof(float));

    print_row("dswap", n,
              time_min([&]{ theblas::dswap(n, xd.data(), 1, yd.data(), 1); }),
              std::size_t{4} * static_cast<std::size_t>(n) * sizeof(double));

    // -----------------------------------------------------------------
    // Complex: caxpy / zaxpy
    // -----------------------------------------------------------------
    print_row("caxpy", n,
              time_min([&]{
                  std::copy(yc0.begin(), yc0.end(), yc.begin());
                  theblas::caxpy(n, {0.5F, 0.0F}, xc.data(), 1, yc.data(), 1);
              }),
              std::size_t{3} * static_cast<std::size_t>(n) * sizeof(std::complex<float>));

    print_row("zaxpy", n,
              time_min([&]{
                  std::copy(yz0.begin(), yz0.end(), yz.begin());
                  theblas::zaxpy(n, {0.5, 0.0}, xz.data(), 1, yz.data(), 1);
              }),
              std::size_t{3} * static_cast<std::size_t>(n) * sizeof(std::complex<double>));

    // -----------------------------------------------------------------
    // isamax / idamax
    // -----------------------------------------------------------------
    print_row("isamax", n,
              time_min([&]{ theblas::isamax(n, xs.data(), 1); }),
              static_cast<std::size_t>(n) * sizeof(float));

    print_row("idamax", n,
              time_min([&]{ theblas::idamax(n, xd.data(), 1); }),
              static_cast<std::size_t>(n) * sizeof(double));

    std::puts("");
    std::puts("Bandwidth is a lower bound (data may be served from cache).");
    std::puts("Run with different N to observe L1/L2/L3/DRAM effects:");
    std::puts("  ./example_benchmark 256       # L1");
    std::puts("  ./example_benchmark 16384     # L2");
    std::puts("  ./example_benchmark 1048576   # L3/DRAM");

    return 0;
}
