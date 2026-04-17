#include "theblas/theblas.h"
#include <cstdio>
#include <cmath>

int main() {
    float v[] = {3.0F, 4.0F};
    float n = theblas::snrm2(2, v, 1);
    std::printf("snrm2([3,4]) = %.4f\n", n);
    return (std::fabs(n - 5.0F) < 1e-5F) ? 0 : 1;
}
