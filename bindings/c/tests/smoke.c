/* Smoke test for the C ABI. Compiled as C99 to validate theblas_c.h as a C header. */
#include "theblas/theblas_c.h"

#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <string.h>

static int close_to(double got, double want) {
    return fabs(got - want) < 1e-6;
}

static char last_routine[16];
static int last_param;

static void record_error(const char *routine, int param) {
    strncpy(last_routine, routine, sizeof last_routine - 1);
    last_param = param;
}

int main(void) {
    /* Level 1, real: y <- 0.5 x + y, 1-based iamax. */
    double x[] = {1.0, -4.0, 3.0};
    double y[] = {10.0, 20.0, 30.0};
    theblas_daxpy(3, 0.5, x, 1, y, 1);
    assert(close_to(y[0], 10.5) && close_to(y[1], 18.0) && close_to(y[2], 31.5));
    assert(theblas_idamax(3, x, 1) == 2);

    /* Level 1, complex: scalar passed as {re, im}, result returned through a pair. */
    float cx[] = {1.0F, 2.0F, 3.0F, -1.0F}; /* [1+2i, 3-i] */
    float cy[] = {0.0F, 1.0F, 2.0F, 0.0F};  /* [i, 2]      */
    float dot[2];
    theblas_cdotc(2, cx, 1, cy, 1, dot); /* conj(1+2i)*i + conj(3-i)*2 = 8 + 3i */
    assert(close_to(dot[0], 8.0) && close_to(dot[1], 3.0));

    const float alpha[] = {0.0F, 1.0F};    /* i */
    theblas_caxpy(2, alpha, cx, 1, cy, 1); /* y <- i*x + y = [-2+2i, 3+3i] */
    assert(close_to(cy[0], -2.0) && close_to(cy[1], 2.0));
    assert(close_to(cy[2], 3.0) && close_to(cy[3], 3.0));

    /* Level 3: C = A * B, column-major. */
    const double a[] = {1, 4, 2, 5, 3, 6};
    const double b[] = {7, 9, 11, 8, 10, 12};
    double c[4] = {0};
    theblas_dgemm('N', 'N', 2, 2, 3, 1.0, a, 2, b, 3, 0.0, c, 2);
    assert(close_to(c[0], 58) && close_to(c[1], 139) && close_to(c[2], 64) && close_to(c[3], 154));

    /* Error handler: an invalid TRANS is reported as parameter 1 of DGEMV. */
    theblas_error_handler_t previous = theblas_set_error_handler(record_error);
    theblas_dgemv('X', 2, 3, 1.0, a, 2, x, 1, 0.0, y, 1);
    assert(strcmp(last_routine, "DGEMV") == 0 && last_param == 1);
    theblas_set_error_handler(previous);

    puts("theblas_c smoke test passed");
    return 0;
}
