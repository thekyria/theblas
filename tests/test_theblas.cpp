#include "theblas/theblas.h"
#include <cassert>
int main() {
    assert(theblas::add(1.0f, 2.0f) == 3.0f);
    return 0;
}
