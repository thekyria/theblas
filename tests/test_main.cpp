#include "test_cases.hpp"

int main() {
    theblas::test::run_level1_tests();
    theblas::test::run_level2_core_tests();
    theblas::test::run_level2_coverage_variant_tests();
    return 0;
}