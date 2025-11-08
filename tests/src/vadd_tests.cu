#include <algorithm>
#include <cmath>
#include <fill_rand.cuh>
#include <gtest/gtest.h>
#include <vadd/vadd.cuh>
#include <vadd/vadd_persist.cuh>

#include <vector>

void* operator new(std::size_t bytes);  // Dumb clangd!

class VectorAddTest : public ::testing::TestWithParam<std::tuple<std::size_t, float>> {
 protected:
  bool vadd_test_impl(std::size_t size, float tol) {
    hsys::Vector<float> a(size);
    hsys::Vector<float> b(size);
    hsys::Vector<float> c1(size);
    hsys::Vector<float> c2(size);

    fill_rand(a);
    fill_rand(b);
    hsys::vadd(c1, a, b);

    hsys::vadd_persist(c2, a, b);
    hsys::vadd_persist(c2, a, b);
    hsys::vadd_persist(c2, a, b);

    std::vector<float> hc1(size);
    std::vector<float> hc2(size);

    c1.data().copy_to_host(hc1.data());
    c2.data().copy_to_host(hc2.data());

    return std::equal(hc1.begin(), hc1.end(), hc2.begin(), [tol](float x, float y) {
      return std::abs(x - y) < tol;
    });
  }
};

TEST_P(VectorAddTest, vadd_test) {
  auto [size, tol] = GetParam();
  EXPECT_TRUE(vadd_test_impl(size, tol));
}

// clang-format off
INSTANTIATE_TEST_SUITE_P(
    VectorAddTestSuite,
    VectorAddTest,
    ::testing::Combine(
      ::testing::Values(1, 2, 3, 127, 128, 129, 512, 513, 1023, 1024),
      ::testing::Values(1e-6)
    )
);
// clang-format on
