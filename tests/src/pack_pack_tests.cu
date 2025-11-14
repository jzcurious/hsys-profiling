#include <core/vector.cuh>
#include <fill_rand.cuh>
#include <gtest/gtest.h>
#include <pack/kernel_cmd_pack.cuh>
#include <task/cmd.cuh>
#include <task/task.cuh>
#include <vector>

void* operator new(std::size_t bytes);  // Dumb clangd!

class PackPackTest : public ::testing::TestWithParam<std::tuple<std::size_t, float>> {
 protected:
  bool pack_test_impl(std::size_t size, float tol) {
    hsys::Vector<float> x1(size);
    hsys::Vector<float> x2(size);
    hsys::Vector<float> x3(size);
    hsys::Vector<float> x4(size);
    hsys::Vector<float> y(size);

    fill_rand(x1, 1);
    fill_rand(x2, 1);
    fill_rand(x3, 1);
    fill_rand(x4, 1);

    // clang-format off
    hsys::run_pack_pack(
      hsys::Task(hsys::Add{}, y.view(), x1.view(), x2.view()), // +
      hsys::Task(hsys::Mul{}, y.view(),  y.view(), x2.view()), // *
      hsys::Task(hsys::Sub{}, y.view(),  y.view(), x3.view()), // -
      hsys::Task(hsys::Div{}, y.view(),  y.view(), x4.view())  // /
    );
    // clang-format on

    std::vector<float> std_x1(size);
    std::vector<float> std_x2(size);
    std::vector<float> std_x3(size);
    std::vector<float> std_x4(size);
    std::vector<float> y_from_device(size);

    x1.data().copy_to_host(std_x1.data());
    x2.data().copy_to_host(std_x2.data());
    x3.data().copy_to_host(std_x3.data());
    x4.data().copy_to_host(std_x4.data());
    y.data().copy_to_host(y_from_device.data());

    for (std::size_t i = 0; i < size; ++i) {
      float y_i = std_x1[i] + std_x2[i];
      y_i *= std_x2[i];
      y_i -= std_x3[i];
      y_i /= std_x4[i];
      if (std::abs(y_from_device[i] - y_i) > tol) return false;
    }

    return true;
  }
};

TEST_P(PackPackTest, pack_test) {
  auto [size, tol] = GetParam();
  EXPECT_TRUE(pack_test_impl(size, tol));
}

// clang-format off
INSTANTIATE_TEST_SUITE_P(
    PackTestSuite,
    PackPackTest,
    ::testing::Combine(
      ::testing::Values(1, 2, 3, 127, 128, 129, 512, 513, 1023, 1024),
      ::testing::Values(1e-5)
    )
);
// clang-format on
