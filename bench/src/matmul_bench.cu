#include <benchmark/benchmark.h>
#include <core/matrix.cuh>
#include <matmul/matmul.cuh>
#include <matmul/matmul_extend_adown.cuh>
#include <matmul/matmul_extend_bright.cuh>
#include <matmul/matmul_extend_symn.cuh>

#include "../include/cuda_timer.cuh"

template <auto matmul_impl_wrapper>
static void BM_matmul(benchmark::State& state) {
  auto size = state.range(0);

  auto a = hsys::Matrix<float>(size, size);
  auto b = hsys::Matrix<float>(size, size);
  auto c = hsys::Matrix<float>(size, size);

  for (auto _ : state) {
    float elapsed_time = 0;

    {
      CUDATimer timer(&elapsed_time);
      matmul_impl_wrapper.operator()(c, a, b);
    }

    benchmark::DoNotOptimize(elapsed_time);
    benchmark::ClobberMemory();

    state.SetIterationTime(elapsed_time);
  }
}

inline constexpr int multiplier = 2;  // NOLINT
inline constexpr auto range = std::make_pair(16, 1 << 12);  // NOLINT
inline constexpr auto unit = benchmark::kMillisecond;  // NOLINT

using Matrix = hsys::Matrix<float>;

static const auto matmul = [](Matrix& c, const Matrix& a, const Matrix& b) {
  hsys::matmul<>(c, a, b);
};

static const auto matmul_extend_bright = [](Matrix& c, const Matrix& a, const Matrix& b) {
  hsys::matmul_extend_bright<>(c, a, b);
};

static const auto matmul_extend_adown = [](Matrix& c, const Matrix& a, const Matrix& b) {
  hsys::matmul_extend_adown<>(c, a, b);
};

static const auto matmul_extend_sym_t16_n2
    = [](Matrix& c, const Matrix& a, const Matrix& b) {
        hsys::matmul_extend_symn<16, 2>(c, a, b);
      };

static const auto matmul_extend_sym_t16_n4
    = [](Matrix& c, const Matrix& a, const Matrix& b) {
        hsys::matmul_extend_symn<16, 4>(c, a, b);
      };

static const auto matmul_extend_sym_t16_n5
    = [](Matrix& c, const Matrix& a, const Matrix& b) {
        hsys::matmul_extend_symn<16, 4>(c, a, b);
      };

static const auto matmul_extend_sym_t32_n2
    = [](Matrix& c, const Matrix& a, const Matrix& b) {
        hsys::matmul_extend_symn<16, 4>(c, a, b);
      };

static const auto matmul_extend_sym_t32_n4
    = [](Matrix& c, const Matrix& a, const Matrix& b) {
        hsys::matmul_extend_symn<16, 4>(c, a, b);
      };

BENCHMARK(BM_matmul<matmul>)
    ->Name("matmul")
    ->RangeMultiplier(multiplier)
    ->Ranges({range})
    ->Unit(unit)
    ->UseManualTime();

BENCHMARK(BM_matmul<matmul_extend_bright>)
    ->Name("matmul_extend_bright")
    ->RangeMultiplier(multiplier)
    ->Ranges({range})
    ->Unit(unit)
    ->UseManualTime();

BENCHMARK(BM_matmul<matmul_extend_adown>)
    ->Name("matmul_extend_adown")
    ->RangeMultiplier(multiplier)
    ->Ranges({range})
    ->Unit(unit)
    ->UseManualTime();

BENCHMARK(BM_matmul<matmul_extend_sym_t16_n2>)
    ->Name("matmul_extend_sym_t16_n2")
    ->RangeMultiplier(multiplier)
    ->Ranges({range})
    ->Unit(unit)
    ->UseManualTime();

BENCHMARK(BM_matmul<matmul_extend_sym_t16_n4>)
    ->Name("matmul_extend_sym_t16_n4")
    ->RangeMultiplier(multiplier)
    ->Ranges({range})
    ->Unit(unit)
    ->UseManualTime();

BENCHMARK(BM_matmul<matmul_extend_sym_t16_n5>)
    ->Name("matmul_extend_sym_t16_n5")
    ->RangeMultiplier(multiplier)
    ->Ranges({range})
    ->Unit(unit)
    ->UseManualTime();

BENCHMARK(BM_matmul<matmul_extend_sym_t32_n2>)
    ->Name("matmul_extend_sym_t32_n2")
    ->RangeMultiplier(multiplier)
    ->Ranges({range})
    ->Unit(unit)
    ->UseManualTime();

BENCHMARK(BM_matmul<matmul_extend_sym_t32_n4>)
    ->Name("matmul_extend_sym_t32_n4")
    ->RangeMultiplier(multiplier)
    ->Ranges({range})
    ->Unit(unit)
    ->UseManualTime();
