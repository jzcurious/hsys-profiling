#include <benchmark/benchmark.h>
#include <core/vector.cuh>
#include <dot/dot_atomic.cuh>
#include <dot/dot_coars.cuh>
#include <dot/dot_nlog.cuh>
#include <dot/dot_shmem.cuh>
#include <dot/dot_vect.cuh>

#include "../include/cuda_timer.cuh"

template <auto dot_impl_wrapper>
static void BM_dot(benchmark::State& state) {
  auto size = state.range(0);

  auto a = hsys::Vector<float>(size);
  auto b = hsys::Vector<float>(size);
  auto c = hsys::Vector<float>(1);

  for (auto _ : state) {
    float elapsed_time = 0;

    {
      CUDATimer timer(&elapsed_time);
      dot_impl_wrapper.operator()(c, a, b);
    }

    benchmark::DoNotOptimize(elapsed_time);
    benchmark::ClobberMemory();

    state.SetIterationTime(elapsed_time);
  }
}

constexpr int multiplier = 8;
constexpr auto range = std::make_pair(8, 1 << 26);
constexpr auto unit = benchmark::kMillisecond;

using Vector = hsys::Vector<float>;

static const auto dot_atomic = [](Vector& c, const Vector& a, const Vector& b) {
  hsys::dot_atomic<>(c, a, b);
};

static const auto dot_coars = [](Vector& c, const Vector& a, const Vector& b) {
  hsys::dot_coars<128>(c, a, b);
};

static const auto dot_vect = [](Vector& c, const Vector& a, const Vector& b) {
  hsys::dot_vect<32>(c, a, b);
};

static const auto dot_shmem = [](Vector& c, const Vector& a, const Vector& b) {
  hsys::dot_shmem<16>(c, a, b);
};

static const auto dot_nlog = [](Vector& c, const Vector& a, const Vector& b) {
  hsys::dot_nlog<128>(c, a, b);
};

BENCHMARK(BM_dot<dot_atomic>)
    ->Name("dot_atomic")
    ->RangeMultiplier(multiplier)
    ->Ranges({range})
    ->Unit(unit)
    ->UseManualTime();

BENCHMARK(BM_dot<dot_coars>)
    ->Name("dot_coars")
    ->RangeMultiplier(multiplier)
    ->Ranges({range})
    ->Unit(unit)
    ->UseManualTime();

BENCHMARK(BM_dot<dot_vect>)
    ->Name("dot_vect")
    ->RangeMultiplier(multiplier)
    ->Ranges({range})
    ->Unit(unit)
    ->UseManualTime();

BENCHMARK(BM_dot<dot_shmem>)
    ->Name("dot_shmem")
    ->RangeMultiplier(multiplier)
    ->Ranges({range})
    ->Unit(unit)
    ->UseManualTime();

BENCHMARK(BM_dot<dot_nlog>)
    ->Name("dot_nlog")
    ->RangeMultiplier(multiplier)
    ->Ranges({range})
    ->Unit(unit)
    ->UseManualTime();
