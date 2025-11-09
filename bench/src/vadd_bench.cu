#include <benchmark/benchmark.h>
#include <vadd/vadd.cuh>
#include <vadd/vadd_persist.cuh>

#include "../include/cuda_timer.cuh"

template <auto vadd_impl_wrapper>
static void BM_vadd(benchmark::State& state) {
  auto size = state.range(0);

  auto a = hsys::Vector<float>(size);
  auto b = hsys::Vector<float>(size);
  auto c = hsys::Vector<float>(size);

  for (auto _ : state) {
    float elapsed_time = 0;

    {
      CUDATimer timer(&elapsed_time);
      vadd_impl_wrapper.operator()(c, a, b);
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

static const auto vadd = [](Vector& c, const Vector& a, const Vector& b) {
  hsys::vadd(c, a, b);
};

static const auto vadd_persist = [](Vector& c, const Vector& a, const Vector& b) {
  hsys::vadd_persist(c, a, b);
};

BENCHMARK(BM_vadd<vadd>)
    ->Name("vadd")
    ->RangeMultiplier(multiplier)
    ->Ranges({range})
    ->Unit(unit)
    ->UseManualTime();

BENCHMARK(BM_vadd<vadd_persist>)
    ->Name("vadd_persist")
    ->RangeMultiplier(multiplier)
    ->Ranges({range})
    ->Unit(unit)
    ->UseManualTime();
