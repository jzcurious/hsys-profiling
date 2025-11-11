#include <benchmark/benchmark.h>
#include <core/vector.cuh>
#include <pack/kernel_cmd.cuh>
#include <task/cmd.cuh>
#include <task/task.cuh>

#include "../include/cuda_timer.cuh"

static void BM_pack(benchmark::State& state) {
  auto size = state.range(0);

  hsys::Vector<float> x1(size);
  hsys::Vector<float> x2(size);
  hsys::Vector<float> x3(size);
  hsys::Vector<float> x4(size);
  hsys::Vector<float> y(size);

  for (auto _ : state) {
    float elapsed_time = 0;

    {
      CUDATimer timer(&elapsed_time);
      // clang-format off
      hsys::run_pack(
        hsys::Task(hsys::Add{}, y.view(), x1.view(), x2.view()), // +
        hsys::Task(hsys::Mul{}, y.view(),  y.view(), x2.view()), // *
        hsys::Task(hsys::Sub{}, y.view(),  y.view(), x3.view()), // -
        hsys::Task(hsys::Div{}, y.view(),  y.view(), x4.view())  // /
      );
      // clang-format on
    }

    benchmark::DoNotOptimize(elapsed_time);
    benchmark::ClobberMemory();

    state.SetIterationTime(elapsed_time);
  }
}

static void BM_unpack(benchmark::State& state) {
  auto size = state.range(0);

  hsys::Vector<float> x1(size);
  hsys::Vector<float> x2(size);
  hsys::Vector<float> x3(size);
  hsys::Vector<float> x4(size);
  hsys::Vector<float> y(size);

  for (auto _ : state) {
    float elapsed_time = 0;

    {
      CUDATimer timer(&elapsed_time);
      // clang-format off
      hsys::run_pack(hsys::Task(hsys::Add{}, y.view(), x1.view(), x2.view()));  // +
      hsys::run_pack(hsys::Task(hsys::Mul{}, y.view(),  y.view(), x2.view()));  // *
      hsys::run_pack(hsys::Task(hsys::Sub{}, y.view(),  y.view(), x3.view()));  // -
      hsys::run_pack(hsys::Task(hsys::Div{}, y.view(),  y.view(), x4.view()));  // /
      // clang-format on
    }

    benchmark::DoNotOptimize(elapsed_time);
    benchmark::ClobberMemory();

    state.SetIterationTime(elapsed_time);
  }
}

constexpr int multiplier = 8;
constexpr auto range = std::make_pair(8, 1 << 26);
constexpr auto unit = benchmark::kMillisecond;

BENCHMARK(BM_pack)
    ->Name("pack")
    ->RangeMultiplier(multiplier)
    ->Ranges({range})
    ->Unit(unit)
    ->UseManualTime();

BENCHMARK(BM_unpack)
    ->Name("unpack")
    ->RangeMultiplier(multiplier)
    ->Ranges({range})
    ->Unit(unit)
    ->UseManualTime();
