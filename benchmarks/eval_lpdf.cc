#include <benchmark/benchmark.h>

#include "src/includes.h"
#include "src/utils/eigen_utils.h"
#include "utils.h"

std::string get_chain_file(std::string algo_id, int dim) {
  std::string fname;
  if (dim == 1) {
    fname = "../resources/benchmarks/chains/" + algo_id +
            "_univariate_gaussian.recordio";
  } else {
    fname = "../resources/benchmarks/chains/" + algo_id +
            "_multi_gaussian_dim_ " + std::to_string(dim) + ".recordio";
  }
  return fname;
}

Eigen::MatrixXd get_data(int dim) {
  const char delim = ' ';
  Eigen::MatrixXd out;
  if (dim == 1) {
    out = bayesmix::read_eigen_matrix(
        "../resources/benchmarks/datasets/univariate_gaussian.csv", delim);
  } else {
    out = bayesmix::read_eigen_matrix(
        "../resources/benchmarks/datasets/multi_gaussian_dim_" +
            std::to_string(dim) + ".csv",
        delim);
  }
  return out;
}

Eigen::MatrixXd get_grid(int dim) {
  Eigen::MatrixXd out;
  if (dim == 1) {
    out.resize(100, 1);
    out.col(0) = Eigen::ArrayXd::LinSpaced(100, -10, 10);
  } else if (dim == 2) {
    out = bayesmix::get_2d_grid(-10, 10, 10, -10, 10, 10);
  }
  return out;
}

static void BM_eval_lpdf_memory_read_seq(benchmark::State& state) {
  int dim = state.range(0);
  std::shared_ptr<BaseAlgorithm> algo = get_algorithm("Neal2", dim);
  Eigen::MatrixXd grid = get_grid(dim);
  MemoryCollector collector;
  algo->set_data(get_data(dim));
  algo->run(&collector);
  for (auto _ : state) {
    algo->eval_lpdf(&collector, grid);
  }
}

static void BM_eval_lpdf_memory_read_par(benchmark::State& state) {
  int dim = state.range(0);
  std::shared_ptr<BaseAlgorithm> algo = get_algorithm("Neal2", dim);
  Eigen::MatrixXd grid = get_grid(dim);
  MemoryCollector collector;
  algo->set_data(get_data(dim));
  algo->run(&collector);
  for (auto _ : state) {
    bayesmix::eval_lpdf_parallel(algo, &collector, grid);
  }
}

// static void BM_eval_lpdf_memory_noread(benchmark::State& state) {
//   int dim = state.range(0);
//   std::shared_ptr<BaseAlgorithm> algo = get_algorithm("Neal2", dim);
//   Eigen::MatrixXd grid = get_grid(dim);
//   MemoryCollector collector;
//   collector.read_from_file<bayesmix::AlgorithmState>(
//       get_chain_file("Neal2", dim));
//   for (auto _ : state) {
//       bayesmix::eval_lpdf_parallel(algo, &collector, grid);
//   }
// }

// static void BM_eval_lpdf_file(benchmark::State& state) {
//   int dim = state.range(0);
//   std::shared_ptr<BaseAlgorithm> algo = get_algorithm("Neal2", dim);
//   Eigen::MatrixXd grid = get_grid(dim);
//   FileCollector collector(get_chain_file("Neal2", dim));
//   for (auto _ : state) {
//     bayesmix::eval_lpdf_parallel(algo, &collector, grid);
//   }
// }

BENCHMARK(BM_eval_lpdf_memory_read_seq)->Arg(1)->Arg(2);
BENCHMARK(BM_eval_lpdf_memory_read_par)->Arg(1)->Arg(2);

// BENCHMARK(BM_eval_lpdf_memory_noread)->Arg(1)->Arg(2);
// BENCHMARK(BM_eval_lpdf_file)->Arg(1)->Arg(2);
