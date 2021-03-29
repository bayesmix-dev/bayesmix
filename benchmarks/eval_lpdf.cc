#include <benchmark/benchmark.h>

#include "src/includes.h"
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

Eigen::MatrixXd get_grid(int dim) {
  Eigen::MatrixXd out;
  if (dim == 1) {
    out.resize(100, 1);
    out.col(0) = Eigen::ArrayXd::LinSpaced(100, -10, 10);
  } else if (dim == 2) {
    Eigen::VectorXd basegrid = Eigen::ArrayXd::LinSpaced(10, -10, 10);
    out.resize(100, 2);
    for (int i = 0; i < basegrid.size(); i++) {
      for (int j = 0; j < basegrid.size(); j++) {
        Eigen::VectorXd curr(2);
        curr << basegrid(i), basegrid(j);
        out.row(i * basegrid.size() + j) = curr;
      }
    }
  }
  return out;
}

static void BM_eval_lpdf_memory_read(benchmark::State& state) {
  int dim = state.range(0);
  std::shared_ptr<BaseAlgorithm> algo = get_algorithm("Neal2", dim);
  Eigen::MatrixXd grid = get_grid(dim);
  MemoryCollector collector;
  for (auto _ : state) {
    collector.read_from_file<bayesmix::AlgorithmState>(
        get_chain_file("Neal2", dim));
    algo->eval_lpdf(&collector, grid);
  }
}

static void BM_eval_lpdf_memory_noread(benchmark::State& state) {
  int dim = state.range(0);
  std::shared_ptr<BaseAlgorithm> algo = get_algorithm("Neal2", dim);
  Eigen::MatrixXd grid = get_grid(dim);
  MemoryCollector collector;
  collector.read_from_file<bayesmix::AlgorithmState>(
      get_chain_file("Neal2", dim));
  for (auto _ : state) {
    algo->eval_lpdf(&collector, grid);
  }
}

static void BM_eval_lpdf_file(benchmark::State& state) {
  int dim = state.range(0);
  std::shared_ptr<BaseAlgorithm> algo = get_algorithm("Neal2", dim);
  Eigen::MatrixXd grid = get_grid(dim);
  FileCollector collector(get_chain_file("Neal2", dim));
  for (auto _ : state) {
    algo->eval_lpdf(&collector, grid);
  }
}

BENCHMARK(BM_eval_lpdf_memory_read)->Arg(1)->Arg(2);
BENCHMARK(BM_eval_lpdf_memory_noread)->Arg(1)->Arg(2);
// BENCHMARK(BM_eval_lpdf_file)->Arg(1)->Arg(2);
