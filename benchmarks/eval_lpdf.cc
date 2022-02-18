#include <benchmark/benchmark.h>

#include "src/includes.h"
#include "src/utils/eigen_utils.h"
#include "src/utils/testing_utils.h"

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
    out.resize(1000, 1);
    out.col(0) = Eigen::ArrayXd::LinSpaced(1000, -10, 10);
  } else if (dim == 2) {
    out = bayesmix::get_2d_grid(-10, 10, 100, -10, 10, 100);
  }
  return out;
}

static void BM_eval_lpdf_memory_seq(benchmark::State& state) {
  int dim = state.range(0);
  Eigen::MatrixXd grid = get_grid(dim);
  std::shared_ptr<BaseAlgorithm> algo = get_algorithm("Neal2", dim);
  MemoryCollector collector;
  algo->set_data(get_data(dim));
  algo->run(&collector);
  for (auto _ : state) {
    algo->eval_lpdf(&collector, grid);
  }
}

static void BM_eval_lpdf_memory_par_lowmem(benchmark::State& state) {
  int dim = state.range(0);
  Eigen::MatrixXd grid = get_grid(dim);
  std::shared_ptr<BaseAlgorithm> algo = get_algorithm("Neal2", dim);
  MemoryCollector collector;
  algo->set_data(get_data(dim));
  algo->run(&collector);
  for (auto _ : state) {
    bayesmix::internal::eval_lpdf_parallel_lowmemory(algo, &collector, grid,
                                                     100);
  }
}

static void BM_eval_lpdf_memory_par_fullmem(benchmark::State& state) {
  int dim = state.range(0);
  Eigen::MatrixXd grid = get_grid(dim);
  std::shared_ptr<BaseAlgorithm> algo = get_algorithm("Neal2", dim);
  MemoryCollector collector;
  algo->set_data(get_data(dim));
  algo->run(&collector);
  for (auto _ : state) {
    bayesmix::internal::eval_lpdf_parallel_fullmemory(algo, &collector, grid,
                                                      8);
  }
}

static void BM_eval_lpdf_file_seq(benchmark::State& state) {
  int dim = state.range(0);
  Eigen::MatrixXd grid = get_grid(dim);
  std::shared_ptr<BaseAlgorithm> algo = get_algorithm("Neal2", dim);
  FileCollector collector(get_chain_file("Neal2", dim));
  algo->set_data(get_data(dim));
  algo->run(&collector);
  for (auto _ : state) {
    std::cout << "one" << std::endl;
    algo->eval_lpdf(&collector, grid);
  }
}

static void BM_eval_lpdf_file_par_lowmem(benchmark::State& state) {
  int dim = state.range(0);
  Eigen::MatrixXd grid = get_grid(dim);
  std::shared_ptr<BaseAlgorithm> algo = get_algorithm("Neal2", dim);
  FileCollector collector(get_chain_file("Neal2", dim));
  algo->set_data(get_data(dim));
  algo->run(&collector);
  for (auto _ : state) {
    bayesmix::internal::eval_lpdf_parallel_lowmemory(algo, &collector, grid,
                                                     100);
  }
}

static void BM_eval_lpdf_file_par_fullmem(benchmark::State& state) {
  int dim = state.range(0);
  Eigen::MatrixXd grid = get_grid(dim);
  std::shared_ptr<BaseAlgorithm> algo = get_algorithm("Neal2", dim);
  FileCollector collector(get_chain_file("Neal2", dim));
  algo->set_data(get_data(dim));
  algo->run(&collector);
  for (auto _ : state) {
    bayesmix::internal::eval_lpdf_parallel_fullmemory(algo, &collector, grid,
                                                      8);
  }
}

BENCHMARK(BM_eval_lpdf_memory_seq)->Arg(1)->Arg(2);
BENCHMARK(BM_eval_lpdf_memory_par_lowmem)->Arg(1)->Arg(2);
BENCHMARK(BM_eval_lpdf_memory_par_fullmem)->Arg(1)->Arg(2);

BENCHMARK(BM_eval_lpdf_file_seq)->Arg(1)->Arg(2);
BENCHMARK(BM_eval_lpdf_file_par_lowmem)->Arg(1)->Arg(2);
BENCHMARK(BM_eval_lpdf_file_par_fullmem)->Arg(1)->Arg(2);
