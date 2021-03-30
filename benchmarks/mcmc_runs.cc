#include <benchmark/benchmark.h>

#include "src/includes.h"
#include "utils.h"

void run(std::shared_ptr<BaseAlgorithm>& algorithm,
         const Eigen::MatrixXd& data, MemoryCollector* collector) {
  algorithm->set_data(data);
  algorithm->run(collector);
}

Eigen::MatrixXd get_data(int dim) {
  Eigen::MatrixXd out;
  if (dim == 1) {
    out = bayesmix::read_eigen_matrix(
        "../resources/benchmarks/datasets/univariate_gaussian.csv");
  } else {
    out = bayesmix::read_eigen_matrix(
        "../resources/benchmarks/datasets/multi_gaussian_dim_" +
        std::to_string(dim) + ".csv");
  }
  return out;
}

std::string get_output_file(std::string algo_id, int dim) {
  std::string outfile;
  if (dim == 1) {
    outfile = "../resources/benchmarks/chains/" + algo_id +
              "_univariate_gaussian.recordio";
  } else {
    outfile = "../resources/benchmarks/chains/" + algo_id +
              "_multi_gaussian_dim_ " + std::to_string(dim) + ".recordio";
  }
  return outfile;
}

static void BM_Neal2(benchmark::State& state) {
  int dim = state.range(0);
  Eigen::MatrixXd data = get_data(dim);
  MemoryCollector collector;
  for (auto _ : state) {
    std::shared_ptr<BaseAlgorithm> algo = get_algorithm("Neal2", dim);
    run(algo, data, &collector);
  }
  collector.write_to_file<bayesmix::AlgorithmState>(
      get_output_file("Neal2", dim));
}

static void BM_Neal3(benchmark::State& state) {
  int dim = state.range(0);
  Eigen::MatrixXd data = get_data(dim);
  MemoryCollector collector;
  for (auto _ : state) {
    std::shared_ptr<BaseAlgorithm> algo = get_algorithm("Neal3", dim);
    run(algo, data, &collector);
  }
  collector.write_to_file<bayesmix::AlgorithmState>(
      get_output_file("Neal3", dim));
}

static void BM_Neal8(benchmark::State& state) {
  int dim = state.range(0);
  Eigen::MatrixXd data = get_data(dim);
  MemoryCollector collector;
  for (auto _ : state) {
    std::shared_ptr<BaseAlgorithm> algo = get_algorithm("Neal8", dim);
    run(algo, data, &collector);
  }
  collector.write_to_file<bayesmix::AlgorithmState>(
      get_output_file("Neal8", dim));
}

BENCHMARK(BM_Neal2)->Arg(1)->Arg(2)->Arg(4)->Arg(8);
BENCHMARK(BM_Neal3)->Arg(1)->Arg(2)->Arg(4)->Arg(8);
BENCHMARK(BM_Neal8)->Arg(1)->Arg(2)->Arg(4)->Arg(8);
