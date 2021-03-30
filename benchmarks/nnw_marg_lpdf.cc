#include <benchmark/benchmark.h>

#include <Eigen/Dense>
#include <iostream>

#include "utils.h"

static void BM_NNWPriorPred(benchmark::State& state) {
  int dim = state.range(0);
  auto hierarchy = get_multivariate_nnw_hierarchy(dim);
  Eigen::VectorXd x = Eigen::VectorXd::Zero(dim);
  for (auto _ : state) {
    hierarchy->prior_pred_lpdf(x);
  }
}

static void BM_NNWSampleFullCond(benchmark::State& state) {
  int dim = state.range(0);
  auto hierarchy = get_multivariate_nnw_hierarchy(dim);
  Eigen::MatrixXd data = Eigen::MatrixXd::Random(10, dim);
  for (auto _ : state) {
    hierarchy->initialize();
    for (int i = 0; i < 10; i++) {
      hierarchy->add_datum(i, data.row(i));
    }
    hierarchy->sample_full_cond();
  }
}

static void BM_NNWConditionalPred(benchmark::State& state) {
  int dim = state.range(0);
  auto hierarchy = get_multivariate_nnw_hierarchy(dim);
  Eigen::MatrixXd data = Eigen::MatrixXd::Random(10, dim);
  for (int i = 0; i < 10; i++) {
    hierarchy->add_datum(i, data.row(i));
  }
  Eigen::VectorXd x = Eigen::VectorXd::Zero(dim);

  for (auto _ : state) {
    std::dynamic_pointer_cast<NNWHierarchy>(hierarchy)
        ->save_posterior_hypers();
    hierarchy->conditional_pred_lpdf(x);
  }
}

BENCHMARK(BM_NNWPriorPred)->RangeMultiplier(2)->Range(2, 2 << 5);
BENCHMARK(BM_NNWSampleFullCond)->RangeMultiplier(2)->Range(2, 2 << 5);
BENCHMARK(BM_NNWConditionalPred)->RangeMultiplier(2)->Range(2, 2 << 5);
