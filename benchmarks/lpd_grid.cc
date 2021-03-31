#include <benchmark/benchmark.h>

#include <Eigen/Dense>
#include <iostream>

#include "src/utils/distributions.h"
#include "utils.h"

Eigen::VectorXd lpdf_cov(const Eigen::MatrixXd &x, const Eigen::VectorXd &mean,
                         const Eigen::MatrixXd &cov) {
  Eigen::VectorXd out(x.rows());
  for (int i = 0; i < x.rows(); i++) {
    out(i) = stan::math::multi_normal_lpdf(x.row(i), mean, cov);
  }
  return out;
}

Eigen::VectorXd lpdf_prec(const Eigen::MatrixXd &x,
                          const Eigen::VectorXd &mean,
                          const Eigen::MatrixXd &prec) {
  Eigen::VectorXd out(x.rows());
  for (int i = 0; i < x.rows(); i++) {
    out(i) = stan::math::multi_normal_prec_lpdf(x.row(i), mean, prec);
  }
  return out;
}

Eigen::VectorXd lpdf_naive(const Eigen::MatrixXd &x,
                           const Eigen::VectorXd &mean,
                           const Eigen::MatrixXd &prec_chol,
                           double prec_logdet) {
  Eigen::VectorXd out(x.rows());
  for (int i = 0; i < x.rows(); i++) {
    out(i) = bayesmix::multi_normal_prec_lpdf(x.row(i), mean, prec_chol,
                                              prec_logdet);
  }
  return out;
}


Eigen::VectorXd lpdf_fully_optimized(const Eigen::MatrixXd &x,
                                     const Eigen::VectorXd &mean,
                                     const Eigen::MatrixXd &prec_chol,
                                     double prec_logdet) {
  using stan::math::NEG_LOG_SQRT_TWO_PI;
  Eigen::VectorXd exp =
      ((x.rowwise() - mean.transpose()) * prec_chol).rowwise().squaredNorm();
  Eigen::VectorXd base = Eigen::ArrayXd::Ones(x.rows()) * prec_logdet +
                         NEG_LOG_SQRT_TWO_PI * x.cols();
  return (base - exp) * 0.5;
}

static void BM_gauss_lpdf_cov(benchmark::State &state) {
  int dim = state.range(0);
  Eigen::VectorXd mean = Eigen::VectorXd::Zero(dim);
  Eigen::MatrixXd cov = get_spd_matrix(dim);
  Eigen::MatrixXd x = Eigen::MatrixXd::Ones(200, dim);
  for (auto _ : state) {
    lpdf_cov(x, mean, cov);
  }
}

static void BM_gauss_lpdf_prec(benchmark::State &state) {
  int dim = state.range(0);
  Eigen::VectorXd mean = Eigen::VectorXd::Zero(dim);
  Eigen::MatrixXd prec = get_spd_matrix(dim);
  Eigen::MatrixXd x = Eigen::MatrixXd::Ones(200, dim);
  for (auto _ : state) {
    lpdf_prec(x, mean, prec);
  }
}

static void BM_gauss_lpdf_naive(benchmark::State &state) {
  int dim = state.range(0);
  Eigen::VectorXd mean = Eigen::VectorXd::Zero(dim);
  Eigen::MatrixXd prec = get_spd_matrix(dim);
  Eigen::MatrixXd prec_chol = Eigen::LLT<Eigen::MatrixXd>(prec).matrixU();
  Eigen::VectorXd diag = prec_chol.diagonal();
  double prec_logdet = 2 * log(diag.array()).sum();
  Eigen::MatrixXd x = Eigen::MatrixXd::Ones(200, dim);

  for (auto _ : state) {
    lpdf_naive(x, mean, prec_chol, prec_logdet);
  }
}


static void BM_gauss_lpdf_fully_optimized(benchmark::State &state) {
  int dim = state.range(0);
  Eigen::VectorXd mean = Eigen::VectorXd::Zero(dim);
  Eigen::MatrixXd prec = get_spd_matrix(dim);
  Eigen::MatrixXd prec_chol = Eigen::LLT<Eigen::MatrixXd>(prec).matrixU();
  Eigen::VectorXd diag = prec_chol.diagonal();
  double prec_logdet = 2 * log(diag.array()).sum();
  Eigen::MatrixXd x = Eigen::MatrixXd::Ones(200, dim);

  for (auto _ : state) {
    lpdf_fully_optimized(x, mean, prec_chol, prec_logdet);
  }
}


BENCHMARK(BM_gauss_lpdf_cov)->RangeMultiplier(2)->Range(2, 2 << 4);
BENCHMARK(BM_gauss_lpdf_cov)->RangeMultiplier(2)->Range(2, 2 << 4);
BENCHMARK(BM_gauss_lpdf_naive)->RangeMultiplier(2)->Range(2, 2 << 4);
BENCHMARK(BM_gauss_lpdf_fully_optimized)->RangeMultiplier(2)->Range(2, 2 << 4);
