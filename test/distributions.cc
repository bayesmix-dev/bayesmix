#include "src/utils/distributions.h"

#include <gtest/gtest.h>

#include <Eigen/Dense>
#include <stan/math/prim.hpp>
#include <vector>

#include "src/utils/rng.h"

TEST(mix_dist, 1) {
  auto& rng = bayesmix::Rng::Instance().get();

  int nclus = 5;
  Eigen::VectorXd weights1 =
      stan::math::dirichlet_rng(Eigen::VectorXd::Ones(nclus), rng);
  Eigen::VectorXd means1(nclus);
  Eigen::VectorXd sds1(nclus);

  for (int i = 0; i < nclus; i++) {
    means1(i) = stan::math::normal_rng(0, 2, rng);
    sds1(i) = stan::math::uniform_rng(0.1, 2.0, rng);
  }

  int nclus2 = 10;
  Eigen::VectorXd weights2 =
      stan::math::dirichlet_rng(Eigen::VectorXd::Ones(nclus2), rng);
  Eigen::VectorXd means2(nclus2);
  Eigen::VectorXd sds2(nclus2);

  for (int i = 0; i < nclus2; i++) {
    means2(i) = stan::math::normal_rng(0, 2, rng);
    sds2(i) = stan::math::uniform_rng(0.1, 2.0, rng);
  }

  double dist = bayesmix::gaussian_mixture_dist(means1, sds1, weights1, means2,
                                                sds2, weights2);

  ASSERT_GE(dist, 0.0);
}

TEST(mix_dist, 2) {
  int nclus = 5;
  auto& rng = bayesmix::Rng::Instance().get();

  Eigen::VectorXd weights1 =
      stan::math::dirichlet_rng(Eigen::VectorXd::Ones(nclus), rng);
  Eigen::VectorXd means1(nclus);
  Eigen::VectorXd sds1(nclus);

  for (int i = 0; i < nclus; i++) {
    means1(i) = stan::math::normal_rng(0, 2, rng);
    sds1(i) = stan::math::uniform_rng(0.1, 2.0, rng);
  }

  double dist_to_self = bayesmix::gaussian_mixture_dist(
      means1, sds1, weights1, means1, sds1, weights1);

  ASSERT_DOUBLE_EQ(dist_to_self, 0.0);
}

TEST(student_t, squareform) {
  Eigen::MatrixXd A = Eigen::MatrixXd::Random(5, 5);
  Eigen::MatrixXd sigma =
      (A * A.transpose()) + 1.0 * Eigen::MatrixXd::Identity(5, 5);
  Eigen::VectorXd mean = Eigen::VectorXd::Zero(5);
  double df = 15;

  Eigen::MatrixXd sigma_inv = stan::math::inverse_spd(sigma);
  Eigen::MatrixXd sigma_inv_chol =
      Eigen::LLT<Eigen::MatrixXd>(sigma_inv).matrixU();

  Eigen::VectorXd x = Eigen::VectorXd::Ones(5);

  double sq1 = (x - mean).transpose() * sigma_inv * (x - mean);
  double sq2 = (sigma_inv_chol * (x - mean)).squaredNorm();

  ASSERT_DOUBLE_EQ(sq1, sq2);
}

TEST(student_t, optimized) {
  Eigen::MatrixXd A = Eigen::MatrixXd::Random(5, 5);
  Eigen::MatrixXd sigma =
      (A * A.transpose()) + 1.0 * Eigen::MatrixXd::Identity(5, 5);
  Eigen::VectorXd mean = Eigen::VectorXd::Zero(5);
  double df = 15;

  Eigen::VectorXd x = Eigen::VectorXd::Ones(5);

  double lpdf_stan = stan::math::multi_student_t_lpdf(x, df, mean, sigma);
  // std::cout << "lpdf_stan: " << lpdf_stan << std::endl;

  Eigen::MatrixXd sigma_inv = stan::math::inverse_spd(sigma);
  Eigen::MatrixXd sigma_inv_chol =
      Eigen::LLT<Eigen::MatrixXd>(sigma_inv).matrixU();
  Eigen::VectorXd diag = sigma_inv_chol.diagonal();
  double logdet = 2 * log(diag.array()).sum();

  double our_lpdf = bayesmix::multi_student_t_invscale_lpdf(
      x, df, mean, sigma_inv_chol, logdet);

  // std::cout << "our_lpdf: " << our_lpdf << std::endl;

  ASSERT_LE(std::abs(our_lpdf - lpdf_stan), 0.001);
}

TEST(student_t, marginal) {
  double var_scaling = 0.1;
  double deg_free = 10;
  int dim = 3;

  Eigen::MatrixXd A = Eigen::MatrixXd::Random(dim, dim);
  Eigen::MatrixXd scale_inv =
      (A * A.transpose()) + 1.0 * Eigen::MatrixXd::Identity(dim, dim);

  Eigen::MatrixXd sigma_n =
      scale_inv * (var_scaling + 1) / (var_scaling * (deg_free - dim + 1));
  double nu_n = deg_free - dim + 1;

  Eigen::VectorXd datum = Eigen::VectorXd::Ones(dim);
  Eigen::VectorXd mean = Eigen::VectorXd::Zero(dim);

  Eigen::MatrixXd scale = stan::math::inverse_spd(scale_inv);
  Eigen::MatrixXd scale_chol = Eigen::LLT<Eigen::MatrixXd>(scale).matrixU();

  double coeff = (var_scaling + 1) / (var_scaling * (deg_free - dim + 1));
  Eigen::MatrixXd scale_chol_n = scale_chol / std::sqrt(coeff);
  Eigen::VectorXd diag = scale_chol_n.diagonal();
  double logdet = 2 * log(diag.array()).sum();

  double old_qf = (datum - mean).transpose() *
                  stan::math::inverse_spd(sigma_n) * (datum - mean);

  double new_qf = (scale_chol_n * (datum - mean)).squaredNorm();

  ASSERT_DOUBLE_EQ(old_qf, new_qf);

  double old_lpdf =
      stan::math::multi_student_t_lpdf(datum, nu_n, mean, sigma_n);

  double new_lpdf = bayesmix::multi_student_t_invscale_lpdf(
      datum, nu_n, mean, scale_chol_n, logdet);

  ASSERT_LE(std::abs(old_lpdf - new_lpdf), 0.001);
}

TEST(mult_normal, lpdf_grid) {
  int dim = 3;

  Eigen::MatrixXd data = Eigen::MatrixXd::Random(20, dim);
  Eigen::VectorXd mean = Eigen::ArrayXd::LinSpaced(dim, 0.0, 10.0);
  Eigen::MatrixXd tmp = Eigen::MatrixXd::Random(dim + 1, dim);
  Eigen::MatrixXd prec =
      tmp.transpose() * tmp + Eigen::MatrixXd::Identity(dim, dim);
  Eigen::MatrixXd prec_chol = Eigen::LLT<Eigen::MatrixXd>(prec).matrixU();
  Eigen::VectorXd diag = prec_chol.diagonal();
  double prec_logdet = 2 * log(diag.array()).sum();

  Eigen::VectorXd lpdfs = bayesmix::multi_normal_prec_lpdf_grid(
      data, mean, prec_chol, prec_logdet);

  for (int i = 0; i < 20; i++) {
    double curr = bayesmix::multi_normal_prec_lpdf(data.row(i), mean,
                                                   prec_chol, prec_logdet);
    ASSERT_DOUBLE_EQ(curr, lpdfs(i));
  }
}

TEST(mult_t, lpdf_grid) {
  int dim = 3;

  Eigen::MatrixXd data = Eigen::MatrixXd::Random(20, dim);
  Eigen::VectorXd mean = Eigen::ArrayXd::LinSpaced(dim, 0.0, 10.0);
  Eigen::MatrixXd tmp = Eigen::MatrixXd::Random(dim + 1, dim);
  Eigen::MatrixXd invscale =
      tmp.transpose() * tmp + Eigen::MatrixXd::Identity(dim, dim);
  Eigen::MatrixXd invscale_chol =
      Eigen::LLT<Eigen::MatrixXd>(invscale).matrixU();
  Eigen::VectorXd diag = invscale_chol.diagonal();
  double invscale_logdet = 2 * log(diag.array()).sum();
  double df = 10;

  Eigen::VectorXd lpdfs = bayesmix::multi_student_t_invscale_lpdf_grid(
      data, df, mean, invscale_chol, invscale_logdet);

  for (int i = 0; i < 20; i++) {
    double curr = bayesmix::multi_student_t_invscale_lpdf(
        data.row(i), df, mean, invscale_chol, invscale_logdet);
    ASSERT_DOUBLE_EQ(curr, lpdfs(i));
  }
}

TEST(lpdf_woodbury, 1) {
  int dim = 1000;
  int q = 10;
  auto& rng = bayesmix::Rng::Instance().get();
  Eigen::VectorXd mean(dim);
  Eigen::VectorXd datum(dim);
  Eigen::VectorXd sigma_diag(dim);
  Eigen::MatrixXd lambda(dim, q);

  for (size_t j = 0; j < dim; j++) {
    mean[j] = stan::math::normal_rng(0, 1, rng);

    sigma_diag[j] = stan::math::inv_gamma_rng(2.5, 1, rng);

    for (size_t i = 0; i < q; i++) {
      lambda(j, i) = stan::math::normal_rng(0, 1, rng);
    }
  }

  Eigen::MatrixXd cov =
      lambda * lambda.transpose() + Eigen::MatrixXd(sigma_diag.asDiagonal());

  datum = stan::math::multi_normal_rng(mean, cov, rng);

  double stan_lpdf = stan::math::multi_normal_lpdf(datum, mean, cov);
  double our_lpdf =
      bayesmix::multi_normal_lpdf_woodbury(datum, mean, sigma_diag, lambda);

  ASSERT_LE(std::abs(stan_lpdf - our_lpdf), 1e-10);
}
