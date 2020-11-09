#include <gtest/gtest.h>

#include <Eigen/Dense>
#include <stan/math/prim/prob.hpp>

#include "../proto/cpp/ls_state.pb.h"
#include "../proto/cpp/marginal_state.pb.h"
#include "../src/hierarchies/nnig_hierarchy.hpp"
#include "../src/hierarchies/nnw_hierarchy.hpp"

TEST(nnighierarchy, draw) {
  auto hier = std::make_shared<NNIGHierarchy>();
  hier->set_mu0(5.0);
  hier->set_lambda(0.1);
  hier->set_alpha0(2.0);
  hier->set_beta0(2.0);
  hier->check_and_initialize();

  auto hier2 = hier->clone();
  hier2->draw();

  ASSERT_TRUE(hier != hier2);
}

TEST(nnighierarchy, sample_given_data) {
  auto hier = std::make_shared<NNIGHierarchy>();
  hier->set_mu0(5.0);
  hier->set_lambda(0.1);
  hier->set_alpha0(2.0);
  hier->set_beta0(2.0);
  hier->check_and_initialize();

  Eigen::VectorXd datum(1);
  datum << 4.5;

  auto hier2 = hier->clone();
  hier2->sample_given_data(datum);

  ASSERT_TRUE(hier != hier2);
}

TEST(nnwhierarchy, draw) {
  auto hier = std::make_shared<NNWHierarchy>();
  Eigen::VectorXd mu0(2);
  mu0 << 5.5, 5.5;
  hier->set_mu0(mu0);
  hier->set_lambda(0.2);
  double nu = 5.0;
  hier->set_nu(nu);
  Eigen::Matrix2d tau0 = (1 / nu) * Eigen::Matrix2d::Identity();
  hier->set_tau0(tau0);
  hier->check_and_initialize();

  auto hier2 = hier->clone();
  hier2->draw();

  ASSERT_TRUE(hier != hier2);
}

TEST(nnwhierarchy, sample_given_data) {
  auto hier = std::make_shared<NNWHierarchy>();
  Eigen::VectorXd mu0(2);
  mu0 << 5.5, 5.5;
  hier->set_mu0(mu0);
  hier->set_lambda(0.2);
  double nu = 5.0;
  hier->set_nu(nu);
  Eigen::Matrix2d tau0 = (1 / nu) * Eigen::Matrix2d::Identity();
  hier->set_tau0(tau0);
  hier->check_and_initialize();

  Eigen::Matrix<double, 1, 2> datum;
  datum << 4.5, 4.5;

  auto hier2 = hier->clone();
  hier2->sample_given_data(datum);

  ASSERT_TRUE(hier != hier2);
}

TEST(lpdf, nnig) {
  NNIGHierarchy hier;

  Eigen::VectorXd datum(1);
  datum << 4.5;
  double mu0 = 5.0;
  double lambda = 0.1;
  double alpha0 = 2.0;
  double beta0 = 2.0;
  hier.set_mu0(mu0);
  hier.set_lambda(lambda);
  hier.set_alpha0(alpha0);
  hier.set_beta0(beta0);
  hier.check_and_initialize();
  double mean = mu0;
  double std = beta0 / (alpha0 - 1);

  // Compute posterior parameters
  double mu_post = (lambda * mu0 + datum(0)) / (lambda + 1);
  double alpha_post = alpha0 + 0.5;
  double beta_post = beta0;
  double lambda_post = lambda + 1;

  // logmarg(x) = logprior(phi) + loglik(x|phi) - logpost(phi|x)
  double prior = stan::math::inv_gamma_lpdf(std, alpha0, beta0) +
                 stan::math::normal_lpdf(mean, mu0, std / sqrt(lambda));
  double lik = hier.lpdf(datum)(0);
  double post =
      stan::math::inv_gamma_lpdf(std, alpha_post, beta_post) +
      stan::math::normal_lpdf(mean, mu_post, std / sqrt(lambda_post));
  double marg = prior + lik - post;
  double marg2 = hier.marg_lpdf(datum)(0);

  ASSERT_EQ(marg, marg2);
}

// TEST(lpdf, nnw) {
//  NNWHierarchy hier;
//  Eigen::VectorXd mu0(2); mu0 << 5.5, 5.5;
//  hier.set_mu0(mu0);
//  hier.set_lambda(0.2);
//  double nu = 5.0;
//  hier.set_nu(nu);
//  Eigen::Matrix2d tau0 = (1 / nu) * Eigen::Matrix2d::Identity();
//  hier.set_tau0(tau0);
//  hier.check_and_initialize();
//}
