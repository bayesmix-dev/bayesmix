#include <gtest/gtest.h>

#include <Eigen/Dense>
#include <stan/math/prim/prob.hpp>

#include "../src/hierarchies/nnig_hierarchy.hpp"
#include "../src/hierarchies/nnw_hierarchy.hpp"
#include "../proto/cpp/marginal_state.pb.h"

TEST(lpdf, nnig) {
  NNIGHierarchy hier;
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
  double std = sqrt(beta0 / (alpha0 - 1));

  Eigen::VectorXd datum(1);
  datum << 4.5;

  // Compute posterior parameters
  double mu_post = (lambda * mu0 + datum(0)) / (lambda + 1);
  double alpha_post = alpha0 + 0.5;
  double beta_post = beta0 + (0.5 * lambda / (lambda + 1)) * (datum(0) - mu0) *
                                 (datum(0) - mu0);
  double lambda_post = lambda + 1;

  // Compute pieces
  double prior1 = stan::math::inv_gamma_lpdf(std, alpha0, 1/beta0);
  double prior2 = stan::math::normal_lpdf(mean, mu0, std / sqrt(lambda));
  double prior = prior1 + prior2;
  double like = hier.lpdf(datum)(0);
  double post1 = stan::math::inv_gamma_lpdf(std, alpha_post, 1/beta_post);
  double post2 =
    stan::math::normal_lpdf(mean, mu_post, std / sqrt(lambda_post));
  double post = post1 + post2;
  // Bayes: logmarg(x) = logprior(phi) + loglik(x|phi) - logpost(phi|x)
  double sum = prior + like - post;
  double marg = hier.marg_lpdf(datum)(0);

  std::cout << "prior1=" << prior1 << std::endl;
  std::cout << "prior2=" << prior2 << std::endl;
  std::cout << "prior =" << prior  << std::endl;
  std::cout << "like  =" << like   << std::endl;
  std::cout << "post1 =" << post1  << std::endl;
  std::cout << "post2 =" << post2  << std::endl;
  std::cout << "post  =" << post   << std::endl;

  ASSERT_EQ(sum, marg);
}



TEST(lpdf, nnw) {
  NNWHierarchy hier;
  Eigen::VectorXd mu0(2);
  mu0 << 5.5, 5.5;
  hier.set_mu0(mu0);
  double lambda = 0.2;
  hier.set_lambda(lambda);
  double nu = 5.0;
  hier.set_nu(nu);
  Eigen::MatrixXd tau0 = (1 / nu) * Eigen::Matrix2d::Identity();
  hier.set_tau0(tau0);
  hier.check_and_initialize();
  Eigen::VectorXd mu = mu0;
  Eigen::MatrixXd tau = lambda * Eigen::Matrix2d::Identity();

  Eigen::RowVectorXd datum(2);
  datum << 4.5, 4.5;

  // Compute prior parameters
  Eigen::MatrixXd tau_pr = lambda * tau0;

  // Compute posterior parameters
  double lambda_post = lambda + 1;
  double nu_post = nu + 0.5;
  Eigen::VectorXd mu_post =
      (lambda * mu0 + datum.transpose()) * (1 / (lambda + 1));
  Eigen::MatrixXd tau_temp =
      stan::math::inverse_spd(tau0) + (0.5 * lambda / (lambda + 1)) *
                                          (datum.transpose() - mu0) *
                                          (datum - mu0.transpose());
  Eigen::MatrixXd tau_post = lambda_post * stan::math::inverse_spd(tau_temp);

  // Compute pieces
  double prior1 = stan::math::wishart_lpdf(tau, nu, tau0);
  double prior2 = stan::math::multi_normal_prec_lpdf(mu, mu0, tau_pr);
  double prior = prior1 + prior2;
  double like = hier.lpdf(datum)(0);
  double post1 = stan::math::wishart_lpdf(tau, nu_post, tau_post);
  double post2 = stan::math::multi_normal_prec_lpdf(mu, mu0, tau_post);
  double post = post1 + post2;
  // Bayes: logmarg(x) = logprior(phi) + loglik(x|phi) - logpost(phi|x)
  double sum = prior + like - post;
  double marg = hier.marg_lpdf(datum)(0);

  std::cout << "prior1=" << prior1 << std::endl;
  std::cout << "prior2=" << prior2 << std::endl;
  std::cout << "prior =" << prior  << std::endl;
  std::cout << "like  =" << like   << std::endl;
  std::cout << "post1 =" << post1  << std::endl;
  std::cout << "post2 =" << post2  << std::endl;
  std::cout << "post  =" << post   << std::endl;

  ASSERT_EQ(sum, marg);
}
