#include <gtest/gtest.h>

#include <Eigen/Dense>
#include <stan/math/prim/fun.hpp>  // lgamma, lmgamma
#include <stan/math/prim/prob.hpp>

#include "../proto/cpp/marginal_state.pb.h"
#include "../src/hierarchies/nnig_hierarchy.hpp"
#include "../src/hierarchies/nnw_hierarchy.hpp"

TEST(lpdf, nnig) {
  NNIGHierarchy hier;
  double mu0 = 5.0;
  double lambda0 = 0.1;
  double alpha0 = 2.0;
  double beta0 = 2.0;
  hier.set_mu0(mu0);
  hier.set_lambda0(lambda0);
  hier.set_alpha0(alpha0);
  hier.set_beta0(beta0);
  hier.check_and_initialize();

  double mean = mu0;
  double sd = sqrt(beta0 / (alpha0 - 1));

  Eigen::VectorXd datum(1);
  datum << 4.5;

  // Compute posterior parameters
  double mu_n = (lambda0 * mu0 + datum(0)) / (lambda0 + 1);
  double alpha_n = alpha0 + 0.5;
  double lambda_n = lambda0 + 1;
  double beta_n = beta0 + (0.5 * lambda0 / (lambda0 + 1)) * (datum(0) - mu0) *
                              (datum(0) - mu0);
  // equiv.ly: beta0 + 0.5*(mu0^2*lambda0 + datum^2 - mu_n^2*lambda_n);

  // Compute pieces
  double prior1 = stan::math::inv_gamma_lpdf(sd * sd, alpha0, beta0);
  double prior2 = stan::math::normal_lpdf(mean, mu0, sd / sqrt(lambda0));
  double prior = prior1 + prior2;
  double like = hier.lpdf(datum);
  double post1 = stan::math::inv_gamma_lpdf(sd * sd, alpha_n, beta_n);
  double post2 = stan::math::normal_lpdf(mean, mu_n, sd / sqrt(lambda_n));
  double post = post1 + post2;

  // Bayes: logmarg(x) = logprior(phi) + loglik(x|phi) - logpost(phi|x)
  double sum = prior + like - post;
  double marg = hier.marg_lpdf(datum);

  ASSERT_FLOAT_EQ(sum, marg);
}

// TEST(lpdf, nnw) {  // TODO
//   using namespace stan::math;
//   NNWHierarchy hier;
//   Eigen::VectorXd mu0(2);
//   mu0 << 5.5, 5.5;
//   hier.set_mu0(mu0);
//   double lambda0 = 0.2;
//   hier.set_lambda0(lambda0);
//   double nu0 = 5.0;
//   hier.set_nu0(nu0);
//   Eigen::MatrixXd tau0 = Eigen::Matrix2d::Identity() / nu0;
//   hier.set_tau0(tau0);
//   hier.check_and_initialize();
//   Eigen::VectorXd mu = mu0;
//   Eigen::MatrixXd tau = lambda0 * Eigen::Matrix2d::Identity();
//
//   Eigen::RowVectorXd datum(2);
//   datum << 4.5, 4.5;
//
//   // Compute prior parameters
//   Eigen::MatrixXd tau_pr = lambda0 * tau0;
//
//   // Compute posterior parameters
//   double lambda_n = lambda0 + 1;
//   double nu_n = nu0 + 0.5;
//   Eigen::VectorXd mu_n =
//       (lambda0 * mu0 + datum.transpose()) / (lambda0 + 1);
//   Eigen::MatrixXd tau_temp =
//       stan::math::inverse_spd(tau0) + (0.5 * lambda0 / (lambda0 + 1)) *
//                                           (datum.transpose() - mu0) *
//                                           (datum - mu0.transpose());
//   Eigen::MatrixXd tau_n = stan::math::inverse_spd(tau_temp);
//   Eigen::MatrixXd tau_post = lambda_n * tau_n;
//
//   // Compute pieces
//   double prior1 = stan::math::wishart_lpdf(tau, nu0, tau0);
//   double prior2 = stan::math::multi_normal_prec_lpdf(mu, mu0, tau_pr);
//   double prior = prior1 + prior2;
//   double like = hier.lpdf(datum);
//   double post1 = stan::math::wishart_lpdf(tau, nu_n, tau_post);
//   double post2 = stan::math::multi_normal_prec_lpdf(mu, mu0, tau_post);
//   double post = post1 + post2;
//   // Bayes: logmarg(x) = logprior(phi) + loglik(x|phi) - logpost(phi|x)
//   double sum = prior + like - post;
//   double marg = hier.marg_lpdf(datum);
//
//   // Compute logdet's
//   Eigen::MatrixXd tauchol0 =
//       Eigen::LLT<Eigen::MatrixXd>(tau0).matrixL().transpose();
//   double logdet0 = 2 * log(tauchol0.diagonal().array()).sum();
//   Eigen::MatrixXd tauchol_n =
//       Eigen::LLT<Eigen::MatrixXd>(tau_n).matrixL().transpose();
//   double logdet_n = 2 * log(tauchol_n.diagonal().array()).sum();
//
//   // lmgamma(dim, x)
//   int dim = 2;
//   double marg_murphy = lmgamma(dim, 0.5 * nu_n) + 0.5 * nu_n * logdet_n +
//                        0.5 * dim * log(lambda0) + dim * NEG_LOG_SQRT_TWO_PI
//                        - lmgamma(dim, 0.5 * nu0) - 0.5 * nu0 * logdet0 - 0.5
//                        * dim * log(lambda_n);
//
//   // std::cout << "prior1=" << prior1 << std::endl;
//   // std::cout << "prior2=" << prior2 << std::endl;
//   // std::cout << "prior =" << prior << std::endl;
//   // std::cout << "like  =" << like << std::endl;
//   // std::cout << "post1 =" << post1 << std::endl;
//   // std::cout << "post2 =" << post2 << std::endl;
//   // std::cout << "post  =" << post << std::endl;
//   std::cout << "sum   =" << sum << std::endl;
//   std::cout << "marg  =" << marg << std::endl;
//   std::cout << "murphy=" << marg_murphy << std::endl;
//   ASSERT_FLOAT_EQ(marg, marg_murphy);
// }
