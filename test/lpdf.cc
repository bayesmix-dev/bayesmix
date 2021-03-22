#include <gtest/gtest.h>

#include <Eigen/Dense>
#include <stan/math/prim/fun.hpp>  // lgamma, lmgamma
#include <stan/math/prim/prob.hpp>

#include "algorithm_state.pb.h"
#include "src/hierarchies/lin_reg_uni_hierarchy.h"
#include "src/hierarchies/nnig_hierarchy.h"
#include "src/hierarchies/nnw_hierarchy.h"
#include "src/utils/proto_utils.h"

TEST(lpdf, nnig) {
  NNIGHierarchy hier;
  bayesmix::NNIGPrior hier_prior;
  double mu0 = 5.0;
  double lambda0 = 0.1;
  double alpha0 = 2.0;
  double beta0 = 2.0;
  hier_prior.mutable_fixed_values()->set_mean(mu0);
  hier_prior.mutable_fixed_values()->set_var_scaling(lambda0);
  hier_prior.mutable_fixed_values()->set_shape(alpha0);
  hier_prior.mutable_fixed_values()->set_scale(beta0);
  hier.get_mutable_prior()->CopyFrom(hier_prior);
  hier.initialize();

  double mean = mu0;
  double var = beta0 / (alpha0 + 1);

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
  double prior1 = stan::math::inv_gamma_lpdf(var, alpha0, beta0);
  double prior2 = stan::math::normal_lpdf(mean, mu0, sqrt(var / lambda0));
  double prior = prior1 + prior2;
  double like = hier.like_lpdf(datum);
  double post1 = stan::math::inv_gamma_lpdf(var, alpha_n, beta_n);
  double post2 = stan::math::normal_lpdf(mean, mu_n, sqrt(var / lambda_n));
  double post = post1 + post2;

  // Bayes: logmarg(x) = logprior(phi) + loglik(x|phi) - logpost(phi|x)
  double sum = prior + like - post;
  double marg = hier.prior_pred_lpdf(datum);

  ASSERT_DOUBLE_EQ(sum, marg);
}


TEST(lpdf, lin_reg_uni) {
  // Create hierarchy objects
  LinRegUniHierarchy hier;
  bayesmix::LinRegUniPrior prior;
  int dim = 3;

  // Generate data
  Eigen::VectorXd datum(1);
  datum << 1.5;
  Eigen::VectorXd cov = Eigen::VectorXd::Random(dim);

  // Create parameters, both Eigen and proto
  Eigen::VectorXd mu0(dim);
  for (int i = 0; i < dim; i++) {
    mu0(i) = 2 * i;
  }
  bayesmix::Vector mu0_proto;
  bayesmix::to_proto(mu0, &mu0_proto);
  auto Lambda0 = Eigen::MatrixXd::Identity(dim, dim);
  bayesmix::Matrix Lambda0_proto;
  bayesmix::to_proto(Lambda0, &Lambda0_proto);
  double alpha0 = 2.0;
  double beta0 = 2.0;
  // Set parameters
  *prior.mutable_fixed_values()->mutable_mean() = mu0_proto;
  *prior.mutable_fixed_values()->mutable_var_scaling() = Lambda0_proto;
  prior.mutable_fixed_values()->set_shape(alpha0);
  prior.mutable_fixed_values()->set_scale(beta0);
  // Initialize hierarchy
  hier.get_mutable_prior()->CopyFrom(prior);
  hier.initialize();

  // Compute prior parameters
  Eigen::VectorXd mean = mu0;
  double var = beta0 / (alpha0 + 1);

  // Compute posterior parameters
  Eigen::MatrixXd Lambda_n = Lambda0 + cov * cov.transpose();
  Eigen::VectorXd mu_n =
      stan::math::inverse_spd(Lambda_n) * (datum(0) * cov + Lambda0 * mu0);
  double alpha_n = alpha0 + 0.5;
  double beta_n =
      beta0 + 0.5 * (datum(0) * datum(0) + mu0.transpose() * Lambda0 * mu0 -
                     mu_n.transpose() * Lambda_n * mu_n);
  // Compute pieces
  double prior1 = stan::math::inv_gamma_lpdf(var, alpha0, beta0);
  double prior2 = stan::math::multi_normal_prec_lpdf(mean, mu0, Lambda0 / var);
  double pr = prior1 + prior2;
  double like = hier.like_lpdf(datum, cov);
  double post1 = stan::math::inv_gamma_lpdf(var, alpha_n, beta_n);
  double post2 =
      stan::math::multi_normal_prec_lpdf(mean, mu_n, Lambda_n / var);
  double post = post1 + post2;

  // Bayes: logmarg(x) = logprior(phi) + loglik(x|phi) - logpost(phi|x)
  double sum = pr + like - post;
  double marg = hier.prior_pred_lpdf(datum, cov);

  ASSERT_FLOAT_EQ(sum, marg);
}

TEST(lpdf, nnw) {
  using namespace stan::math;
  // Initialize prior and hierarchy
  NNWHierarchy hier;
  bayesmix::NNWPrior prior;
  Eigen::VectorXd mu0(2);
  mu0 << 5.5, 5.5;
  bayesmix::Vector mu0_proto;
  bayesmix::to_proto(mu0, &mu0_proto);
  double lambda0 = 0.2;
  double nu0 = 5.0;
  Eigen::Matrix2d tau0 = Eigen::Matrix2d::Identity();
  bayesmix::Matrix tau0_proto;
  bayesmix::to_proto(tau0, &tau0_proto);
  *prior.mutable_fixed_values()->mutable_mean() = mu0_proto;
  prior.mutable_fixed_values()->set_var_scaling(lambda0);
  prior.mutable_fixed_values()->set_deg_free(nu0);
  *prior.mutable_fixed_values()->mutable_scale() = tau0_proto;
  hier.get_mutable_prior()->CopyFrom(prior);
  hier.initialize();
  // std::cout << hier.get_posterior_hypers().mean << std::endl;
  // Reproduce hierarchy state
  Eigen::VectorXd mu = mu0;
  Eigen::MatrixXd tau = lambda0 * Eigen::Matrix2d::Identity();
  // Datum
  Eigen::RowVectorXd datum(2);
  datum << 4.5, 4.5;
  hier.add_datum(1, datum, true);
  // std::cout << hier.get_posterior_hypers().mean << std::endl;
  // Compute prior
  double logprior1 = stan::math::wishart_lpdf(tau, nu0, tau0);
  double logprior2 = stan::math::multi_normal_prec_lpdf(mu, mu0,
  	                                                    lambda0 * tau0);
  double logprior = logprior1 + logprior2;
  // Compute likelihood
  double loglike = hier.like_lpdf(datum);
  // Compute posterior
  auto post = hier.get_posterior_hypers();
  double logpost1 = stan::math::wishart_lpdf(tau, post.deg_free,
  	  post.scale_inv);
  double logpost2 = stan::math::multi_normal_prec_lpdf(mu, post.mean,
      post.var_scaling * post.scale_inv);
  double logpost = logpost1 + logpost2;
  // Bayes: logmarg(x) = logprior(phi) + loglik(x|phi) - logpost(phi|x)
  double logsum = logprior + loglike - logpost;
  double logmarg = hier.prior_pred_lpdf(datum);
  // Print values
  std::cout << "prior1=" << logprior1 << std::endl;
  std::cout << "prior2=" << logprior2 << std::endl;
  std::cout << "prior =" << logprior << std::endl;
  std::cout << "like  =" << loglike << std::endl;
  std::cout << "post1 =" << logpost1 << std::endl;
  std::cout << "post2 =" << logpost2 << std::endl;
  std::cout << "post  =" << logpost << std::endl;
  std::cout << "sum   =" << logsum << std::endl;
  std::cout << "marg  =" << logmarg << std::endl;
  ASSERT_DOUBLE_EQ(logsum, logmarg);
}
