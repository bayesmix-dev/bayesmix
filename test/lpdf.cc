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

TEST(lpdf, nnw) {  // TODO
  using namespace stan::math;
  NNWHierarchy hier;

  NNW::Hyperparams prior_params;
  prior_params.mean = Eigen::VectorXd::Ones(2) * 5.5;
  prior_params.var_scaling = 0.2;
  prior_params.deg_free = 10;
  prior_params.scale = Eigen::MatrixXd::Identity(2, 2) * 0.5;
  prior_params.scale_inv = inverse_spd(prior_params.scale);
  hier.set_hypers(prior_params);
  std::cout << "set hypers ok" << std::endl;

  NNW::State curr_state;
  curr_state.mean = Eigen::VectorXd::Ones(2) * 5.0;
  curr_state.prec = Eigen::MatrixXd::Identity(2, 2) * 1.0;
  curr_state.prec_chol =
      Eigen::LLT<Eigen::MatrixXd>(curr_state.prec).matrixL().transpose();
  Eigen::VectorXd diag = curr_state.prec_chol.diagonal();
  curr_state.prec_logdet = 2 * log(diag.array()).sum();
  hier.set_state(curr_state);

  Eigen::VectorXd datum = Eigen::VectorXd::Ones(2) * 15.5;
  hier.clear_data();
  hier.add_datum(0, datum);
  NNW::Hyperparams post_params = hier.get_posterior_parameters();

  double prior_lpdf =
      multi_normal_prec_lpdf(curr_state.mean, prior_params.mean,
                             curr_state.prec * prior_params.var_scaling) +
      wishart_lpdf(curr_state.prec, prior_params.deg_free, prior_params.scale);

  double post_lpdf =
      multi_normal_prec_lpdf(curr_state.mean, post_params.mean,
                             curr_state.prec * post_params.var_scaling) +
      wishart_lpdf(curr_state.prec, post_params.deg_free, post_params.scale);

  double like_lpdf = hier.like_lpdf(datum);

  double marg_lpdf = hier.marg_lpdf(prior_params, datum);

  double marg_murphy;
  double num = lmgamma(2, 0.5 * post_params.deg_free) +
               0.5 * prior_params.deg_free *
                   std::log(prior_params.scale_inv.determinant());
  double den = 2 * ((-1) * NEG_LOG_SQRT_TWO_PI) *
                   lmgamma(2, 0.5 * prior_params.deg_free) +
               0.5 * prior_params.deg_free *
                   std::log(post_params.scale_inv.determinant());

  marg_murphy =
      num - den +
      1.0 * std::log(prior_params.var_scaling / post_params.var_scaling);

  std::cout << "marg_murphy: " << marg_murphy << std::endl;
  std::cout << "marg_lpdf: " << marg_lpdf << std::endl;
  std::cout << "sums: " << prior_lpdf + like_lpdf - post_lpdf << std::endl;

  ASSERT_DOUBLE_EQ(marg_lpdf, prior_lpdf + like_lpdf - post_lpdf);
  ASSERT_DOUBLE_EQ(marg_murphy, prior_lpdf + like_lpdf - post_lpdf);
  ASSERT_DOUBLE_EQ(marg_lpdf, prior_lpdf + like_lpdf - post_lpdf);
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
