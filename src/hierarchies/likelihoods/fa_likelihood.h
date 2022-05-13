#ifndef BAYESMIX_HIERARCHIES_LIKELIHOODS_FA_LIKELIHOOD_H_
#define BAYESMIX_HIERARCHIES_LIKELIHOODS_FA_LIKELIHOOD_H_

#include <google/protobuf/stubs/casts.h>

#include <memory>
#include <stan/math/prim.hpp>
#include <stan/math/rev.hpp>
#include <vector>

#include "algorithm_state.pb.h"
#include "base_likelihood.h"
#include "states/includes.h"

//! A gaussian factor analytic likelihood, that is
//!     Y ~ N_p(mu, Sigma + Lambda * Lambda^T)
//! Where Lambda is a `p x d` matrix, usually d << p and `Sigma` is a diagonal
//! matrix.
//!
//! Parameters are stored in a `State::FA` state.
//! We store as summary statistics the sum of the y_i's, but it is
//! not sufficient for all the updates involved. Therefore, all the
//! observations allocated to a cluster are processed when computing the
//! cluster lpdf.

class FALikelihood : public BaseLikelihood<FALikelihood, State::FA> {
 public:
  FALikelihood() = default;
  ~FALikelihood() = default;
  bool is_multivariate() const override { return true; };
  bool is_dependent() const override { return false; };
  void clear_summary_statistics() override;
  void set_dim(unsigned int dim_) {
    dim = dim_;
    clear_summary_statistics();
  };
  unsigned int get_dim() const { return dim; };
  Eigen::VectorXd get_data_sum() const { return data_sum; };

 protected:
  double compute_lpdf(const Eigen::RowVectorXd& datum) const override;
  void update_sum_stats(const Eigen::RowVectorXd& datum, bool add) override;

  unsigned int dim;
  Eigen::VectorXd data_sum;
};

#endif  // BAYESMIX_HIERARCHIES_LIKELIHOODS_FA_LIKELIHOOD_H_
