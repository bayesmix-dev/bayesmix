#ifndef BAYESMIX_HIERARCHIES_LIKELIHOODS_MULTI_NORM_LIKELIHOOD_H_
#define BAYESMIX_HIERARCHIES_LIKELIHOODS_MULTI_NORM_LIKELIHOOD_H_

#include <google/protobuf/stubs/casts.h>

#include <memory>
#include <stan/math/prim.hpp>
#include <stan/math/rev.hpp>
#include <vector>

#include "algorithm_state.pb.h"
#include "base_likelihood.h"
#include "states/includes.h"

//! A multivariate normal likelihood

class MultiNormLikelihood
    : public BaseLikelihood<MultiNormLikelihood, State::MultiLS> {
 public:
  MultiNormLikelihood() = default;
  ~MultiNormLikelihood() = default;
  bool is_multivariate() const override { return true; };
  bool is_dependent() const override { return false; };
  void clear_summary_statistics() override;

  void set_dim(unsigned int dim_) {
    dim = dim_;
    clear_summary_statistics();
  };
  unsigned int get_dim() const { return dim; };
  Eigen::VectorXd get_data_sum() const { return data_sum; };
  Eigen::MatrixXd get_data_sum_squares() const { return data_sum_squares; };

 protected:
  double compute_lpdf(const Eigen::RowVectorXd &datum) const override;
  void update_sum_stats(const Eigen::RowVectorXd &datum, bool add) override;

  unsigned int dim;
  Eigen::VectorXd data_sum;
  Eigen::MatrixXd data_sum_squares;
};

#endif  // BAYESMIX_HIERARCHIES_LIKELIHOODS_MULTI_NORM_LIKELIHOOD_H_
