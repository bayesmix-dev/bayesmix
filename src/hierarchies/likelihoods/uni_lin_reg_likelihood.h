#ifndef BAYESMIX_HIERARCHIES_LIKELIHOODS_UNI_LIN_REG_LIKELIHOOD_H_
#define BAYESMIX_HIERARCHIES_LIKELIHOODS_UNI_LIN_REG_LIKELIHOOD_H_

#include <google/protobuf/stubs/casts.h>

#include <memory>
#include <stan/math/rev.hpp>
#include <vector>

#include "algorithm_state.pb.h"
#include "base_likelihood.h"
#include "states/includes.h"

/**
 * A scalar linear regression model, using the `State::UniLinRegLS` state.
 * Represents the model:
 *
 * \f[
 *    y_i \mid \bm{x}_i, \bm{\beta}, \sigma^2
 * \stackrel{\small\mathrm{ind}}{\sim} N(\bm{x}_i^T\bm{\beta},\sigma^2), \f]
 *
 * where \f$ (\bm{\beta}, \sigma^2) \f$ are stored in a `State::UniLinRegLS`
 * state. The sufficient statistics stored are the sum of \f$ y_i^2 \f$, the
 * sum of \f$ \bm{x}_i^T \bm{x}_i \f$ and the sum of \f$ y_i \bm{x}_i^T \f$.
 */

class UniLinRegLikelihood
    : public BaseLikelihood<UniLinRegLikelihood, State::UniLinRegLS> {
 public:
  UniLinRegLikelihood() = default;
  ~UniLinRegLikelihood() = default;
  bool is_multivariate() const override { return false; };
  bool is_dependent() const override { return true; };
  void clear_summary_statistics() override;

  // Getters and Setters
  unsigned int get_dim() const { return dim; };
  void set_dim(unsigned int dim_) {
    dim = dim_;
    clear_summary_statistics();
  };
  double get_data_sum_squares() const { return data_sum_squares; };
  Eigen::MatrixXd get_covar_sum_squares() const { return covar_sum_squares; };
  Eigen::VectorXd get_mixed_prod() const { return mixed_prod; };

 protected:
  double compute_lpdf(const Eigen::RowVectorXd &datum,
                      const Eigen::RowVectorXd &covariate) const override;
  void update_sum_stats(const Eigen::RowVectorXd &datum,
                        const Eigen::RowVectorXd &covariate,
                        bool add) override;

  // Dimension of the coefficients vector
  unsigned int dim;
  // Represents pieces of y^t y
  double data_sum_squares;
  // Represents pieces of X^T X
  Eigen::MatrixXd covar_sum_squares;
  // Represents pieces of X^t y
  Eigen::VectorXd mixed_prod;
};

#endif  // BAYESMIX_HIERARCHIES_LIKELIHOODS_UNI_LIN_REG_LIKELIHOOD_H_
