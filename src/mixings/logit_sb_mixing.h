#ifndef BAYESMIX_MIXINGS_LOGIT_SB_MIXING_H_
#define BAYESMIX_MIXINGS_LOGIT_SB_MIXING_H_

#include <google/protobuf/message.h>

#include <memory>
#include <stan/math/rev.hpp>
#include <vector>

#include "base_mixing.h"
#include "mixing_id.pb.h"
#include "mixing_prior.pb.h"
#include "src/hierarchies/abstract_hierarchy.h"

namespace LogitSB {
struct State {
  Eigen::MatrixXd regression_coeffs, precision;
};
};  // namespace LogitSB

/**
 * Class that represents the logit stick-breaking process indroduced in Rigon
 * and Durante (2020).
 * That is, a prior for weights \f$ (w_1,\dots,w_H) \f$, depending on
 * covariates \f$ x \f$ in \f$ \mathbb{R}^p \f$, in the H-1 dimensional unit
 * simplex, defined as follows:
 *
 * \f[
 *    w_1 &= v_1 \\
 *    w_j &= v_j (1 - v_1) ... (1 - v_{j-1}), \quad \text{for } j=1, ... H-1 \\
 *    w_H &= 1 - (w_1 + w_2 + ... + w_{H-1}) \\
 *    v_j(x) &= 1 / exp(- <\alpha_j, x> ), for j = 1, ..., H-1
 * \f]
 *
 * The main difference with the mentioned paper is that the authors propose a
 * Gibbs sampler in which the full conditionals are available in close form
 * thanks to a Polya-Gamma augmentation. Here instead, a Metropolis-adjusted
 * Langevin algorithm (MALA) step is used. The step-size of the MALA step must
 * be passed in the LogSBPrior Protobuf message.
 * For more information about the class, please refer instead to base classes,
 * `AbstractMixing` and `BaseMixing`.
 */

class LogitSBMixing
    : public BaseMixing<LogitSBMixing, LogitSB::State, bayesmix::LogSBPrior> {
 public:
  LogitSBMixing() = default;
  ~LogitSBMixing() = default;

  //! Performs conditional update of state, given allocations and unique values
  //! @param unique_values  A vector of (pointers to) Hierarchy objects
  //! @param allocations    A vector of allocations label
  void update_state(
      const std::vector<std::shared_ptr<AbstractHierarchy>> &unique_values,
      const std::vector<unsigned int> &allocations) override;

  //! Read and set state values from a given Protobuf message
  void set_state_from_proto(const google::protobuf::Message &state_) override;

  //! Writes current state to a Protobuf message and return a shared_ptr
  //! New hierarchies have to first modify the field 'oneof val' in the
  //! MixingState message by adding the appropriate type
  std::shared_ptr<bayesmix::MixingState> get_state_proto() const override;

  //! Returns the Protobuf ID associated to this class
  bayesmix::MixingId get_id() const override {
    return bayesmix::MixingId::LogSB;
  }

  //! Returns the normalized acceptance rates of the Metropolis steps
  Eigen::VectorXd get_acceptance_rates() const {
    return acceptance_rates / n_iter;
  }

  //! Returns whether the mixing is conditional or marginal
  bool is_conditional() const override { return true; }

  //! Returns whether the mixing depends on covariate values or not
  bool is_dependent() const override { return true; }

 protected:
  //! Returns mixing weights (for conditional mixings only)
  //! @param log        Whether to return logarithm-scale values or not
  //! @param propto     Whether to include normalizing constants or not
  //! @param covariate  Covariate vector
  //! @return           The vector of mixing weights
  Eigen::VectorXd mixing_weights(
      const bool log, const bool propto,
      const Eigen::RowVectorXd &covariate) const override;

  //! Initializes state parameters to appropriate values
  void initialize_state() override;

  //! Sigmoid function
  double sigmoid(const double x) const { return 1.0 / (1.0 + std::exp(-x)); }

  //! Full-condit. distribution in alpha, given allocations and unique values
  double full_cond_lpdf(const Eigen::VectorXd &alpha, const unsigned int clust,
                        const std::vector<unsigned int> &allocations);

  //! Gradient of `full_cond_lpdf()`
  Eigen::VectorXd grad_full_cond_lpdf(
      const Eigen::VectorXd &alpha, const unsigned int clust,
      const std::vector<unsigned int> &allocations);

  //! Dimension of the coefficients vector
  unsigned int dim;

  //! Acceptance rates of the Metropolis steps
  Eigen::VectorXd acceptance_rates;

  //! Number of Metropolis steps performed
  int n_iter = 0;
};

#endif  // BAYESMIX_MIXINGS_LOGIT_SB_MIXING_H_
