#ifndef BAYESMIX_MIXINGS_LOGIT_SB_MIXING_H_
#define BAYESMIX_MIXINGS_LOGIT_SB_MIXING_H_

#include <google/protobuf/message.h>

#include <Eigen/Dense>
#include <memory>
#include <vector>

#include "base_mixing.h"
#include "mixing_id.pb.h"
#include "mixing_prior.pb.h"
#include "src/hierarchies/abstract_hierarchy.h"

//! Class that represents the logit stick-break mixture model.

//! This class represents the dependent logit stick-break model introduced in
//! Rigon and Durante (2020). Please refer to this paper for a thorough
//! explanation of the model. In short, it represents a regressive model with
//! an arbitrary fixed number of maximum components which uses an enhanced
//! stick-breaking representation of the Dirichlet Process model. Its state is
//! composed of the regression coefficients and a precision matrix. Instead of
//! the Polya-Gamma augmentation proposed in the paper, which would mandate the
//! need to sample from non-standard distributions, a Metropolis-adjusted
//! Langevin algorithm (MALA) step is used, which requires the implementation
//! of some mathematical functions and gradients in this class. It also stores
//! acceptance rates for the performed MALA trials.
//! For more information about the class, please refer instead to base classes,
//! `AbstractMixing` and `BaseMixing`.

namespace LogitSB {
struct State {
  Eigen::MatrixXd regression_coeffs, precision;
};
};  // namespace LogitSB

class LogitSBMixing
    : public BaseMixing<LogitSBMixing, LogitSB::State, bayesmix::LogSBPrior> {
 public:
  LogitSBMixing() = default;
  ~LogitSBMixing() = default;

  void initialize() override;

  void update_state(
      const std::vector<std::shared_ptr<AbstractHierarchy>> &unique_values,
      const std::vector<unsigned int> &allocations) override;

  Eigen::VectorXd get_weights(const bool log, const bool propto,
                              const Eigen::RowVectorXd &covariate =
                                  Eigen::RowVectorXd(0)) const override;

  void set_state_from_proto(const google::protobuf::Message &state_) override;

  void write_state_to_proto(google::protobuf::Message *out) const override;

  bayesmix::MixingId get_id() const override {
    return bayesmix::MixingId::LogSB;
  }

  //! Returns the normalized acceptance rates of the Metropolis steps
  Eigen::VectorXd get_acceptance_rates() const {
    return acceptance_rates / n_iter;
  }

  bool is_conditional() const override { return true; }

  bool is_dependent() const override { return true; }

 protected:
  //! Dimension of the coefficients vector
  unsigned int dim;

  //! Acceptance rates of the Metropolis steps
  Eigen::VectorXd acceptance_rates;

  //! Number of Metropolis steps performed
  int n_iter = 0;

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
};

#endif  // BAYESMIX_MIXINGS_LOGIT_SB_MIXING_H_
