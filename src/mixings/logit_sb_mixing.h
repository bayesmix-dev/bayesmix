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

//! Class that represents the logit stick-breaking process indroduced
//! in Rigon and Durante (2020).
//! That it, a prior for weights (w_1,...,w_H), depending on covariates
//! x in R^p, in the H-1 dimensional unit simplex, defined as follows
//!   w_1(x) = v_1(x)
//!   w_j(x) = v_j(x) (1 - v_1(x)) ... (1 - v_{j-1}(x)), for j=2, ... H-1
//!   w_H(x) = 1 - (w_1(x) + w_2 + ... + w_{H-1}(x))
//! and
//!   v_j(x) = 1 / exp(- <alpha_j, x> ), for j = 1, ..., H-1
//!
//! The main difference with the paper Rigon and Durante (2020) is that 
//! they propose a Gibbs sampler in which the full conditionals are available
//! in close form thanks to a Polya-Gamma augmentation. 
//! Here instead, a Metropolis-adjusted Langevin algorithm (MALA) step is used.
//! The step-size of the MALA step must be passed in the LogSBPrior protobuf
//! message.
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
