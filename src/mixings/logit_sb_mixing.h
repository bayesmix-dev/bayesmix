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

namespace LogitSB {
struct State {
  Eigen::MatrixXd regression_coeffs, precision;
};
};  // namespace LogitSB

class LogitSBMixing
    : public BaseMixing<LogitSBMixing, LogitSB::State, bayesmix::LogSBPrior> {
 protected:
  unsigned int dim;
  Eigen::VectorXd acceptance_rates;
  int n_iter = 0;

  //!
  void initialize_state() override;
  //!
  double sigmoid(const double x) const { return 1.0 / (1.0 + std::exp(-x)); }
  //!
  double full_cond_lpdf(const Eigen::VectorXd &alpha, const unsigned int clust,
                        const std::vector<unsigned int> &allocations);
  //!
  Eigen::VectorXd grad_full_cond_lpdf(
      const Eigen::VectorXd &alpha, const unsigned int clust,
      const std::vector<unsigned int> &allocations);

 public:
  // DESTRUCTOR AND CONSTRUCTORS
  ~LogitSBMixing() = default;
  LogitSBMixing() = default;
  //!
  Eigen::VectorXd get_weights(const bool log, const bool propto,
                              const Eigen::RowVectorXd &covariate =
                                  Eigen::RowVectorXd(0)) const override;
  //!
  virtual bool is_conditional() const { return true; }
  //! Returns true if the hierarchy has covariates i.e. is a dependent model
  virtual bool is_dependent() const { return true; }

  //!
  void initialize() override;
  //!
  void update_state(
      const std::vector<std::shared_ptr<AbstractHierarchy>> &unique_values,
      const std::vector<unsigned int> &allocations) override;

  // GETTERS AND SETTERS
  void set_state_from_proto(const google::protobuf::Message &state_) override;
  void write_state_to_proto(google::protobuf::Message *out) const override;
  bayesmix::MixingId get_id() const override {
    return bayesmix::MixingId::LogSB;
  }
  Eigen::VectorXd get_acceptance_rates() const {
    return acceptance_rates / n_iter;
  }
};

#endif  // BAYESMIX_MIXINGS_LOGIT_SB_MIXING_H_
