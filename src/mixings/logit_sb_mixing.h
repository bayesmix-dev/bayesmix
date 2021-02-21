#ifndef BAYESMIX_MIXINGS_LOGIT_SB_MIXING_H_
#define BAYESMIX_MIXINGS_LOGIT_SB_MIXING_H_

#include <google/protobuf/message.h>

#include <Eigen/Dense>
#include <memory>

#include "base_mixing.h"
#include "conditional_mixing.h"
#include "mixing_id.pb.h"
#include "mixing_prior.pb.h"
#include "src/hierarchies/base_hierarchy.h"

class LogitSBMixing : public ConditionalMixing {
 public:
  struct State {
    Eigen::MatrixXd regression_coeffs;
  };

 protected:
  unsigned int num_clusters;
  unsigned int dim;
  State state;
  Eigen::MatrixXd precision;  // TODO should it be in the state?

  //!
  void create_empty_prior() override { prior.reset(new bayesmix::LogSBPrior); }
  //!
  std::shared_ptr<bayesmix::LogSBPrior> cast_prior() {
    return std::dynamic_pointer_cast<bayesmix::LogSBPrior>(prior);
  }
  //!
  void initialize_state() override;
  //!
  double sigmoid(const double x) const { return 1.0 / (1.0 + std::exp(-x)); }
  //!
  Eigen::VectorXd grad_log_full_cond(const Eigen::VectorXd &alpha,
                                     const std::vector<bool> &is_curr_clus,
                                     const std::vector<bool> &is_prev_clus);
  //!
  double log_like(const Eigen::VectorXd &alpha,
                  const std::vector<bool> &is_curr_clus,
                  const std::vector<bool> &is_prev_clus);

 public:
  // DESTRUCTOR AND CONSTRUCTORS
  ~LogitSBMixing() = default;
  LogitSBMixing() = default;

  Eigen::VectorXd get_weights(const Eigen::VectorXd &covariate) const override;

  std::shared_ptr<BaseMixing> clone() const override {
    return std::make_shared<LogitSBMixing>(*this);
  }

  //! Returns true if the hierarchy has covariates i.e. is a dependent model
  bool is_dependent() const override { return true; }

  //!
  void initialize() override;
  //!
  void update_state(
      const std::vector<std::shared_ptr<AbstractHierarchy>> &unique_values,
      const std::vector<unsigned int> &allocations,
      const unsigned int n) override;

  // GETTERS AND SETTERS
  State get_state() const { return state; }
  void set_state_from_proto(const google::protobuf::Message &state_) override;
  void write_state_to_proto(google::protobuf::Message *out) const override;
  bayesmix::MixingId get_id() const override {
    return bayesmix::MixingId::LogSB;
  }
};

#endif  // BAYESMIX_MIXINGS_LOGIT_SB_MIXING_H_
