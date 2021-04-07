#ifndef BAYESMIX_MIXINGS_TRUNCATED_SB_MIXING_H_
#define BAYESMIX_MIXINGS_TRUNCATED_SB_MIXING_H_

#include <google/protobuf/message.h>

#include <Eigen/Dense>
#include <memory>
#include <vector>

#include "base_mixing.h"
#include "mixing_id.pb.h"
#include "mixing_prior.pb.h"
#include "src/hierarchies/abstract_hierarchy.h"

namespace TruncSB {
struct State {
  Eigen::VectorXd sticks, logweights;
};
};  // namespace TruncSB

class TruncatedSBMixing : public BaseMixing<TruncatedSBMixing, TruncSB::State,
                                            bayesmix::TruncSBPrior> {
 protected:
  //!
  Eigen::VectorXd logweights_from_sticks() const;
  //!
  Eigen::MatrixXd get_prior_shape_parameters() const;
  //!
  void initialize_state() override;

 public:
  // DESTRUCTOR AND CONSTRUCTORS
  ~TruncatedSBMixing() = default;
  TruncatedSBMixing() = default;
  //!
  Eigen::VectorXd get_weights(const bool log, const bool propto,
                              const Eigen::RowVectorXd &covariate =
                                  Eigen::RowVectorXd(0)) const override;
  //!
  virtual bool is_conditional() const { return true; }
  //! Returns true if the hierarchy has covariates i.e. is a dependent model
  bool is_dependent() const override { return false; }
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
    return bayesmix::MixingId::TruncSB;
  }
};

#endif  // BAYESMIX_MIXINGS_TRUNCATED_SB_MIXING_H_
