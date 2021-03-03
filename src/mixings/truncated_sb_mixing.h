#ifndef BAYESMIX_MIXINGS_TRUNCATED_SB_MIXING_H_
#define BAYESMIX_MIXINGS_TRUNCATED_SB_MIXING_H_

#include <google/protobuf/message.h>

#include <Eigen/Dense>
#include <memory>
#include <vector>

#include "base_mixing.h"
#include "conditional_mixing.h"
#include "mixing_id.pb.h"
#include "mixing_prior.pb.h"
#include "src/hierarchies/abstract_hierarchy.h"

class TruncatedSBMixing : public ConditionalMixing {
 public:
  struct State {
    Eigen::VectorXd sticks;
    Eigen::VectorXd logweights;
  };

 protected:
  State state;

  //!
  Eigen::VectorXd logweights_from_sticks() const;
  //!
  Eigen::MatrixXd get_prior_shape_parameters() const;

  //!
  void create_empty_prior() override {
    prior.reset(new bayesmix::TruncSBPrior);
  }
  //!
  std::shared_ptr<bayesmix::TruncSBPrior> cast_prior() const {
    return std::dynamic_pointer_cast<bayesmix::TruncSBPrior>(prior);
  }
  //!
  void initialize_state() override;

 public:
  // DESTRUCTOR AND CONSTRUCTORS
  ~TruncatedSBMixing() = default;
  TruncatedSBMixing() = default;

  Eigen::VectorXd get_weights(const bool log, const bool propto,
                              const Eigen::RowVectorXd &covariate =
                                  Eigen::RowVectorXd(0)) const override;

  std::shared_ptr<BaseMixing> clone() const override {
    return std::make_shared<TruncatedSBMixing>(*this);
  }

  //! Returns true if the hierarchy has covariates i.e. is a dependent model
  bool is_dependent() const override { return false; }
  //!
  void initialize() override;
  //!
  void update_state(
      const std::vector<std::shared_ptr<AbstractHierarchy>> &unique_values,
      const std::vector<unsigned int> &allocations) override;

  // GETTERS AND SETTERS
  State get_state() const { return state; }
  void set_state_from_proto(const google::protobuf::Message &state_) override;
  void write_state_to_proto(google::protobuf::Message *out) const override;
  bayesmix::MixingId get_id() const override {
    return bayesmix::MixingId::TruncSB;
  }
};

#endif  // BAYESMIX_MIXINGS_TRUNCATED_SB_MIXING_H_
