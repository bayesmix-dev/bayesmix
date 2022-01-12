#ifndef BAYESMIX_HIERARCHIES_BASE_PRIORMODEL_H_
#define BAYESMIX_HIERARCHIES_BASE_PRIORMODEL_H_

#include <google/protobuf/message.h>

#include <Eigen/Dense>
#include <memory>
#include <random>
#include <set>
#include <stan/math/prim.hpp>

#include "abstract_prior_model.h"
#include "algorithm_state.pb.h"
#include "hierarchy_id.pb.h"
#include "src/utils/rng.h"

template <class Derived, typename Hyperparams, typename Prior>
class BasePriorModel : public AbstractPriorModel {
 public:
  BasePriorModel() = default;

  ~BasePriorModel() = default;

  virtual std::shared_ptr<AbstractPriorModel> clone() const override {
    auto out = std::make_shared<Derived>(static_cast<Derived const &>(*this));
    return out;
  }

  // sample method, che posso fare??

  virtual google::protobuf::Message *get_mutable_prior() override {
    if (prior == nullptr) {
      create_empty_prior();
    }
    return prior.get();
  }

  Hyperparams get_hypers() const { return *hypers; }

  void write_hypers_to_proto(google::protobuf::Message *out) const override;

 protected:
  void check_prior_is_set() const {
    if (prior == nullptr) {
      throw std::invalid_argument("Hierarchy prior was not provided");
    }
  }

  void create_empty_prior() { prior.reset(new Prior); }

  virtual std::shared_ptr<bayesmix::AlgorithmState::HierarchyHypers>
  get_hypers_proto() const = 0;

  virtual void initialize_hypers() = 0;

  bayesmix::AlgorithmState::HierarchyHypers *downcast_hypers(
      google::protobuf::Message *state_) const {
    return google::protobuf::internal::down_cast<
        bayesmix::AlgorithmState::HierarchyHypers *>(state_);
  }

  const bayesmix::AlgorithmState::HierarchyHypers &downcast_hypers(
      const google::protobuf::Message &state_) const {
    return google::protobuf::internal::down_cast<
        const bayesmix::AlgorithmState::HierarchyHypers &>(state_);
  }

  std::shared_ptr<Hyperparams> hypers;
  std::shared_ptr<Prior> prior;
};

template <class Derived, typename Hyperparams, typename Prior>
void BasePriorModel<Derived, Hyperparams, Prior>::write_hypers_to_proto(
    google::protobuf::Message *out) const {
  std::shared_ptr<bayesmix::AlgorithmState::HierarchyHypers> hypers_ =
      get_hypers_proto();
  auto *out_cast = downcast_hypers(out);
  out_cast->CopyFrom(*hypers_.get());
}

#endif  // BAYESMIX_HIERARCHIES_BASE_PRIORMODEL_H_
