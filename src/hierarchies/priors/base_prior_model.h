#ifndef BAYESMIX_HIERARCHIES_PRIORS_BASE_PRIOR_MODEL_H_
#define BAYESMIX_HIERARCHIES_PRIORS_BASE_PRIOR_MODEL_H_

#include <google/protobuf/message.h>

#include <memory>
#include <random>
#include <set>
#include <stan/math/prim.hpp>
#include <stan/math/rev.hpp>

#include "abstract_prior_model.h"
#include "algorithm_state.pb.h"
#include "hierarchy_id.pb.h"
#include "src/utils/rng.h"

namespace internal {

template <class Prior, typename T>
auto lpdf_from_unconstrained(
    const Prior &prior,
    Eigen::Matrix<T, Eigen::Dynamic, 1> unconstrained_params, int)
    -> decltype(prior.template lpdf_from_unconstrained<T>(
        unconstrained_params)) {
  return prior.template lpdf_from_unconstrained<T>(unconstrained_params);
}

template <class Prior, typename T>
auto lpdf_from_unconstrained(
    const Prior &prior,
    Eigen::Matrix<T, Eigen::Dynamic, 1> unconstrained_params, double) -> T {
  throw(std::runtime_error("lpdf_from_unconstrained() not yet implemented"));
}

}  // namespace internal

template <class Derived, typename HyperParams, typename Prior>
class BasePriorModel : public AbstractPriorModel {
 public:
  //! Default constructor
  BasePriorModel() = default;

  //! Default destructor
  ~BasePriorModel() = default;

  //! Evaluates the log likelihood for unconstrained parameter values.
  //! By unconstrained parameters we mean that each entry of
  //! the parameter vector can range over (-inf, inf).
  //! Usually, some kind of transformation is required from the unconstrained
  //! parameterization to the actual parameterization.
  //! @param unconstrained_params vector collecting the unconstrained
  //! parameters
  //! @return The evaluation of the log likelihood of the prior model
  double lpdf_from_unconstrained(
      Eigen::VectorXd unconstrained_params) const override {
    return internal::lpdf_from_unconstrained(
        static_cast<const Derived &>(*this), unconstrained_params, 0);
  }

  //! Evaluates the log likelihood for unconstrained parameter values.
  //! By unconstrained parameters we mean that each entry of
  //! the parameter vector can range over (-inf, inf).
  //! Usually, some kind of transformation is required from the unconstrained
  //! parameterization to the actual parameterization. This version using
  //! `stan::math::var` type is required for Stan automatic aifferentiation.
  //! @param unconstrained_params vector collecting the unconstrained
  //! parameters
  //! @return The evaluation of the log likelihood of the prior model
  stan::math::var lpdf_from_unconstrained(
      Eigen::Matrix<stan::math::var, Eigen::Dynamic, 1> unconstrained_params)
      const override {
    return internal::lpdf_from_unconstrained(
        static_cast<const Derived &>(*this), unconstrained_params, 0);
  }

  //! Returns an independent, data-less copy of this object
  std::shared_ptr<AbstractPriorModel> clone() const override;

  //! Returns an independent, data-less deep copy of this object
  std::shared_ptr<AbstractPriorModel> deep_clone() const override;

  //! Returns a pointer to the Protobuf message of the prior of this cluster
  google::protobuf::Message *get_mutable_prior() override;

  //! Returns the struct of the current prior hyperparameters
  HyperParams get_hypers() const { return *hypers; }

  //! Returns the struct of the current posterior hyperparameters
  HyperParams get_posterior_hypers() const { return post_hypers; }

  //! Updates the current value of the posterior hyperparameters
  void set_posterior_hypers(const HyperParams &_post_hypers) {
    post_hypers = _post_hypers;
  };

  //! Writes current values of the hyperparameters to a Protobuf message by
  //! pointer
  void write_hypers_to_proto(google::protobuf::Message *out) const override;

  //! Initializes the prior model (both prior and hyperparameters)
  void initialize();

 protected:
  //! Raises an error if the prior pointer is not initialized
  void check_prior_is_set() const;

  //! Re-initializes the prior of the hierarchy to a newly created object
  void create_empty_prior() { prior.reset(new Prior); }

  //! Re-initializes the hyperparameters of the hierarchy to a newly created
  //! object
  void create_empty_hypers() { hypers.reset(new HyperParams); }

  //! Down-casts the given generic proto message to a HierarchyHypers proto
  bayesmix::AlgorithmState::HierarchyHypers *downcast_hypers(
      google::protobuf::Message *state_) const {
    return google::protobuf::internal::down_cast<
        bayesmix::AlgorithmState::HierarchyHypers *>(state_);
  }

  //! Down-casts the given generic proto message to a HierarchyHypers proto
  const bayesmix::AlgorithmState::HierarchyHypers &downcast_hypers(
      const google::protobuf::Message &state_) const {
    return google::protobuf::internal::down_cast<
        const bayesmix::AlgorithmState::HierarchyHypers &>(state_);
  }

  //! Down-casts the given generic proto message to a ClusterState proto
  const bayesmix::AlgorithmState::ClusterState &downcast_state(
      const google::protobuf::Message &state_) const {
    return google::protobuf::internal::down_cast<
        const bayesmix::AlgorithmState::ClusterState &>(state_);
  }

  //! Container for prior hyperparameters values
  std::shared_ptr<HyperParams> hypers = std::make_shared<HyperParams>();

  //! Container for posterior hyperparameters values
  HyperParams post_hypers;

  //! Pointer to a Protobuf prior object for this class
  std::shared_ptr<Prior> prior;
};

/* *** Methods Definitions *** */
template <class Derived, typename HyperParams, typename Prior>
std::shared_ptr<AbstractPriorModel>
BasePriorModel<Derived, HyperParams, Prior>::clone() const {
  auto out = std::make_shared<Derived>(static_cast<Derived const &>(*this));
  return out;
}

template <class Derived, typename HyperParams, typename Prior>
std::shared_ptr<AbstractPriorModel>
BasePriorModel<Derived, HyperParams, Prior>::deep_clone() const {
  auto out = std::make_shared<Derived>(static_cast<Derived const &>(*this));

  // Prior Deep-clone
  out->create_empty_prior();
  std::shared_ptr<google::protobuf::Message> new_prior(prior->New());
  new_prior->CopyFrom(*prior.get());
  out->get_mutable_prior()->CopyFrom(*new_prior.get());

  // HyperParams Deep-clone
  out->create_empty_hypers();
  auto curr_hypers_proto = get_hypers_proto();
  out->set_hypers_from_proto(*curr_hypers_proto.get());

  // Initialization of Deep-cloned object
  out->initialize();

  return out;
}

template <class Derived, typename HyperParams, typename Prior>
google::protobuf::Message *
BasePriorModel<Derived, HyperParams, Prior>::get_mutable_prior() {
  if (prior == nullptr) {
    create_empty_prior();
  }
  return prior.get();
}

template <class Derived, typename HyperParams, typename Prior>
void BasePriorModel<Derived, HyperParams, Prior>::write_hypers_to_proto(
    google::protobuf::Message *out) const {
  std::shared_ptr<bayesmix::AlgorithmState::HierarchyHypers> hypers_ =
      get_hypers_proto();
  auto *out_cast = downcast_hypers(out);
  out_cast->CopyFrom(*hypers_.get());
}

template <class Derived, typename HyperParams, typename Prior>
void BasePriorModel<Derived, HyperParams, Prior>::initialize() {
  check_prior_is_set();
  create_empty_hypers();
  initialize_hypers();
}

template <class Derived, typename HyperParams, typename Prior>
void BasePriorModel<Derived, HyperParams, Prior>::check_prior_is_set() const {
  if (prior == nullptr) {
    throw std::invalid_argument("Hierarchy prior was not provided");
  }
}

// OLD STUFF
// double lpdf_from_unconstrained(
//     Eigen::VectorXd unconstrained_params) const override {
//   return static_cast<const Derived &>(*this)
//       .template lpdf_from_unconstrained<double>(unconstrained_params);
// }

// stan::math::var lpdf_from_unconstrained(
//     Eigen::Matrix<stan::math::var, Eigen::Dynamic, 1>
//     unconstrained_params) const override {
//   return static_cast<const Derived &>(*this)
//       .template lpdf_from_unconstrained<stan::math::var>(
//           unconstrained_params);
// }

#endif  // BAYESMIX_HIERARCHIES_PRIORS_BASE_PRIOR_MODEL_H_
