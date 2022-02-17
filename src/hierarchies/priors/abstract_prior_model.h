#ifndef BAYESMIX_HIERARCHIES_PRIORS_ABSTRACT_PRIOR_MODEL_H_
#define BAYESMIX_HIERARCHIES_PRIORS_ABSTRACT_PRIOR_MODEL_H_

#include <google/protobuf/message.h>

#include <random>
#include <stan/math/prim.hpp>
#include <vector>

#include "algorithm_state.pb.h"
#include "src/hierarchies/likelihoods/states/includes.h"
#include "src/utils/rng.h"

class AbstractPriorModel {
 public:
  virtual ~AbstractPriorModel() = default;

  // IMPLEMENTED in BasePriorModel
  virtual std::shared_ptr<AbstractPriorModel> clone() const = 0;

  virtual double lpdf(const google::protobuf::Message &state_) = 0;

  //! Evaluates the log likelihood for unconstrained parameter values.
  //! By unconstrained parameters we mean that each entry of
  //! the parameter vector can range over (-inf, inf).
  //! Usually, some kind of transformation is required from the unconstrained
  //! parameterization to the actual parameterization.
  virtual double lpdf_from_unconstrained(
      Eigen::VectorXd unconstrained_params) const {
    throw std::runtime_error("lpdf_from_unconstrained() not yet implemented");
  }

  virtual stan::math::var lpdf_from_unconstrained(
      Eigen::Matrix<stan::math::var, Eigen::Dynamic, 1> unconstrained_params)
      const {
    throw std::runtime_error(
        "cluster_lpdf_from_unconstrained() not yet implemented");
  }

  // Da pensare, come restituisco lo stato? magari un pointer? Oppure delego
  virtual std::shared_ptr<google::protobuf::Message> sample(
      bool use_post_hypers) = 0;

  virtual void update_hypers(
      const std::vector<bayesmix::AlgorithmState::ClusterState> &states) = 0;

  virtual google::protobuf::Message *get_mutable_prior() = 0;

  virtual void set_hypers_from_proto(
      const google::protobuf::Message &state_) = 0;

  // IMPLEMENTED in BasePriorModel
  virtual void write_hypers_to_proto(google::protobuf::Message *out) const = 0;

 protected:
  virtual std::shared_ptr<bayesmix::AlgorithmState::HierarchyHypers>
  get_hypers_proto() const = 0;

  virtual void initialize_hypers() = 0;
};

#endif  // BAYESMIX_HIERARCHIES_PRIORS_ABSTRACT_PRIOR_MODEL_H_
