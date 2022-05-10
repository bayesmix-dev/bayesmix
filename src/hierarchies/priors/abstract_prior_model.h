#ifndef BAYESMIX_HIERARCHIES_PRIORS_ABSTRACT_PRIOR_MODEL_H_
#define BAYESMIX_HIERARCHIES_PRIORS_ABSTRACT_PRIOR_MODEL_H_

#include <google/protobuf/message.h>

#include <random>
#include <stan/math/prim.hpp>
#include <vector>

#include "algorithm_state.pb.h"
#include "src/hierarchies/likelihoods/states/includes.h"
#include "src/utils/rng.h"

//! Abstract class for a generic prior model
//!
//! This class is the basis for a curiously recurring template pattern (CRTP)
//! for `PriorModel` objects, ad it is solely composed of interface functions
//! for derived classes to use.
//!
//! A prior model represents the prior for the parameters in the likelihood.
//! Hence, it can evaluate the log probability density function (lpdf) for a
//! given parameter state.
//!
//! We also store a pointer to the protobuf object that represents the type of
//! prior used fot the parameters in the likelihood.

class AbstractPriorModel {
 public:
  // Useful type aliases
  using ProtoHypersPtr =
      std::shared_ptr<bayesmix::AlgorithmState::HierarchyHypers>;
  using ProtoHypers = ProtoHypersPtr::element_type;

  //! Default destructor
  virtual ~AbstractPriorModel() = default;

  //! Returns an independent, data-less copy of this object
  virtual std::shared_ptr<AbstractPriorModel> clone() const = 0;

  //! Returns an independent, data-less deep copy of this object
  virtual std::shared_ptr<AbstractPriorModel> deep_clone() const = 0;

  //! Evaluates the log likelihood for the prior model, given the state of the
  //! cluster
  //! @param state_ A Protobuf message storing the current state of the cluster
  //! @return The evaluation of the log likelihood
  virtual double lpdf(const google::protobuf::Message &state_) = 0;

  //! Evaluates the log likelihood for unconstrained parameter values.
  //! By unconstrained parameters we mean that each entry of
  //! the parameter vector can range over (-inf, inf).
  //! Usually, some kind of transformation is required from the unconstrained
  //! parameterization to the actual parameterization.
  //! @param unconstrained_params vector collecting the unconstrained
  //! parameters
  //! @return The evaluation of the log likelihood of the prior model
  virtual double lpdf_from_unconstrained(
      Eigen::VectorXd unconstrained_params) const {
    throw std::runtime_error("lpdf_from_unconstrained() not yet implemented");
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
  virtual stan::math::var lpdf_from_unconstrained(
      Eigen::Matrix<stan::math::var, Eigen::Dynamic, 1> unconstrained_params)
      const {
    throw std::runtime_error(
        "cluster_lpdf_from_unconstrained() not yet implemented");
  }

  //! Sampling from the prior model
  //! @param hier_hypers A pointer to a
  //! `bayesmix::AlgorithmState::hierarchyHypers` object, which defines the
  //! parameters to use for the sampling. The default behaviour (i.e.
  //! `hier_hypers = nullptr`) uses prior hyperparameters
  //! @return A Protobuf message storing the state sampled from the prior model
  virtual std::shared_ptr<google::protobuf::Message> sample_proto(
      ProtoHypersPtr hier_hypers = nullptr) = 0;

  //! Updates hyperparameter values given a vector of cluster states
  virtual void update_hypers(
      const std::vector<bayesmix::AlgorithmState::ClusterState> &states) = 0;

  //! Returns a pointer to the Protobuf message of the prior of this cluster
  virtual google::protobuf::Message *get_mutable_prior() = 0;

  //! Read and set hyperparameter values from a given Protobuf message
  virtual void set_hypers_from_proto(
      const google::protobuf::Message &state_) = 0;

  //! Writes current values of the hyperparameters to a Protobuf message by
  //! pointer
  virtual void write_hypers_to_proto(google::protobuf::Message *out) const = 0;

  //! Writes current value of hyperparameters to a Protobuf message and
  //! return a shared_ptr.
  //! New hierarchies have to first modify the field 'oneof val' in the
  //! AlgoritmState::HierarchyHypers message by adding the appropriate type
  virtual std::shared_ptr<bayesmix::AlgorithmState::HierarchyHypers>
  get_hypers_proto() const = 0;

 protected:
  //! Initializes hierarchy hyperparameters to appropriate values
  virtual void initialize_hypers() = 0;
};

#endif  // BAYESMIX_HIERARCHIES_PRIORS_ABSTRACT_PRIOR_MODEL_H_
