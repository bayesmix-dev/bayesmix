#ifndef BAYESMIX_MIXINGS_BASE_MIXING_H_
#define BAYESMIX_MIXINGS_BASE_MIXING_H_

#include <google/protobuf/message.h>

#include <Eigen/Dense>
#include <memory>

#include "abstract_mixing.h"

//! Base template class for a mixing object.

//! This class is a templatized version of, and derived from, the
//! `AbstractMixing` class, and the second stage of the curiously recurring
//! template pattern for `Mixing` objects (please see the file of the parent
//! class for further information). It includes class members and some more
//! functions which could not be implemented in the non-templatized abstract
//! class. When deriving a class from `BaseMixing`, its own name must be passed
//! to the first template argument, and custom containers for state and prior
//! values must be provided in the remaining ones.

//! @tparam Derived  Name of the implemented derived class
//! @tparam State    Class name of the container for state values
//! @tparam Prior    Class name of the container for prior parameters

template <class Derived, typename State, typename Prior>
class BaseMixing : public AbstractMixing {
 public:
  BaseMixing() = default;
  ~BaseMixing() = default;

  //! Returns the struct of the current state
  State get_state() const { return state; }

  //! Returns the current number of clusters of the mixture model
  unsigned int get_num_components() const override { return num_components; }

  //! Returns a pointer to the Protobuf message of the prior of this cluster
  google::protobuf::Message *get_mutable_prior() override;

  //! Sets the current number of clusters of the mixture model
  void set_num_components(const unsigned int num_) override {
    num_components = num_;
  }

  //! Sets the (pointer to the) covariate matrix for the mixture model
  void set_covariates(Eigen::MatrixXd *covar) override {
    covariates_ptr = covar;
  }

  //! Writes current state to a Protobuf message by pointer
  void write_state_to_proto(google::protobuf::Message *out) const override;

  //! Main function that initializes members to appropriate values
  void initialize() override;

 protected:
  //! Re-initializes the prior of the mixing to a newly created object
  void create_empty_prior() { prior.reset(new Prior); }

  //! Initializes state parameters to appropriate values
  virtual void initialize_state() = 0;

  //! Down-casts the given generic proto message to a MixingState proto
  bayesmix::MixingState *downcast_state(google::protobuf::Message *out) const {
    return google::protobuf::internal::down_cast<bayesmix::MixingState *>(out);
  }

  //! Down-casts the given generic proto message to a MixingState proto
  const bayesmix::MixingState &downcast_state(
      const google::protobuf::Message &state_) const {
    return google::protobuf::internal::down_cast<
        const bayesmix::MixingState &>(state_);
  }

  //! Converts prior from generic Protobuf message to its own type
  std::shared_ptr<Prior> cast_prior() const {
    return std::dynamic_pointer_cast<Prior>(prior);
  }

  //! Container object for the mixing state
  State state;

  //! Pointer to a Protobuf object representing the mixing's prior distribution
  std::shared_ptr<google::protobuf::Message> prior;

  //! Pointer to the covariate matrix for the mixture model
  const Eigen::MatrixXd *covariates_ptr;

  //! Current number of clusters of the mixture model
  unsigned int num_components;
};

template <class Derived, typename State, typename Prior>
google::protobuf::Message *
BaseMixing<Derived, State, Prior>::get_mutable_prior() {
  if (prior == nullptr) {
    create_empty_prior();
  }
  return prior.get();
}

template <class Derived, typename State, typename Prior>
void BaseMixing<Derived, State, Prior>::write_state_to_proto(
    google::protobuf::Message *out) const {
  auto outcast = downcast_state(out);
  outcast->CopyFrom(*get_state_proto().get());
}

template <class Derived, typename State, typename Prior>
void BaseMixing<Derived, State, Prior>::initialize() {
  if (prior == nullptr) {
    throw std::invalid_argument("Mixing prior was not provided");
  }
  initialize_state();
}

#endif  // BAYESMIX_MIXINGS_BASE_MIXING_H_
