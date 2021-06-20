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

  State get_state() const { return state; }

  unsigned int get_num_components() const override { return num_components; }

  google::protobuf::Message *get_mutable_prior() override {
    if (prior == nullptr) {
      create_empty_prior();
    }
    return prior.get();
  }

  void set_num_components(const unsigned int num_) override {
    num_components = num_;
  }

  void set_covariates(Eigen::MatrixXd *covar) override {
    covariates_ptr = covar;
  }

 protected:
  void create_empty_prior() { prior.reset(new Prior); }
  //! Initializes the mixing state to appropriate values
  virtual void initialize_state() = 0;
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

#endif  // BAYESMIX_MIXINGS_BASE_MIXING_H_
