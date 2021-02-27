#ifndef BAYESMIX_MIXINGS_BASE_MIXING_H_
#define BAYESMIX_MIXINGS_BASE_MIXING_H_

#include <google/protobuf/message.h>

#include <Eigen/Dense>
#include <memory>
#include <vector>

#include "mixing_id.pb.h"
#include "src/hierarchies/abstract_hierarchy.h"

//! Abstract base class for a generic mixture model

//! This class represents a mixture model object to be used in a BNP iterative
//! algorithm.

class BaseMixing {
 protected:
  //!
  std::shared_ptr<google::protobuf::Message> prior;
  //!
  const Eigen::MatrixXd *covariates_ptr;

  //!
  virtual void create_empty_prior() = 0;
  //!
  virtual void initialize_state() = 0;

 public:
  // DESTRUCTOR AND CONSTRUCTORS
  virtual ~BaseMixing() = default;
  BaseMixing() = default;
  virtual std::shared_ptr<BaseMixing> clone() const = 0;

  virtual void initialize() = 0;
  //! Returns true if the mixing has covariates i.e. is a dependent model
  virtual bool is_dependent() const { return false; }
  //!
  virtual bool is_conditional() const = 0;

  virtual void update_state(
      const std::vector<std::shared_ptr<AbstractHierarchy>> &unique_values,
      const std::vector<unsigned int> &allocations) = 0;

  // GETTERS AND SETTERS
  google::protobuf::Message *get_mutable_prior() {
    if (prior == nullptr) {
      create_empty_prior();
    }
    return prior.get();
  }
  void set_covariates(Eigen::MatrixXd *covar) { covariates_ptr = covar; }
  virtual void set_state_from_proto(
      const google::protobuf::Message &state_) = 0;
  virtual void write_state_to_proto(google::protobuf::Message *out) const = 0;
  virtual bayesmix::MixingId get_id() const = 0;
};

#endif  // BAYESMIX_MIXINGS_BASE_MIXING_H_
