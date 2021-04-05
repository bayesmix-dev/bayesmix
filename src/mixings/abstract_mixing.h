#ifndef BAYESMIX_MIXINGS_ABSTRACT_MIXING_H_
#define BAYESMIX_MIXINGS_ABSTRACT_MIXING_H_

#include <google/protobuf/message.h>

#include <Eigen/Dense>
#include <memory>
#include <vector>

#include "mixing_id.pb.h"
#include "src/hierarchies/abstract_hierarchy.h"

//! Abstract base class for a generic mixture model

//! This class represents a mixture model object to be used in a BNP iterative
//! algorithm.

class AbstractMixing {
 public:
  // DESTRUCTOR AND CONSTRUCTORS
  virtual ~AbstractMixing() = default;
  AbstractMixing() = default;

  virtual void initialize() = 0;
  //! Returns true if the mixing has covariates i.e. is a dependent model
  virtual bool is_dependent() const { return false; }
  //!
  virtual bool is_conditional() const = 0;

  virtual void update_state(
      const std::vector<std::shared_ptr<AbstractHierarchy>> &unique_values,
      const std::vector<unsigned int> &allocations) = 0;

  // GETTERS AND SETTERS
  virtual unsigned int get_num_components() const = 0;
  virtual google::protobuf::Message *get_mutable_prior() = 0;
  virtual void set_num_components(const unsigned int num_) = 0;
  virtual void set_covariates(Eigen::MatrixXd *covar) = 0;
  virtual void set_state_from_proto(
      const google::protobuf::Message &state_) = 0;
  virtual void write_state_to_proto(google::protobuf::Message *out) const = 0;
  virtual bayesmix::MixingId get_id() const = 0;
};

#endif  // BAYESMIX_MIXINGS_ABSTRACT_MIXING_H_
