#ifndef BAYESMIX_MIXINGS_BASE_MIXING_H_
#define BAYESMIX_MIXINGS_BASE_MIXING_H_

#include <google/protobuf/message.h>

#include <Eigen/Dense>
#include <memory>

#include "mixing_id.pb.h"
#include "src/hierarchies/base_hierarchy.h"

//! Abstract base class for a generic mixture model

//! This class represents a mixture model object to be used in a BNP iterative
//! algorithm. By definition, a mixture is a probability distribution that
//! integrates over a density kernel to generate the actual distribution for
//! the data. However, in the context of this library, where a clustering
//! structure is generated on the data, a certain mixture translates to a
//! certain way of weighing the insertion of data in old clusters vs the
//! creation of new clusters. Therefore any mixture object inheriting from the
//! class must have methods that provide the probabilities for the two
//! aforementioned events. The class will then have its own parameters, and
//! maybe even prior distributions on them.

class BaseMixing {
 protected:
  std::shared_ptr<google::protobuf::Message> prior;

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
      unsigned int n) = 0;

  // GETTERS AND SETTERS
  google::protobuf::Message *get_mutable_prior() {
    if (prior == nullptr) {
      create_empty_prior();
    }
    return prior.get();
  }
  virtual void set_state_from_proto(
      const google::protobuf::Message &state_) = 0;
  virtual void write_state_to_proto(google::protobuf::Message *out) const = 0;
  virtual bayesmix::MixingId get_id() const = 0;
};

#endif  // BAYESMIX_MIXINGS_BASE_MIXING_H_
