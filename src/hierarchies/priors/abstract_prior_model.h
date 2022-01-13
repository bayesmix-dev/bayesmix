#ifndef BAYESMIX_HIERARCHIES_ABSTRACT_PRIORMODEL_H_
#define BAYESMIX_HIERARCHIES_ABSTRACT_PRIORMODEL_H_

#include <google/protobuf/message.h>

#include <random>
#include <stan/math/prim.hpp>
#include <vector>

#include "algorithm_state.pb.h"
#include "src/utils/rng.h"

class AbstractPriorModel {
 public:
  virtual ~AbstractPriorModel() = default;

  virtual std::shared_ptr<AbstractPriorModel> clone() const = 0;

  virtual double lpdf() = 0;

  // Da pensare, come restituisco lo stato? magari un pointer?
  virtual void sample() = 0;

  virtual void update_hypers(
      const std::vector<bayesmix::AlgorithmState::ClusterState> &states) = 0;

  virtual void initialize_hypers() = 0;

  virtual google::protobuf::Message *get_mutable_prior() = 0;

  virtual void set_hypers_from_proto(
      const google::protobuf::Message &state_) = 0;

  virtual void write_hypers_to_proto(google::protobuf::Message *out) const = 0;

 protected:
  virtual std::shared_ptr<bayesmix::AlgorithmState::HierarchyHypers>
  get_hypers_proto() const = 0;

  virtual void initialize_hypers() = 0;
};

#endif