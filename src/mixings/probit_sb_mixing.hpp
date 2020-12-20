#ifndef BAYESMIX_MIXINGS_PROBIT_SB_MIXING_HPP_
#define BAYESMIX_MIXINGS_PROBIT_SB_MIXING_HPP_

#include <google/protobuf/message.h>
#include <memory>

#include "base_mixing.hpp"
#include "../hierarchies/base_hierarchy.hpp"

class ProbitSBMixing : public BaseMixing {  // TODO all
 public:
  struct State {};

 protected:
  State state;

 public:
  // DESTRUCTOR AND CONSTRUCTORS
  ~ProbitSBMixing() = default;
  ProbitSBMixing() = default;

  void initialize() override {}

  void update_state(
      const std::vector<std::shared_ptr<BaseHierarchy>> &unique_values,
      unsigned int n) override;

  // GETTERS AND SETTERS
  State get_state() const { return state; }
  void set_state_from_proto(
  	const google::protobuf::Message &state_) override {}
  void write_state_to_proto(google::protobuf::Message *out) const override;
  std::string get_id() const override { return "ProbitSB"; }
};

#endif  // BAYESMIX_MIXINGS_PROBIT_SB_MIXING_HPP_
