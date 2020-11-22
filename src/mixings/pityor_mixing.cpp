#include "pityor_mixing.hpp"

#include <google/protobuf/stubs/casts.h>

#include "../../proto/cpp/mixing_state.pb.h"

void PitYorMixing::write_state_to_proto(google::protobuf::Message *out) const {
  bayesmix::PYState state;
  state.set_strength(strength);
  state.set_discount(discount);

  google::protobuf::internal::down_cast<bayesmix::PYState *>(out)->CopyFrom(
      state);
}
