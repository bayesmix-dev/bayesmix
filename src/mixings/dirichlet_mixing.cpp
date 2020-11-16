#include "dirichlet_mixing.hpp"
#include <google/protobuf/stubs/casts.h>
#include "../../proto/cpp/mixings.pb.h"

void DirichletMixing::set_state(const google::protobuf::Message &curr) {
  DPState *currcast = google::protobuf::internal::down_cast<DPState *>(curr);
  state = *currcast;
  if(state.has_fixed_value()){
  	totalmass = state.totalmass.value();
  }
  else if(state.has_gamma_prior()){
  	totalmass = state.totalmass.alpha() / state.totalmass.beta();
  }
  else {
  	std::invalid_argument("Error: argument proto is not appropriate");
  }
}
