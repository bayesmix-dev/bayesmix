#include "dirichlet_mixing.hpp"

#include <google/protobuf/stubs/casts.h>

#include <stan/math/prim/prob.hpp>

#include "../../proto/cpp/mixings.pb.h"
#include "../utils/rng.hpp"

void DirichletMixing::update_hypers(
    const std::vector<std::shared_ptr<BaseHierarchy>> &unique_values,
    unsigned int n) {
  if (params.has_fixed_value()) {
    return;
  } else if (params.has_gamma_prior()) {
    auto &rng = bayesmix::Rng::Instance().get();

    // Recover parameters
    unsigned int k = unique_values.size();
    double alpha = params.gamma_prior().alpha();
    double beta = params.gamma_prior().beta();

    double phi = stan::math::gamma_rng(totalmass + 1, n, rng);
    double odds = (alpha + k - 1) / (n * (beta - log(phi)));
    double prob = odds / (1 + odds);
    double p = stan::math::uniform_rng(0.0, 1.0, rng);
    if (p <= prob) {
      totalmass = stan::math::gamma_rng(alpha + k, beta - log(phi), rng);
    } else {
      totalmass = stan::math::gamma_rng(alpha + k - 1, beta - log(phi), rng);
    }
  } else {
    std::invalid_argument("Error: no possible valid update");
  }
}

void DirichletMixing::set_params(const google::protobuf::Message &params_) {
  const bayesmix::DPParams &currcast =
      google::protobuf::internal::down_cast<const bayesmix::DPParams &>(
          params_);
  params = currcast;
  if (params.has_fixed_value()) {
    totalmass = params.fixed_value().value();
  } else if (params.has_gamma_prior()) {
    totalmass = params.gamma_prior().alpha() / params.gamma_prior().beta();
  } else {
    std::invalid_argument("Error: argument proto is not appropriate");
  }
}
