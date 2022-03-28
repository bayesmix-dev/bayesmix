#ifndef BAYESMIX_HIERARCHIES_PRIORS_PRIOR_MODEL_INTERNAL_H_
#define BAYESMIX_HIERARCHIES_PRIORS_PRIOR_MODEL_INTERNAL_H_

namespace internal {

template <class Prior, typename T>
auto lpdf_from_unconstrained(
    const Prior &prior,
    Eigen::Matrix<T, Eigen::Dynamic, 1> unconstrained_params, int)
    -> decltype(prior.template lpdf_from_unconstrained<T>(
        unconstrained_params)) {
  return prior.template lpdf_from_unconstrained<T>(unconstrained_params);
}

template <class Prior, typename T>
auto lpdf_from_unconstrained(
    const Prior &prior,
    Eigen::Matrix<T, Eigen::Dynamic, 1> unconstrained_params, double) -> T {
  throw(std::runtime_error("lpdf_from_unconstrained() not yet implemented"));
}

}  // namespace internal

#endif  // BAYESMIX_HIERARCHIES_PRIORS_PRIOR_MODEL_INTERNAL_H_
