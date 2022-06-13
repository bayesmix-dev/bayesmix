#ifndef BAYESMIX_HIERARCHIES_PRIORS_PRIOR_MODEL_INTERNAL_H_
#define BAYESMIX_HIERARCHIES_PRIORS_PRIOR_MODEL_INTERNAL_H_

//! These functions exploit SFINAE to manage exception handling in all methods
//! required only if end user wants to rely on Metropolis-like updaters. SFINAE
//! (Substitution Failure Is Not An Error) is a C++ rule that applies during
//! overload resolution of function templates: When substituting the explicitly
//! specified or deduced type for the template parameter fails, the
//! specialization is discarded from the overload set instead of causing a
//! compile error. This feature is used in template metaprogramming.

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
