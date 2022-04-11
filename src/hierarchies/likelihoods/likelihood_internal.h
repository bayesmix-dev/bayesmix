#ifndef BAYESMIX_HIERARCHIES_LIKELIHOODS_LIKELIHOOD_INTERNAL_H_
#define BAYESMIX_HIERARCHIES_LIKELIHOODS_LIKELIHOOD_INTERNAL_H_

//! These functions exploit SFINAE to manage exception handling in all methods
//! required only if end user wants to rely on Metropolis-like updaters. SFINAE
//! (Substitution Failure Is Not An Error) is a C++ rule that applies during
//! overload resolution of function templates: When substituting the explicitly
//! specified or deduced type for the template parameter fails, the
//! specialization is discarded from the overload set instead of causing a
//! compile error. This feature is used in template metaprogramming.

namespace internal {

/* SFINAE for cluster_lpdf_from_unconstrained() */
template <class Like, typename T>
auto cluster_lpdf_from_unconstrained(
    const Like &like, Eigen::Matrix<T, Eigen::Dynamic, 1> unconstrained_params,
    int)
    -> decltype(like.template cluster_lpdf_from_unconstrained<T>(
        unconstrained_params)) {
  return like.template cluster_lpdf_from_unconstrained<T>(
      unconstrained_params);
}
template <class Like, typename T>
auto cluster_lpdf_from_unconstrained(
    const Like &like, Eigen::Matrix<T, Eigen::Dynamic, 1> unconstrained_params,
    double) -> T {
  throw(std::runtime_error(
      "cluster_lpdf_from_unconstrained() not yet implemented"));
}

/* SFINAE for get_unconstrained_state() */
template <class State>
auto get_unconstrained_state(const State &state, int)
    -> decltype(state.get_unconstrained()) {
  return state.get_unconstrained();
}
template <class State>
auto get_unconstrained_state(const State &state, double) -> Eigen::VectorXd {
  throw(std::runtime_error("get_unconstrained_state() not yet implemented"));
}

/* SFINAE for set_state_from_unconstrained() */
template <class State>
auto set_state_from_unconstrained(State &state,
                                  const Eigen::VectorXd &unconstrained_state,
                                  int)
    -> decltype(state.set_from_unconstrained(unconstrained_state)) {
  state.set_from_unconstrained(unconstrained_state);
}
template <class State>
auto set_state_from_unconstrained(State &state,
                                  const Eigen::VectorXd &unconstrained_state,
                                  double) -> void {
  throw(std::runtime_error(
      "set_state_from_unconstrained() not yet implemented"));
}

}  // namespace internal

#endif  // BAYESMIX_HIERARCHIES_LIKELIHOODS_LIKELIHOOD_INTERNAL_H_
