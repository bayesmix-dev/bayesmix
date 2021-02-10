#ifndef BAYESMIX_HIERARCHIES_CONJUGATE_HIERARCHY_H_
#define BAYESMIX_HIERARCHIES_CONJUGATE_HIERARCHY_H_

#include "base_hierarchy.h"

template <class Derived, typename State, typename Hyperparams, typename Prior>
class ConjugateHierarchy
    : public BaseHierarchy<Derived, State, Hyperparams, Prior> {
 public:
  ~ConjugateHierarchy() = default;
  ConjugateHierarchy() = default;

  using BaseHierarchy<Derived, State, Hyperparams, Prior>::posterior_hypers;
  using BaseHierarchy<Derived, State, Hyperparams, Prior>::state;

  virtual Hyperparams get_posterior_parameters() = 0;

  void save_posterior_hypers() {
    posterior_hypers =
        static_cast<Derived *>(this)->get_posterior_parameters();
  }

  //! Generates new values for state from the centering posterior distribution
  void sample_full_cond(bool update_params = true) override {
    Hyperparams params =
        update_params
            ? static_cast<Derived *>(this)->get_posterior_parameters()
            : posterior_hypers;
    state = static_cast<Derived *>(this)->draw(params);
  }
};

#endif