#ifndef BAYESMIX_MIXINGS_MIXTURE_FINITE_MIXTURES_H_
#define BAYESMIX_MIXINGS_MIXTURE_FINITE_MIXTURES_H_

#include <google/protobuf/message.h>

#include <memory>
#include <stan/math/rev.hpp>
#include <vector>

#include "base_mixing.h"
#include "mixing_id.pb.h"
#include "mixing_prior.pb.h"
#include "src/hierarchies/abstract_hierarchy.h"

namespace Mixture_Finite {
struct State {
  double lambda, gamma;
};
};  // namespace Mixture_Finite

//! Class that represents the Mixture of Finite Mixtures (MFM) [1]
//! The basic idea is to take usual finite mixture model with Dirichlet weights
//! and put a prior (Poisson) on the number of components. The EPPF induced by
//! MFM depends on a Dirichlet parameter 'gamma' and number \f$V_n(t)\f$, where
//! \f$V_n(t)\f$ depends on the Poisson rate parameter 'lambda'.
//! \f[
//!      V_n(t) = \sum_{k=1}^{\infty} ( k_(t)p_K(k) / (\gamma*k)^(n) )
//! \f]
//! Given a clustering of n elements into k clusters, each with cardinality
//! \f$n_j, j=1, ..., k\f$, the EPPF of the MFM gives the following
//! probabilities for the cluster membership of the (n+1)-th observation: \f[
//!      p(\text{j-th cluster} | ...) &= (n_j + \gamma) / D \\
//!      p(\text{k+1-th cluster} | ...) &= V[k+1]/V[k] \gamma / D \\
//!      D &= n_j + \gamma / (n + \gamma * (k + V[k+1]/V[k]))
//! \f]
//! For numerical reasons each value of V is multiplied with a constant  C
//! computed as the first term of the series of V_n[0].
//! For more information about the class, please refer instead to base
//! classes, `AbstractMixing` and `BaseMixing`.
//! [1] "Mixture Models with a Prior on the Number of Components", J.W.Miller
//! and M.T.Harrison, 2015, arXiv:1502.06241v1

class MixtureFiniteMixing
    : public BaseMixing<MixtureFiniteMixing, Mixture_Finite::State,
                        bayesmix::MFMPrior> {
 public:
  MixtureFiniteMixing() = default;
  ~MixtureFiniteMixing() = default;

  //! Performs conditional update of state, given allocations and unique values
  //! @param unique_values  A vector of (pointers to) Hierarchy objects
  //! @param allocations    A vector of allocations label
  void update_state(
      const std::vector<std::shared_ptr<AbstractHierarchy>> &unique_values,
      const std::vector<unsigned int> &allocations) override;

  //! Read and set state values from a given Protobuf message
  void set_state_from_proto(const google::protobuf::Message &state_) override;

  //! Writes current state to a Protobuf message and return a shared_ptr
  //! New hierarchies have to first modify the field 'oneof val' in the
  //! MixingState message by adding the appropriate type
  std::shared_ptr<bayesmix::MixingState> get_state_proto() const override;

  //! Returns the Protobuf ID associated to this class
  bayesmix::MixingId get_id() const override {
    return bayesmix::MixingId::MFM;
  }

  //! Returns whether the mixing is conditional or marginal
  bool is_conditional() const override { return false; }

 protected:
  //! Returns probability mass for an old cluster (for marginal mixings only)
  //! @param n          Total dataset size
  //! @param n_clust    Number of clusters
  //! @param log        Whether to return logarithm-scale values or not
  //! @param propto     Whether to include normalizing constants or not
  //! @param hier       `Hierarchy` object representing the cluster
  //! @return           Probability value
  double mass_existing_cluster(
      const unsigned int n, const unsigned int n_clust, const bool log,
      const bool propto,
      const std::shared_ptr<AbstractHierarchy> hier) const override;

  //! Returns probability mass for a new cluster (for marginal mixings only)
  //! @param n          Total dataset size
  //! @param log        Whether to return logarithm-scale values or not
  //! @param propto     Whether to include normalizing constants or not
  //! @param n_clust    Current number of clusters
  //! @return           Probability value
  double mass_new_cluster(const unsigned int n, const unsigned int n_clust,
                          const bool log, const bool propto) const override;

  //! Initializes state parameters to appropriate values
  void initialize_state() override;

 protected:
  //! Vector V needed to cumpute the probabilities of a new or existing cluster
  mutable std::vector<double> V{};

  //! Constant that is multipied by each value of V for numerical reasons, it
  //! is computed as the first term of the series of V_n[0].
  mutable double C;

  //! Checks if V has been initialized by init_V_C
  mutable bool V_C_are_initialized = false;

  //! Initializes V to a vector of -1 of length n+1 and computes and assigns C
  void init_V_and_C(const unsigned int n) const;

  //! Computes V_n[t] and stores it in V
  void compute_V_t(const double t, const unsigned int n) const;

  //! Gets V_n[t] or computes and stores it if it has not been computed before
  double get_V_t(const double t, const unsigned int n) const;
};

#endif  // BAYESMIX_MIXINGS_MIXTURE_FINITE_MIXTURES_H_
