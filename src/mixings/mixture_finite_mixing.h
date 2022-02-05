#ifndef BAYESMIX_MIXINGS_MIXTURE_FINITE_MIXTURES_H_
#define BAYESMIX_MIXINGS_MIXTURE_FINITE_MIXTURES_H_

#include <google/protobuf/message.h>

#include <Eigen/Dense>
#include <memory>
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
  //! @param log        Whether to return logarithm-scale values or not
  //! @param propto     Whether to include normalizing constants or not
  //! @param hier       `Hierarchy` object representing the cluster
  //! @return           Probability value
  double mass_existing_cluster(const unsigned int n, const bool log,
                               const bool propto,
                               std::shared_ptr<AbstractHierarchy> hier,
                               const unsigned int n_clust) const override;

  //! Returns probability mass for a new cluster (for marginal mixings only)
  //! @param n          Total dataset size
  //! @param log        Whether to return logarithm-scale values or not
  //! @param propto     Whether to include normalizing constants or not
  //! @param n_clust    Current number of clusters
  //! @return           Probability value
  double mass_new_cluster(const unsigned int n, const bool log,
                          const bool propto,
                          const unsigned int n_clust) const override;

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
  void init_V_C(unsigned int n) const;

  //! Computes V_n[t] and stores it in V
  void compute_V_t(double t, unsigned int n) const;
};

#endif  // BAYESMIX_MIXINGS_MIXTURE_FINITE_MIXTURES_H_
