#ifndef BAYESMIX_HIERARCHIES_PRIORS_FA_PRIOR_MODEL_H_
#define BAYESMIX_HIERARCHIES_PRIORS_FA_PRIOR_MODEL_H_

#include <memory>
#include <stan/math/rev.hpp>
#include <vector>

#include "base_prior_model.h"
#include "hierarchy_prior.pb.h"
#include "hyperparams.h"
#include "src/utils/rng.h"

/**
 * A priormodel for the factor analyzers likelihood, that is
 *
 * \f[
 *    \bm{\mu} &\sim N_p(\tilde{\bm{\mu}}, \psi I) \\
 *    \Lambda &\sim DL(\alpha) \\
 *    \Sigma &= \mathrm{diag}(\sigma^2_1, \ldots, \sigma^2_p) \\
 *    \sigma^2_j &\stackrel{\small\mathrm{iid}}{\sim} InvGamma(a,b)  \quad
 * j=1,...,p \f]
 *
 * Where \f$ DL \f$ is the Dirichlet-Laplace distribution.
 * See Bhattacharya A., Pati D., Pillai N.S., Dunson D.B. (2015).
 * JASA 110(512), 1479–1490 for details.
 */

class FAPriorModel
    : public BasePriorModel<FAPriorModel, State::FA, Hyperparams::FA,
                            bayesmix::FAPrior> {
 public:
  using AbstractPriorModel::ProtoHypers;
  using AbstractPriorModel::ProtoHypersPtr;

  FAPriorModel() = default;
  ~FAPriorModel() = default;

  double lpdf(const google::protobuf::Message &state_) override;

  State::FA sample(ProtoHypersPtr hier_hypers = nullptr) override;

  void update_hypers(const std::vector<bayesmix::AlgorithmState::ClusterState>
                         &states) override;

  void set_hypers_from_proto(
      const google::protobuf::Message &hypers_) override;

  unsigned int get_dim() const { return dim; };

  std::shared_ptr<bayesmix::AlgorithmState::HierarchyHypers> get_hypers_proto()
      const override;

 protected:
  void initialize_hypers() override;

  unsigned int dim;
};

#endif  // BAYESMIX_HIERARCHIES_PRIORS_FA_PRIOR_MODEL_H_
