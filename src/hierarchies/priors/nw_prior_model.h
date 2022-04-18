#ifndef BAYESMIX_HIERARCHIES_PRIORS_NW_PRIOR_MODEL_H_
#define BAYESMIX_HIERARCHIES_PRIORS_NW_PRIOR_MODEL_H_

#include <memory>
#include <stan/math/prim.hpp>
#include <stan/math/rev.hpp>
#include <vector>

#include "base_prior_model.h"
#include "hierarchy_prior.pb.h"
#include "hyperparams.h"
#include "src/utils/rng.h"

class NWPriorModel
    : public BasePriorModel<NWPriorModel, State::MultiLS, Hyperparams::NW,
                            bayesmix::NNWPrior> {
 public:
  NWPriorModel() = default;
  ~NWPriorModel() = default;

  double lpdf(const google::protobuf::Message &state_) override;

  State::MultiLS sample(ProtoHypersPtr hier_hypers = nullptr) override;

  void update_hypers(const std::vector<bayesmix::AlgorithmState::ClusterState>
                         &states) override;

  void set_hypers_from_proto(
      const google::protobuf::Message &hypers_) override;

  void write_prec_to_state(const Eigen::MatrixXd &prec_, State::MultiLS *out);

  unsigned int get_dim() const { return dim; };

  std::shared_ptr<bayesmix::AlgorithmState::HierarchyHypers> get_hypers_proto()
      const override;

 protected:
  void initialize_hypers() override;

  unsigned int dim;
};

#endif  // BAYESMIX_HIERARCHIES_PRIORS_NW_PRIOR_MODEL_H_
