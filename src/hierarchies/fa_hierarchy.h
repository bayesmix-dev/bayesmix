#ifndef BAYESMIX_HIERARCHIES_FA_HIERARCHY_H_
#define BAYESMIX_HIERARCHIES_FA_HIERARCHY_H_

#include "base_hierarchy.h"
#include "hierarchy_id.pb.h"
#include "likelihoods/fa_likelihood.h"
#include "priors/fa_prior_model.h"
#include "src/utils/distributions.h"
#include "updaters/fa_updater.h"

class FAHierarchy
    : public BaseHierarchy<FAHierarchy, FALikelihood, FAPriorModel> {
 public:
  FAHierarchy() = default;
  ~FAHierarchy() = default;

  bayesmix::HierarchyId get_id() const override {
    return bayesmix::HierarchyId::FA;
  }

  void set_default_updater() { updater = std::make_shared<FAUpdater>(); }

  void initialize_state() override {
    // Initialize likelihood dimension to prior one
    like->set_dim(prior->get_dim());
    // Get hypers and data dimension
    auto hypers = prior->get_hypers();
    unsigned int dim = like->get_dim();
    // Initialize likelihood state
    State::FA state;
    state.mu = hypers.mutilde;
    state.psi = hypers.beta / (hypers.alpha0 + 1.);
    state.lambda = Eigen::MatrixXd::Zero(dim, hypers.q);
    state.psi_inverse = state.psi.cwiseInverse().asDiagonal();
    state.compute_wood_factors();
    like->set_state(state);
  }
};

inline void set_fa_hyperparams_from_data(FAHierarchy* hier) {
  auto dataset_ptr =
      std::static_pointer_cast<FALikelihood>(hier->get_likelihood())
          ->get_dataset();
  auto hypers =
      std::static_pointer_cast<FAPriorModel>(hier->get_prior())->get_hypers();
  unsigned int dim =
      std::static_pointer_cast<FALikelihood>(hier->get_likelihood())
          ->get_dim();

  // Automatic initialization
  if (dim == 0) {
    hypers.mutilde = dataset_ptr->colwise().mean();
    dim = hypers.mutilde.size();
  }
  if (hypers.beta.size() == 0) {
    Eigen::MatrixXd centered =
        dataset_ptr->rowwise() - dataset_ptr->colwise().mean();
    auto cov_llt =
        ((centered.transpose() * centered) / double(dataset_ptr->rows() - 1.))
            .llt();
    Eigen::MatrixXd precision_matrix(
        cov_llt.solve(Eigen::MatrixXd::Identity(dim, dim)));
    hypers.beta =
        (hypers.alpha0 - 1) * precision_matrix.diagonal().cwiseInverse();
    if (hypers.alpha0 == 1) {
      throw std::invalid_argument(
          "Scale parameter must be different than 1 when automatic "
          "initialization is used");
    }
  }

  bayesmix::AlgorithmState::HierarchyHypers state;
  bayesmix::to_proto(hypers.mutilde,
                     state.mutable_fa_state()->mutable_mutilde());
  bayesmix::to_proto(hypers.beta, state.mutable_fa_state()->mutable_beta());
  state.mutable_fa_state()->set_alpha0(hypers.alpha0);
  state.mutable_fa_state()->set_phi(hypers.phi);
  state.mutable_fa_state()->set_q(hypers.q);
  hier->get_prior()->set_hypers_from_proto(state);
};

#endif  // BAYESMIX_HIERARCHIES_FA_HIERARCHY_H_
