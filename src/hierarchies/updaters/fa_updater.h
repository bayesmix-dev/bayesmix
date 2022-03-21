#ifndef BAYESMIX_HIERARCHIES_UPDATERS_FA_UPDATER_H_
#define BAYESMIX_HIERARCHIES_UPDATERS_FA_UPDATER_H_

#include "abstract_updater.h"
#include "src/hierarchies/likelihoods/fa_likelihood.h"
#include "src/hierarchies/likelihoods/states/includes.h"
#include "src/hierarchies/priors/fa_prior_model.h"
#include "src/hierarchies/priors/hyperparams.h"
#include "src/utils/proto_utils.h"

class FAUpdater : public AbstractUpdater {
 public:
  FAUpdater() = default;
  ~FAUpdater() = default;
  void draw(AbstractLikelihood& like, AbstractPriorModel& prior,
            bool update_params) override;

 protected:
  void sample_eta(State::FA& state, const Hyperparams::FA& hypers,
                  const FALikelihood& like);
  void sample_mu(State::FA& state, const Hyperparams::FA& hypers,
                 const FALikelihood& like);
  void sample_lambda(State::FA& state, const Hyperparams::FA& hypers,
                     const FALikelihood& like);
  void sample_psi(State::FA& state, const Hyperparams::FA& hypers,
                  const FALikelihood& like);
  // void sample_eta(State::FA & state, const Hyperparams::FA & hypers, const
  // Eigen::MatrixXd * dataset_ptr, const std::set<int> & cluster_data_idx);
  // void sample_mu(State::FA & state, const Hyperparams::FA & hypers, const
  // Eigen::VectorXd & data_sum); void sample_lambda(State::FA & state, const
  // Hyperparams::FA & hypers, const Eigen::MatrixXd * dataset_ptr, const
  // std::set<int> & cluster_data_idx, size_t dim); void sample_psi(State::FA &
  // state, const Hyperparams::FA & hypers, const Eigen::MatrixXd *
  // dataset_ptr, const std::set<int> & cluster_data_idx, size_t dim);
};

#endif  // BAYESMIX_HIERARCHIES_UPDATERS_FA_UPDATER_H_
