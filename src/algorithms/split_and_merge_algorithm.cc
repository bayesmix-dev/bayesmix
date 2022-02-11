#include "split_and_merge_algorithm.h"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <random>

#include "src/utils/distributions.h"

void SplitAndMergeAlgorithm::read_params_from_proto(
    const bayesmix::AlgorithmParams &params) {
  BaseAlgorithm::read_params_from_proto(params);
  n_restr_gs_updates = params.splitmerge_n_restr_gs_updates();
  n_mh_updates = params.splitmerge_n_mh_updates();
  n_full_gs_updates = params.splitmerge_n_full_gs_updates();
}

void SplitAndMergeAlgorithm::initialize() {
  MarginalAlgorithm::initialize();

  if (mixing->get_id() != bayesmix::MixingId::DP) {
    throw std::invalid_argument(
        "Invalid mixing supplied to Split and Merge, only DP mixing "
        "supported");
  }
}

void SplitAndMergeAlgorithm::print_startup_message() const {
  std::string msg = "Running Split and Merge algorithm with " +
                    bayesmix::HierarchyId_Name(unique_values[0]->get_id()) +
                    " hierarchies, " +
                    bayesmix::MixingId_Name(mixing->get_id()) + " mixing...";
  std::cout << msg << std::endl;
}

void SplitAndMergeAlgorithm::sample_allocations() {
  auto &rng = bayesmix::Rng::Instance().get();

  // MH updates
  for (unsigned int k = 0; k < n_mh_updates; ++k) {
    unsigned int n_data = data.rows();

    // Sample first_random_idx and second_random_idx from the datapoints
    Eigen::VectorXd probas = Eigen::VectorXd::Constant(n_data, 1.0 / n_data);
    unsigned int first_random_idx = bayesmix::categorical_rng(probas, rng, 0);
    probas = Eigen::VectorXd::Constant(n_data - 1, 1.0 / (n_data - 1));
    unsigned int second_random_idx = bayesmix::categorical_rng(probas, rng, 0);
    if (second_random_idx >= first_random_idx) {
      second_random_idx++;
    }

    compute_restricted_gs_data_idx(first_random_idx, second_random_idx);
    compute_restricted_gs_unique_values(first_random_idx, second_random_idx);

    // Restricted GS steps
    for (unsigned int t = 0; t < n_restr_gs_updates; ++t) {
      restricted_gibbs_sampling();
    }

    split_or_merge(first_random_idx, second_random_idx);
  }

  // Full GS updates
  for (unsigned int k = 0; k < n_full_gs_updates; ++k) {
    full_gibbs_sampling();
  }
}

void SplitAndMergeAlgorithm::sample_unique_values() {
  for (auto &un : unique_values) {
    un->sample_full_cond(!update_hierarchy_params());
  }
}

Eigen::VectorXd SplitAndMergeAlgorithm::lpdf_marginal_component(
    std::shared_ptr<AbstractHierarchy> hier, const Eigen::MatrixXd &grid,
    const Eigen::RowVectorXd &covariate) const {
  return hier->prior_pred_lpdf_grid(grid, covariate);
}

void SplitAndMergeAlgorithm::compute_restricted_gs_data_idx(
    const unsigned int first_random_idx,
    const unsigned int second_random_idx) {
  restricted_gs_data_idx = {};

  std::set<int> temp_set_idx = {};
  if (allocations[first_random_idx] == allocations[second_random_idx]) {
    temp_set_idx =
        unique_values[allocations[first_random_idx]]->get_data_idx();
  } else {
    const auto &first_clus =
        unique_values[allocations[first_random_idx]]->get_data_idx();
    const auto &second_clus =
        unique_values[allocations[second_random_idx]]->get_data_idx();
    std::set_union(first_clus.begin(), first_clus.end(), second_clus.begin(),
                   second_clus.end(),
                   std::inserter(temp_set_idx, temp_set_idx.begin()));
  }
  temp_set_idx.erase(first_random_idx);
  temp_set_idx.erase(second_random_idx);

  restricted_gs_data_idx.insert(restricted_gs_data_idx.begin(),
                                temp_set_idx.begin(), temp_set_idx.end());
}

void SplitAndMergeAlgorithm::compute_restricted_gs_unique_values(
    const unsigned int first_random_idx,
    const unsigned int second_random_idx) {
  allocations_restricted_gs.clear();
  allocations_restricted_gs.resize(restricted_gs_data_idx.size());

  restricted_gs_unique_values.clear();
  restricted_gs_unique_values.push_back(unique_values[0]->clone());
  restricted_gs_unique_values.push_back(unique_values[0]->clone());

  auto &rng = bayesmix::Rng::Instance().get();
  Eigen::VectorXd probas(2);
  probas(0) = 0.5;
  probas(1) = 0.5;
  for (size_t k = 0; k < restricted_gs_data_idx.size(); ++k) {
    if (bayesmix::categorical_rng(probas, rng, 0)) {
      allocations_restricted_gs[k] = 1;
      restricted_gs_unique_values[1]->add_datum(
          restricted_gs_data_idx[k], data.row(restricted_gs_data_idx[k]),
          false);
    } else {
      allocations_restricted_gs[k] = 0;
      restricted_gs_unique_values[0]->add_datum(
          restricted_gs_data_idx[k], data.row(restricted_gs_data_idx[k]),
          false);
    }
  }
  /* We update the posterior parameters only at the last insertion to ease
   * computations.
   */
  restricted_gs_unique_values[0]->add_datum(first_random_idx,
                                            data.row(first_random_idx), true);
  restricted_gs_unique_values[1]->add_datum(second_random_idx,
                                            data.row(second_random_idx), true);
}

void SplitAndMergeAlgorithm::compute_log_ratio_like_and_prior(
    const unsigned int first_random_idx, const unsigned int second_random_idx,
    bool split, double &log_ratio_prior_prob, double &log_ratio_likelihoods) {
  /* In the function, we treat the two clusterings as the "divided" one, where
   * first_random_idx and second_random_idx are in different clusters, and as
   * the "united" one, where the two datapoints are in the same cluster.
   * Only in the end we control, via the variable "split", which configuration
   * is the original one and which one is the proposal, in order to compute
   * correctly the returned values.
   */
  std::vector<std::shared_ptr<AbstractHierarchy>> divided_clust_unique_values =
      {unique_values[0]->clone(), unique_values[0]->clone()};
  std::shared_ptr<AbstractHierarchy> united_clust_unique_values(
      unique_values[0]->clone());
  Eigen::VectorXd pred_lpdf_divided_clust = Eigen::VectorXd::Zero(2);
  double pred_lpdf_united_clust = 0;

  /* We treat log_ratio_prior_prob as in the split case, then we change its
   * sign at the end if we are in the merge case.
   */
  log_ratio_prior_prob = mixing->get_mass_new_cluster(
      allocations.size(), true, true, unique_values.size());
  log_ratio_likelihoods = 0;

  std::vector<unsigned int> random_idxs = {first_random_idx,
                                           second_random_idx};

  for (unsigned int clust_idx = 0; clust_idx <= 1; ++clust_idx) {
    pred_lpdf_divided_clust(clust_idx) +=
        divided_clust_unique_values[clust_idx]->prior_pred_lpdf(
            data.row(random_idxs[clust_idx]));
    divided_clust_unique_values[clust_idx]->add_datum(
        random_idxs[clust_idx], data.row(random_idxs[clust_idx]),
        update_hierarchy_params());

    if (united_clust_unique_values->get_card() == 0) {
      pred_lpdf_united_clust += united_clust_unique_values->prior_pred_lpdf(
          data.row(random_idxs[clust_idx]));
    } else {
      pred_lpdf_united_clust +=
          united_clust_unique_values->conditional_pred_lpdf(
              data.row(random_idxs[clust_idx]));
      log_ratio_prior_prob -= mixing->get_mass_existing_cluster(
          allocations.size(), true, true, united_clust_unique_values);
    }
    united_clust_unique_values->add_datum(random_idxs[clust_idx],
                                          data.row(random_idxs[clust_idx]),
                                          update_hierarchy_params());

    std::set<int> set_to_cycle;
    std::set<int>::const_iterator it;
    std::set<int>::const_iterator end;
    if (split) {
      set_to_cycle = restricted_gs_unique_values[clust_idx]->get_data_idx();
      set_to_cycle.erase(random_idxs[clust_idx]);
      it = set_to_cycle.cbegin();
      end = set_to_cycle.cend();
    } else {
      set_to_cycle =
          unique_values[allocations[random_idxs[clust_idx]]]->get_data_idx();
      set_to_cycle.erase(random_idxs[clust_idx]);
      it = set_to_cycle.cbegin();
      end = set_to_cycle.cend();
    }

    for (; it != end; ++it) {
      unsigned int curr_idx = (*it);
      pred_lpdf_divided_clust(clust_idx) +=
          divided_clust_unique_values[clust_idx]->conditional_pred_lpdf(
              data.row(curr_idx));
      log_ratio_prior_prob += mixing->get_mass_existing_cluster(
          allocations.size(), true, true,
          divided_clust_unique_values[clust_idx]);
      divided_clust_unique_values[clust_idx]->add_datum(
          curr_idx, data.row(curr_idx), update_hierarchy_params());

      pred_lpdf_united_clust +=
          united_clust_unique_values->conditional_pred_lpdf(
              data.row(curr_idx));
      log_ratio_prior_prob -= mixing->get_mass_existing_cluster(
          allocations.size(), true, true, united_clust_unique_values);
      united_clust_unique_values->add_datum(curr_idx, data.row(curr_idx),
                                            update_hierarchy_params());
    }
  }

  if (!split) {
    log_ratio_prior_prob = -log_ratio_prior_prob;
  }

  log_ratio_likelihoods =
      pred_lpdf_divided_clust.sum() - pred_lpdf_united_clust;
  if (!split) {
    log_ratio_likelihoods = -log_ratio_likelihoods;
  }
}

void SplitAndMergeAlgorithm::split_or_merge(
    const unsigned int first_random_idx,
    const unsigned int second_random_idx) {
  if (allocations[first_random_idx] == allocations[second_random_idx]) {
    split(first_random_idx, second_random_idx);
  } else {
    merge(first_random_idx, second_random_idx);
  }
}

void SplitAndMergeAlgorithm::split(const unsigned int first_random_idx,
                                   const unsigned int second_random_idx) {
  double log_ratio_transition_prob = -restricted_gibbs_sampling(true);

  double log_ratio_likelihoods = 0;
  double log_ratio_prior_prob = 0;
  compute_log_ratio_like_and_prior(first_random_idx, second_random_idx, true,
                                   log_ratio_prior_prob,
                                   log_ratio_likelihoods);

  const double AcRa =
      std::min(1.0, std::exp(log_ratio_transition_prob + log_ratio_prior_prob +
                             log_ratio_likelihoods));
  if (accepted_proposal(AcRa)) {
    proposal_update_allocations(first_random_idx, second_random_idx, true);
  }
}

void SplitAndMergeAlgorithm::merge(const unsigned int first_random_idx,
                                   const unsigned int second_random_idx) {
  double log_ratio_transition_prob = 0;
  // Fake Gibbs Sampling in order to compute the transition probability
  for (unsigned int k = 0; k < restricted_gs_data_idx.size(); k++) {
    restricted_gs_unique_values[allocations_restricted_gs[k]]->remove_datum(
        restricted_gs_data_idx[k], data.row(restricted_gs_data_idx[k]),
        update_hierarchy_params());

    Eigen::VectorXd logprobas(2);

    if (restricted_gs_unique_values[0]->get_card() >= 1 and
        restricted_gs_unique_values[1]->get_card() >= 1) {
      logprobas(0) = restricted_gs_unique_values[0]->conditional_pred_lpdf(
          data.row(restricted_gs_data_idx[k]));
      logprobas(1) = restricted_gs_unique_values[1]->conditional_pred_lpdf(
          data.row(restricted_gs_data_idx[k]));
    } else {
      if (restricted_gs_unique_values[1]->get_card() == 0) {
        logprobas(1) = restricted_gs_unique_values[1]->prior_pred_lpdf(
            data.row(restricted_gs_data_idx[k]));
        logprobas(0) = restricted_gs_unique_values[0]->conditional_pred_lpdf(
            data.row(restricted_gs_data_idx[k]));
      } else {
        logprobas(0) = restricted_gs_unique_values[0]->prior_pred_lpdf(
            data.row(restricted_gs_data_idx[k]));
        logprobas(1) = restricted_gs_unique_values[1]->conditional_pred_lpdf(
            data.row(restricted_gs_data_idx[k]));
      }
    }
    if (allocations[restricted_gs_data_idx[k]] ==
        allocations[first_random_idx]) {
      allocations_restricted_gs[k] = 0;
      restricted_gs_unique_values[0]->add_datum(
          restricted_gs_data_idx[k], data.row(restricted_gs_data_idx[k]),
          update_hierarchy_params());
      log_ratio_transition_prob += log(stan::math::softmax(logprobas)(0));
    } else {
      allocations_restricted_gs[k] = 1;
      restricted_gs_unique_values[1]->add_datum(
          restricted_gs_data_idx[k], data.row(restricted_gs_data_idx[k]),
          update_hierarchy_params());
      log_ratio_transition_prob += log(stan::math::softmax(logprobas)(1));
    }
  }

  double log_ratio_likelihoods = 0;
  double log_ratio_prior_prob = 0;
  compute_log_ratio_like_and_prior(first_random_idx, second_random_idx, false,
                                   log_ratio_prior_prob,
                                   log_ratio_likelihoods);

  const double AcRa =
      std::min(1.0, std::exp(log_ratio_transition_prob + log_ratio_prior_prob +
                             log_ratio_likelihoods));
  if (accepted_proposal(AcRa)) {
    proposal_update_allocations(first_random_idx, second_random_idx, false);
  }
}

bool SplitAndMergeAlgorithm::accepted_proposal(const double acRa) const {
  std::uniform_real_distribution<> UnifDis(0.0, 1.0);
  return (UnifDis(bayesmix::Rng::Instance().get()) <= acRa);
}

void SplitAndMergeAlgorithm::proposal_update_allocations(
    const unsigned int first_random_idx, const unsigned int second_random_idx,
    const bool split) {
  /* Split case: all the points in the cluster of
   * restricted_gs_unique_values[0] are moved in a new cluster.
   * Merge case: all the points in the cluster of first_random_idx are moved
   * in the cluster of second_random_idx.
   */
  int label_old_cluster;
  int label_new_cluster;
  std::set<int> data_to_move_idx;
  if (split) {
    label_old_cluster = allocations[first_random_idx];
    label_new_cluster = unique_values.size();
    unique_values.push_back(unique_values[0]->clone());
    data_to_move_idx = restricted_gs_unique_values[0]->get_data_idx();
  } else {
    label_old_cluster = allocations[first_random_idx];
    label_new_cluster = allocations[second_random_idx];
    data_to_move_idx = unique_values[label_old_cluster]->get_data_idx();
  }

  auto curr_it = data_to_move_idx.cbegin();
  auto next_it = curr_it;
  next_it++;
  auto end_it = data_to_move_idx.cend();
  for (; curr_it != end_it; next_it++, curr_it++) {
    const unsigned int curr_idx = *curr_it;
    if (next_it == end_it) {
      if (split) {
        unique_values[label_old_cluster]->remove_datum(
            curr_idx, data.row(curr_idx), update_hierarchy_params());
      } else {
        remove_singleton(label_old_cluster);
        /* When a singleton is removed, the other labels are shifted.
         * Therefore, it is necessary to update label_new_cluster accordingly.
         */
        label_new_cluster = allocations[second_random_idx];
      }
      unique_values[label_new_cluster]->add_datum(curr_idx, data.row(curr_idx),
                                                  update_hierarchy_params());
    } else {
      unique_values[label_old_cluster]->remove_datum(
          curr_idx, data.row(curr_idx), false);
      unique_values[label_new_cluster]->add_datum(curr_idx, data.row(curr_idx),
                                                  false);
    }
    allocations[curr_idx] = label_new_cluster;
  }
}

double SplitAndMergeAlgorithm::restricted_gibbs_sampling(
    bool return_log_res_prod /*=false*/) {
  auto &rng = bayesmix::Rng::Instance().get();

  double log_res_prod = 0;

  for (size_t k = 0; k < restricted_gs_data_idx.size(); ++k) {
    restricted_gs_unique_values[allocations_restricted_gs[k]]->remove_datum(
        restricted_gs_data_idx[k], data.row(restricted_gs_data_idx[k]),
        update_hierarchy_params());

    Eigen::VectorXd logprobas(2);
    logprobas(0) = mixing->get_mass_existing_cluster(
        restricted_gs_data_idx.size() + 2 - 1, true, true,
        restricted_gs_unique_values[0]);
    logprobas(0) += restricted_gs_unique_values[0]->conditional_pred_lpdf(
        data.row(restricted_gs_data_idx[k]));
    logprobas(1) = mixing->get_mass_existing_cluster(
        restricted_gs_data_idx.size() + 2 - 1, true, true,
        restricted_gs_unique_values[1]);
    logprobas(1) += restricted_gs_unique_values[1]->conditional_pred_lpdf(
        data.row(restricted_gs_data_idx[k]));

    Eigen::VectorXd probas = stan::math::softmax(logprobas);
    unsigned int c_new = bayesmix::categorical_rng(probas, rng, 0);

    if (return_log_res_prod) {
      log_res_prod += log(probas(c_new));
    }

    allocations_restricted_gs[k] = c_new;
    restricted_gs_unique_values[allocations_restricted_gs[k]]->add_datum(
        restricted_gs_data_idx[k], data.row(restricted_gs_data_idx[k]),
        update_hierarchy_params());
  }

  return log_res_prod;
}

void SplitAndMergeAlgorithm::full_gibbs_sampling() {
  unsigned int n_data = data.rows();
  auto &rng = bayesmix::Rng::Instance().get();
  for (size_t i = 0; i < n_data; ++i) {
    bool singleton = (unique_values[allocations[i]]->get_card() <= 1);
    unsigned int c_old = allocations[i];
    if (singleton) {
      remove_singleton(c_old);
    } else {
      unique_values[c_old]->remove_datum(i, data.row(i),
                                         update_hierarchy_params());
    }
    unsigned int n_clust = unique_values.size();

    Eigen::VectorXd logprobas(n_clust + 1);
    for (size_t j = 0; j < n_clust; ++j) {
      logprobas(j) = mixing->get_mass_existing_cluster(n_data - 1, true, true,
                                                       unique_values[j]);
      logprobas(j) += unique_values[j]->conditional_pred_lpdf(data.row(i));
    }
    logprobas(n_clust) =
        mixing->get_mass_new_cluster(n_data - 1, true, true, n_clust);
    logprobas(n_clust) += unique_values[0]->prior_pred_lpdf(data.row(i));

    unsigned int c_new =
        bayesmix::categorical_rng(stan::math::softmax(logprobas), rng, 0);

    if (c_new == n_clust) {
      std::shared_ptr<AbstractHierarchy> new_unique =
          unique_values[0]->clone();
      new_unique->add_datum(i, data.row(i), update_hierarchy_params());
      new_unique->sample_full_cond(!update_hierarchy_params());
      unique_values.push_back(new_unique);
      allocations[i] = unique_values.size() - 1;
    } else {
      allocations[i] = c_new;
      unique_values[c_new]->add_datum(i, data.row(i),
                                      update_hierarchy_params());
    }
  }
}
