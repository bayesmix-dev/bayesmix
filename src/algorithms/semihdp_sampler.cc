#include "semihdp_sampler.h"

#include <algorithm>

#include "marginal_state.pb.h"
#include "src/utils/distributions.h"
#include "src/utils/eigen_utils.h"

SemiHdpSampler::SemiHdpSampler(const std::vector<Eigen::MatrixXd>& data,
                               std::shared_ptr<BaseHierarchy> hier,
                               bayesmix::SemiHdpParams params)
    : data(data), params(params) {
  ngroups = data.size();
  n_by_group.resize(ngroups);
  for (int i = 0; i < ngroups; i++) n_by_group[i] = data[i].size();
  G0_master_hierarchy = hier->clone();
  G00_master_hierarchy = hier->clone();
  totalmass_rest = params.totalmass_rest();
  totalmass_hdp = params.totalmass_hdp();
}

void SemiHdpSampler::initialize() {
  auto& rng = bayesmix::Rng::Instance().get();
  dirichlet_concentration = Eigen::VectorXd::Ones(ngroups).array() *
                            params.dirichlet_concentration();

  int INIT_N_CLUS = 5;
  Eigen::VectorXd probas = Eigen::VectorXd::Ones(INIT_N_CLUS);
  probas /= (1.0 * INIT_N_CLUS);
  rest_allocs.resize(ngroups);
  is_used_rest.resize(ngroups);
  table_allocs.resize(ngroups);
  table_to_private.resize(ngroups);
  table_to_shared.resize(ngroups);
  rest_tables.resize(ngroups);
  private_tables.resize(ngroups);
  n_by_table.resize(ngroups);
  rest_tables_pseudo.resize(ngroups);
  n_by_table_pseudo.resize(ngroups);

  for (int l = 0; l < INIT_N_CLUS; l++) {
    std::shared_ptr<BaseHierarchy> hierarchy = G00_master_hierarchy->clone();
    hierarchy->draw();
    shared_tables.push_back(hierarchy);
  }

  for (int i = 0; i < ngroups; i++) {
    rest_allocs[i] = i;
    is_used_rest[i] = true;
    table_allocs[i].resize(n_by_group[i]);

    for (int j = 0; j < INIT_N_CLUS; j++) table_allocs[i][j] = j;
    for (int j = INIT_N_CLUS; j < n_by_group[i]; j++) {
      table_allocs[i][j] = bayesmix::categorical_rng(probas, rng);
    }

    table_to_private[i].resize(2 * INIT_N_CLUS);
    table_to_shared[i].resize(2 * INIT_N_CLUS);

    for (int l = 0; l < INIT_N_CLUS; l++) {
      std::shared_ptr<BaseHierarchy> hierarchy = G0_master_hierarchy->clone();
      hierarchy->draw();
      private_tables[i].push_back(hierarchy);
      rest_tables[i].push_back(hierarchy);
      table_to_private[i][l] = l;
      table_to_shared[i][l] = -1;
    }

    for (int l = 0; l < INIT_N_CLUS; l++) {
      rest_tables[i].push_back(shared_tables[l]);
      table_to_private[i][INIT_N_CLUS + l] = -1;
      table_to_shared[i][INIT_N_CLUS + l] = l;
    }
  }
  cnt_shared_tables = std::vector<int>(shared_tables.size(), 0);

  for (int i = 0; i < ngroups; i++) {
    n_by_table[i] = std::vector<int>(rest_tables[i].size(), 0);
    for (int j = 0; j < n_by_group[i]; j++) {
      n_by_table[i][table_allocs[i][j]] += 1;
    }
  }
}

void SemiHdpSampler::update_unique_vals() {
  // Loop through the theta stars, if its equal to some private_tables, we
  // update it on the fly. Otherwise we store the the data in a vector
  // and then update all the shared_tables

  std::vector<Eigen::MatrixXd> data_by_shared(shared_tables.size());
  for (int h = 0; h < shared_tables.size(); h++) {
    data_by_shared[h] = Eigen::MatrixXd(0, 0);
  }

  for (int r = 0; r < ngroups; r++) {
    std::vector<Eigen::MatrixXd> data_by_theta_star(rest_tables[r].size());
    for (int i = 0; i < ngroups; i++) {
      if (rest_allocs[i] == r) {
        for (int j = 0; j < n_by_group[i]; j++) {
          bayesmix::append_by_row(&data_by_theta_star[table_allocs[i][j]],
                                  data[i].row(j));
        }
      }
    }

    for (int l = 0; l < rest_tables[r].size(); l++) {
      if (table_to_private[r][l] >= 0) {
        if (data_by_theta_star[l].rows() > 0)
          private_tables[r][table_to_private[r][l]]->sample_given_data(
              data_by_theta_star[l]);
        else
          private_tables[r][table_to_private[r][l]]->draw();
        rest_tables[r][l] = private_tables[r][table_to_private[r][l]];
      } else {
        bayesmix::append_by_row(&data_by_shared[table_to_shared[r][l]],
                                data_by_theta_star[l]);
      }
    }
  }

  for (int h = 0; h < shared_tables.size(); h++) {
    if (data_by_shared[h].rows() > 0)
      shared_tables[h]->sample_given_data(data_by_shared[h]);
    else
      shared_tables[h]->draw();
  }

  // reassign stuff to teta_stars
  for (int i = 0; i < ngroups; i++) {
    for (int l = 0; l < rest_tables[i].size(); l++) {
      if (table_to_shared[i][l] >= 0) {
        rest_tables[i][l] = shared_tables[table_to_shared[i][l]];
      }
    }
  }
}

void SemiHdpSampler::update_table_allocs() {
  auto& rng = bayesmix::Rng::Instance().get();
  const double logw = std::log(semihdp_weight);
  const double log1mw = std::log(1 - semihdp_weight);
  const double logalpha = std::log(totalmass_rest);
  const double loggamma = std::log(totalmass_hdp);

  // compute counts
  _count_m();
  _count_n_by_theta_star();

  std::vector<std::vector<int>> log_n_by_table(ngroups);
  for (int r = 0; r < ngroups; r++) {
    for (int n : n_by_table[r]) {
      log_n_by_table[r].push_back(std::log(1.0 * n));
    }
  }
  std::vector<int> log_m;
  for (int l = 0; l < shared_tables.size(); l++) {
    log_m.push_back(std::log(1.0 * cnt_shared_tables[l]));
  }

  double m_sum =
      std::accumulate(cnt_shared_tables.begin(), cnt_shared_tables.end(), 0) +
      1e-20;
  double logmsum = std::log(1.0 * m_sum);

  // cicle through observations
  for (int i = 0; i < ngroups; i++) {
    int r = rest_allocs[i];
    for (int j = 0; j < n_by_group[i]; j++) {
      int s_old = table_allocs[i][j];
      // remove current observation from its allocation
      n_by_table[r][table_allocs[i][j]] -= 1;
      log_n_by_table[r][table_allocs[i][j]] =
          std::log(1.0 * n_by_table[r][table_allocs[i][j]]);
      if (table_to_shared[r][s_old] >= 0) {
        cnt_shared_tables[table_to_shared[r][s_old]] -= 1;
        log_m[table_to_shared[r][s_old]] = std::log(table_to_shared[r][s_old]);
      }

      Eigen::VectorXd probas =
          Eigen::VectorXd::Zero(rest_tables[r].size() + 1);
#pragma omp parallel for
      for (int l = 0; l < rest_tables[r].size(); l++) {
        double log_n = log_n_by_table[r][l];
        probas[l] = log_n + rest_tables[r][l]->like_lpdf(data[i].row(j));
      }

      double margG0 =
          logw + G0_master_hierarchy->marg_lpdf(false, data[i].row(j));

      Eigen::VectorXd hdp_contribs(shared_tables.size() + 1);
#pragma omp parallel for
      for (int h = 0; h < shared_tables.size(); h++) {
        double logm = log_m[h];
        hdp_contribs[h] =
            logm - logmsum + shared_tables[h]->like_lpdf(data[i].row(j));
      }

      hdp_contribs[shared_tables.size()] =
          loggamma - logmsum +
          G00_master_hierarchy->marg_lpdf(false, data[i].row(j));
      double margHDP = log1mw + stan::math::log_sum_exp(hdp_contribs);
      Eigen::VectorXd marg(2);
      marg << margG0, margHDP;
      probas[rest_tables[r].size()] = logalpha + stan::math::log_sum_exp(marg);

      int snew = bayesmix::categorical_rng(stan::math::softmax(probas), rng);
      table_allocs[i][j] = snew;
      if (snew < rest_tables[r].size()) {
        n_by_table[r][snew] += 1;
        log_n_by_table[r][snew] = std::log(1.0 * n_by_table[r][snew]);
      } else {
        n_by_table[r].push_back(1);
        log_n_by_table[r].push_back(0);
        if (stan::math::uniform_rng(0, 1, rng) < semihdp_weight) {
          // sample from G0, add it to rest_tables and private_tables and
          // adjust counts and stuff
          std::shared_ptr<BaseHierarchy> hierarchy =
              G0_master_hierarchy->clone();
          hierarchy->sample_given_data(data[i].row(j));
          private_tables[r].push_back(hierarchy);
          rest_tables[r].push_back(hierarchy);
          table_to_shared[r].push_back(-1);
          table_to_private[r].push_back(private_tables[r].size() - 1);
          n_by_table[r].push_back(1);
          log_n_by_table[r].push_back(1);
        } else {
          // sample from Gtilde
          table_to_private[r].push_back(-1);
          int tnew = bayesmix::categorical_rng(
              stan::math::softmax(hdp_contribs), rng);
          table_to_shared[r].push_back(tnew);
          if (tnew < shared_tables.size()) {
            rest_tables[r].push_back(shared_tables[tnew]);
            cnt_shared_tables[tnew] += 1;
          } else {
            // std::cout << "creating new tau!" << std::endl;
            std::shared_ptr<BaseHierarchy> hierarchy =
                G00_master_hierarchy->clone();
            hierarchy->sample_given_data(data[i].row(j));
            shared_tables.push_back(hierarchy);
            cnt_shared_tables.push_back(1);
            log_m.push_back(0);
            rest_tables[r].push_back(hierarchy);
          }
        }
      }
    }
  }
}

void SemiHdpSampler::update_to_shared() {
  _count_m();

  for (int r = 0; r < ngroups; r++) {
    // aggregate data together
    std::vector<Eigen::MatrixXd> data_by_theta_star;
    for (int l = 0; l < rest_tables[r].size(); l++) {
      data_by_theta_star.push_back(Eigen::MatrixXd::Zero(0, 0));
    }
    for (int i = 0; i < ngroups; i++) {
      if (rest_allocs[i] == r) {
        for (int j = 0; j < n_by_group[i]; j++) {
          data_by_theta_star[table_allocs[i][j]] = bayesmix::append_by_row(
              data_by_theta_star[table_allocs[i][j]], data[i].row(j));
        }
      }
    }
    // compute logprobas
    for (int l = 0; l < rest_tables[r].size(); l++) {
      if ((table_to_shared[r][l] >= 0) && (data_by_theta_star[l].size() > 0)) {
        Eigen::VectorXd probas(shared_tables.size() + 1);
        cnt_shared_tables[table_to_shared[r][l]] -= 1;
        for (int k = 0; k < shared_tables.size(); k++) {
          probas(k) = std::log(cnt_shared_tables[k]) +
                      shared_tables[k]
                          ->like_lpdf_grid(data_by_theta_star[l],
                                           Eigen::MatrixXd(0, 0))
                          .sum();  // TODO
        }
        probas(shared_tables.size()) =
            std::log(totalmass_hdp) +
            G00_master_hierarchy
                ->marg_lpdf_grid(false, Eigen::MatrixXd(0, 0),
                                 data_by_theta_star[l])
                .sum();  // TODO

        probas = stan::math::softmax(probas);

        int newt =
            bayesmix::categorical_rng(probas, bayesmix::Rng::Instance().get());
        if (newt < shared_tables.size()) {
          cnt_shared_tables[newt] += 1;
          table_to_shared[r][l] = newt;
        } else {
          cnt_shared_tables.push_back(1);
          table_to_shared[r][l] = newt;
          std::shared_ptr<BaseHierarchy> hier = G00_master_hierarchy->clone();
          hier->sample_given_data(data_by_theta_star[l]);
          shared_tables.push_back(hier);
        }
      }
    }
  }
}

void SemiHdpSampler::update_rest_allocs() {
  auto& rng = bayesmix::Rng::Instance().get();
  for (int i = 0; i < ngroups; i++) {
    int curr_r = rest_allocs[i];
    int new_r = curr_r;

    if (params.rest_allocs_update() == "full") {
      Eigen::VectorXd probas(ngroups);
// Compute probability for group change
#pragma omp parallel for
      for (int r = 0; r < ngroups; r++)
        probas(r) =
            lpdf_for_group(i, r) + std::log(dirichlet_concentration(r));
      probas = stan::math::softmax(probas);
      new_r = bayesmix::categorical_rng(probas, rng);
    } else {
      int prop_r = curr_r;
      if (params.rest_allocs_update() == "metro_base") {
        std::uniform_int_distribution<int> proposal_dens(0, ngroups - 1);
        prop_r = proposal_dens(rng);
      } else if (params.rest_allocs_update() == "metro_dist") {
        Eigen::VectorXd dists = _compute_mixture_distance(curr_r);
        Eigen::VectorXd proposal_weights = Eigen::VectorXd::Ones(ngroups);
        for (int r = 0; r < ngroups; r++)
          proposal_weights[r] += 0.1 / (0.0001 + dists(r));
        proposal_weights /= proposal_weights.sum();
        prop_r = bayesmix::categorical_rng(proposal_weights, rng);
      }

      double num = lpdf_for_group(i, prop_r) +
                   std::log(dirichlet_concentration(prop_r));
      double den = lpdf_for_group(i, curr_r) +
                   std::log(dirichlet_concentration(curr_r));

      if (std::log(stan::math::uniform_rng(0, 1, rng)) < num - den) {
        new_r = prop_r;
      }
    }

    if (new_r != curr_r) {
      reassign_group(i, new_r, curr_r);
    }
  }
}

void SemiHdpSampler::update_semihdp_weight() {
  auto& rng = bayesmix::Rng::Instance().get();
  // cnt how many from the iodisincratic
  int cnt = 0;
  int tot = 0;
  for (int r = 0; r < ngroups; r++) {
    if (std::find(rest_allocs.begin(), rest_allocs.end(), r) !=
        rest_allocs.end()) {
      for (auto table_to_private : table_to_private[r]) {
        tot += 1;
        cnt += 1.0 * (table_to_private >= 0);
      }
    }
  }
  semihdp_weight =
      stan::math::beta_rng(params.w_prior().shape1() + cnt,
                           params.w_prior().shape2() + tot - cnt, rng);
}

void SemiHdpSampler::update_dirichlet_concentration() {
  Eigen::VectorXd cnts = Eigen::VectorXd::Zero(ngroups);
  for (int i = 0; i < ngroups; i++) {
    cnts[rest_allocs[i]] += 1;
  }

  dirichlet_concentration = stan::math::dirichlet_rng(
      cnts.array() + params.dirichlet_concentration(),
      bayesmix::Rng::Instance().get());
}

void SemiHdpSampler::relabel() {
  for (int r = 0; r < ngroups; r++) {
    // std::cout << "restaurant: " << r << std::endl;
    std::vector<int> groups;
    std::vector<bool> isused(rest_tables[r].size(), false);
    for (int i = 0; i < ngroups; i++) {
      if (rest_allocs[i] == r) groups.push_back(i);
    }

    for (auto i : groups) {
      for (int j = 0; j < n_by_group[i]; j++) {
        isused[table_allocs[i][j]] = true;
      }
    }

    for (int l = isused.size() - 1; l >= 0; l--) {
      if (!isused[l]) {
        // rest_tables
        rest_tables[r].erase(rest_tables[r].begin() + l);

        // maybe theta tilde
        if (table_to_private[r][l] >= 0) {
          for (int k = 0; k < table_to_private[r].size(); k++) {
            if (table_to_private[r][k] > table_to_private[r][l]) {
              table_to_private[r][k] -= 1;
            }
          }

          private_tables[r].erase(private_tables[r].begin() +
                                  table_to_private[r][l]);
        }
        // table_to_private variables
        table_to_private[r].erase(table_to_private[r].begin() + l);

        // table_to_shared variables
        table_to_shared[r].erase(table_to_shared[r].begin() + l);

        for (auto i : groups) {
          // cluster allocations
          for (int j = 0; j < n_by_group[i]; j++) {
            if (table_allocs[i][j] >= l) {
              table_allocs[i][j] -= 1;
            }
          }
          int max_s = *std::max_element(table_allocs[i].begin(),
                                        table_allocs[i].end());
          if (max_s >= rest_tables[r].size()) {
            throw std::invalid_argument(
                "max table_allocs greater than rest_tables size");
          }
        }
      }
    }
  }

  // find the shared_tables which are not allocated and delete them
  std::vector<bool> isused_tau(shared_tables.size(), false);
  for (int i = 0; i < ngroups; i++) {
    for (int l = 0; l < table_to_shared[i].size(); l++) {
      if (table_to_shared[i][l] >= 0) {
        isused_tau[table_to_shared[i][l]] = true;
      }
    }
  }

  for (int l = isused_tau.size() - 1; l >= 0; l--) {
    if (!isused_tau[l]) {
      shared_tables.erase(shared_tables.begin() + l);
      for (int i = 0; i < table_to_shared.size(); i++) {
        for (int k = 0; k < table_to_shared[i].size(); k++) {
          if (table_to_shared[i][k] >= l) {
            table_to_shared[i][k] -= 1;
          }
        }
      }
    }
  }

  _count_n_by_theta_star();
  _count_m();
}

void SemiHdpSampler::sample_pseudo_prior() {
  auto& rng = bayesmix::Rng::Instance().get();
  Eigen::VectorXd probas =
      Eigen::VectorXd::Ones(pseudo_iter).array() / (1.0 * pseudo_iter);
  int iter = bayesmix::categorical_rng(probas, rng);
  for (int r = 0; r < ngroups; r++) {
    bayesmix::MarginalState state;
    pseudoprior_collectors[r].get_state(iter, &state);
    // compute the cardinalities
    int nclus = state.cluster_states_size();
    Eigen::VectorXd cards = Eigen::VectorXd::Zero(nclus);
    for (size_t j = 0; j < state.cluster_allocs_size(); j++) {
      cards[state.cluster_allocs(j)] += 1;
    }

    // generate a multinomial random variable with the weights given by
    // the (normalized) cardinalities of the pseudoprior
    cards = cards.array() / cards.sum();
    double weight = params.pseudo_prior().card_weight();
    Eigen::VectorXd cards_perturb =
        weight * cards.array() +
        (1 - weight) * Eigen::VectorXd::Ones(cards.size()).array() /
            cards.size();
    n_by_table_pseudo[r] =
        stan::math::multinomial_rng(cards_perturb, n_by_group[r], rng);

    rest_tables_pseudo[r].resize(0);
    for (int l = 0; l < state.cluster_states_size(); l++) {
      bayesmix::MarginalState::ClusterState clusval = state.cluster_states(l);
      // perturb(&clusval);
      std::shared_ptr<BaseHierarchy> curr_clus = G0_master_hierarchy->clone();
      curr_clus->set_state_from_proto(clusval);
      rest_tables_pseudo[r].push_back(curr_clus);
    }
  }
}

void SemiHdpSampler::perturb(bayesmix::MarginalState::ClusterState* out) {
  auto& rng = bayesmix::Rng::Instance().get();
  if (out->has_uni_ls_state()) {
    double cnt_shared_tables =
        out->uni_ls_state().mean() +
        stan::math::normal_rng(0, params.pseudo_prior().mean_perturb_sd(),
                               rng);
    double curr_var = out->uni_ls_state().var();
    double var = curr_var +
                 stan::math::uniform_rng(
                     -curr_var / params.pseudo_prior().var_perturb_frac(),
                     curr_var / params.pseudo_prior().var_perturb_frac(), rng);
    out->mutable_uni_ls_state()->set_mean(cnt_shared_tables);
    out->mutable_uni_ls_state()->set_var(var);
  } else {
    throw std::invalid_argument("Case not implemented yet!");
  }
}

double SemiHdpSampler::lpdf_for_group(int i, int r) {
  Eigen::VectorXd lpdf_data(n_by_group[i]);
  Eigen::MatrixXd lpdf_local;
  if (is_used_rest[r]) {
    int nr = std::accumulate(n_by_table[r].begin(), n_by_table[r].end(), 0);
    lpdf_local.resize(n_by_group[i], rest_tables[r].size());
    for (int h = 0; h < rest_tables[r].size(); h++) {
      lpdf_local.col(h) = log(1.0 * n_by_table[r][h] / (totalmass_rest + nr)) +
                          rest_tables[r][h]
                              ->like_lpdf_grid(data[i], Eigen::MatrixXd(0, 0))
                              .array();  // TODO
    }

  } else {
    int nr = std::accumulate(n_by_table_pseudo[r].begin(),
                             n_by_table_pseudo[r].end(), 0);
    lpdf_local.resize(n_by_group[i], rest_tables_pseudo[r].size());
    for (int h = 0; h < rest_tables_pseudo[r].size(); h++) {
      lpdf_local.col(h) =
          log(1.0 * n_by_table_pseudo[r][h] / (totalmass_rest + nr)) +
          rest_tables_pseudo[r][h]
              ->like_lpdf_grid(data[i], Eigen::MatrixXd(0, 0))
              .array();  // TODO
    }
  }
  for (int j = 0; j < n_by_group[i]; j++)
    lpdf_data(j) = stan::math::log_sum_exp(lpdf_local.row(j));

  return lpdf_data.sum();
}

void SemiHdpSampler::reassign_group(int i, int new_r, int old_r) {
  auto& rng = bayesmix::Rng::Instance().get();
  rest_allocs[i] = new_r;
  is_used_rest[old_r] = (std::find(rest_allocs.begin(), rest_allocs.end(),
                                   old_r) != rest_allocs.end());

  if (!is_used_rest[new_r]) {
    // std::cout << "Copying from pseudoprior" << std::endl;
    rest_tables[new_r] = rest_tables_pseudo[new_r];
    n_by_table[new_r] = n_by_table_pseudo[new_r];
    private_tables[new_r] = rest_tables_pseudo[new_r];
    table_to_shared[new_r] = std::vector<int>(rest_tables[new_r].size(), -1);
    table_to_private[new_r].resize(rest_tables[new_r].size());
    for (int l = 0; l < rest_tables[new_r].size(); l++)
      table_to_private[new_r][l] = l;
    is_used_rest[new_r] = true;
  }

  for (int j = 0; j < n_by_group[i]; j++) {
    Eigen::VectorXd probas = Eigen::VectorXd::Zero(rest_tables[new_r].size());
    for (int l = 0; l < rest_tables[new_r].size(); l++) {
      double log_n;
      if (n_by_table[new_r][l] > 0)
        log_n = std::log(1.0 * n_by_table[new_r][l]);
      else
        log_n = 1e-20;

      probas[l] = log_n + rest_tables[new_r][l]->like_lpdf(data[i].row(j));
    }
    table_allocs[i][j] =
        bayesmix::categorical_rng(stan::math::softmax(probas), rng);
  }
}

Eigen::VectorXd SemiHdpSampler::_compute_mixture_distance(int i) {
  std::vector<bayesmix::MarginalState::ClusterState> clus1(
      rest_tables[i].size());
  Eigen::VectorXd weights1(rest_tables[i].size());
  for (int l = 0; l < rest_tables[i].size(); l++) {
    bayesmix::MarginalState::ClusterState clus;
    rest_tables[i][l]->write_state_to_proto(&clus);
    clus1[l] = clus;
    weights1(l) = n_by_table[i][l];
  }
  weights1 = weights1.array() / weights1.sum();
  Eigen::VectorXd dists(ngroups);

  for (int r = 0; r < ngroups; r++) {
    std::vector<bayesmix::MarginalState::ClusterState> clus2(
        rest_tables[r].size());
    Eigen::VectorXd weights2(rest_tables[r].size());
    for (int l = 0; l < rest_tables[r].size(); l++) {
      bayesmix::MarginalState::ClusterState clus;
      rest_tables[r][l]->write_state_to_proto(&clus);
      clus2[l] = clus;
      weights2(l) = n_by_table[r][l];
    }
    weights2 = weights2.array() / weights2.sum();

    dists(r) =
        bayesmix::gaussian_mixture_dist(clus1, weights1, clus2, weights2);
  }
  return dists;
}

void SemiHdpSampler::_count_m() {
  cnt_shared_tables = std::vector<int>(shared_tables.size(), 0);
  for (int r = 0; r < ngroups; r++) {
    for (int l = 0; l < table_to_shared[r].size(); l++)
      if (table_to_shared[r][l] >= 0) {
        cnt_shared_tables[table_to_shared[r][l]] += 1;
      }
  }
}
void SemiHdpSampler::_count_n_by_theta_star() {
  for (int r = 0; r < ngroups; r++) {
    n_by_table[r] = std::vector<int>(rest_tables[r].size(), 0);
    for (int i = 0; i < ngroups; i++) {
      if (rest_allocs[i] == r) {
        for (int j = 0; j < n_by_group[i]; j++) {
          n_by_table[r][table_allocs[i][j]] += 1;
        }
      }
    }
  }
}

bayesmix::SemiHdpState SemiHdpSampler::get_state_as_proto() {
  bayesmix::SemiHdpState state;
  for (int i = 0; i < ngroups; i++) {
    bayesmix::SemiHdpState::RestaurantState curr_restaurant;

    for (int l = 0; l < rest_tables[i].size(); l++) {
      bayesmix::SemiHdpState::ClusterState clusval;
      rest_tables[i][l]->write_state_to_proto(&clusval);
      curr_restaurant.add_theta_stars()->CopyFrom(clusval);
    }
    *curr_restaurant.mutable_n_by_clus() = {n_by_table[i].begin(),
                                            n_by_table[i].end()};
    *curr_restaurant.mutable_table_to_shared() = {table_to_shared[i].begin(),
                                                  table_to_shared[i].end()};
    *curr_restaurant.mutable_table_to_idio() = {table_to_private[i].begin(),
                                                table_to_private[i].end()};

    state.add_restaurants()->CopyFrom(curr_restaurant);

    bayesmix::SemiHdpState::GroupState curr_group;
    *curr_group.mutable_cluster_allocs() = {table_allocs[i].begin(),
                                            table_allocs[i].end()};
    state.add_groups()->CopyFrom(curr_group);

    for (int l = 0; l < shared_tables.size(); l++) {
      bayesmix::SemiHdpState::ClusterState clusval;
      shared_tables[l]->write_state_to_proto(&clusval);
      state.add_taus()->CopyFrom(clusval);
    }

    *state.mutable_c() = {rest_allocs.begin(), rest_allocs.end()};
    state.set_w(semihdp_weight);
  }
  return state;
}

void SemiHdpSampler::print_debug_string() {
  std::cout << "rest_allocs: ";
  for (auto& k : rest_allocs) std::cout << k << ", ";
  std::cout << std::endl << std::endl;

  std::cout << "semihdp_weight: " << semihdp_weight << std::endl;

  for (int r = 0; r < ngroups; r++) {
    std::cout << "**** RESTAURANT: " << r << " *****" << std::endl;
    std::vector<Eigen::MatrixXd> data_by_theta_star(rest_tables[r].size());
    std::cout << "rest_tables[r].size(): " << rest_tables[r].size()
              << std::endl;
    for (int i = 0; i < ngroups; i++) {
      if (rest_allocs[i] == r) {
        for (int j = 0; j < n_by_group[i]; j++) {
          bayesmix::append_by_row(&data_by_theta_star[table_allocs[i][j]],
                                  data[i].row(j));
        }
      }
    }
    for (int l = 0; l < rest_tables[r].size(); l++) {
      std::cout << ", DATA: " << data_by_theta_star[l].transpose()
                << std::endl;
    }
  }
}

void SemiHdpSampler::check() {
  // make sure there are no holes in table_to_private and table_to_shared
  int min_t = INT_MAX;
  for (int i = 0; i < ngroups; i++) {
    std::vector<int> v_sorted(table_to_private[i]);
    std::sort(v_sorted.begin(), v_sorted.end());

    auto it = std::upper_bound(v_sorted.begin(), v_sorted.end(), -1);
    if (it != table_to_private[i].end()) {
      int min_v = *it;
      assert(min_v == 0);
    }

    std::vector<int> t_sorted(table_to_shared[i]);
    std::sort(t_sorted.begin(), t_sorted.end());

    auto it2 = std::upper_bound(t_sorted.begin(), t_sorted.end(), -1);
    if (it2 != table_to_shared[i].end()) {
      int min_t_temp = *it2;
      min_t = std::min(min_t, min_t_temp);
    }
  }
  assert(min_t == 0);

  for (int r = 0; r < ngroups; r++) {
    if (n_by_table[r].size() != rest_tables[r].size()) {
      throw "Error";
    }
  }

  for (int i = 0; i < ngroups; i++) {
    int r = rest_allocs[i];
    // std::cout << "i: " << i << ", r: " << r << std::endl;
    int max_s =
        *std::max_element(table_allocs[i].begin(), table_allocs[i].end());
    if (max_s >= rest_tables[r].size()) {
      throw std::invalid_argument(
          "max table_allocs greater than rest_tables size");
    }
  }
}
