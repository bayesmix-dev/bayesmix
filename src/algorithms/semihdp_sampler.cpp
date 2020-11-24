#include "semihdp_sampler.hpp"

#include <proto/cpp/marginal_state.pb.h>

#include <algorithm>
#include <src/utils/eigen_utils.hpp>

#include "../utils/distributions.hpp"

SemiHdpSampler::SemiHdpSampler(const std::vector<MatrixXd>& data,
                               std::string c_update)
    : data(data), c_update(c_update) {
  ngroups = data.size();
  n_by_group.resize(ngroups);
  for (int i = 0; i < ngroups; i++) n_by_group[i] = data[i].size();
}

void SemiHdpSampler::initialize() {
  auto& rng = bayesmix::Rng::Instance().get();

  // compute overall mean
  double mu0 = std::accumulate(
      data.begin(), data.end(), 0,
      [&](int curr, const MatrixXd dat) { return curr + dat.sum(); });
  mu0 /= (1.0 * std::accumulate(n_by_group.begin(), n_by_group.end(), 0));
  omega = VectorXd::Ones(ngroups).array();

  master_hierarchy.set_mu0(mu0);
  master_hierarchy.set_lambda0(0.1);
  master_hierarchy.set_alpha0(2);
  master_hierarchy.set_beta0(2);

  int INIT_N_CLUS = 5;
  VectorXd probas = VectorXd::Ones(INIT_N_CLUS);
  probas /= (1.0 * INIT_N_CLUS);
  c.resize(ngroups);
  is_used_c.resize(ngroups);
  s.resize(ngroups);
  v.resize(ngroups);
  t.resize(ngroups);
  theta_star.resize(ngroups);
  theta_tilde.resize(ngroups);
  n_by_theta_star.resize(ngroups);
  theta_star_pseudo.resize(ngroups);
  n_by_theta_star_pseudo.resize(ngroups);

  for (int l = 0; l < INIT_N_CLUS; l++) {
    NNIGHierarchy hierarchy = master_hierarchy;
    hierarchy.draw();
    taus.push_back(hierarchy);
  }

  for (int i = 0; i < ngroups; i++) {
    c[i] = i;
    is_used_c[i] = true;
    s[i].resize(n_by_group[i]);

    for (int j = 0; j < INIT_N_CLUS; j++) s[i][j] = j;
    for (int j = INIT_N_CLUS; j < n_by_group[i]; j++) {
      s[i][j] = bayesmix::categorical_rng(probas, rng);
    }

    v[i].resize(2 * INIT_N_CLUS);
    t[i].resize(2 * INIT_N_CLUS);

    for (int l = 0; l < INIT_N_CLUS; l++) {
      NNIGHierarchy hierarchy = master_hierarchy;
      hierarchy.draw();
      theta_tilde[i].push_back(hierarchy);
      theta_star[i].push_back(hierarchy);
      v[i][l] = l;
      t[i][l] = -1;
    }

    for (int l = 0; l < INIT_N_CLUS; l++) {
      theta_star[i].push_back(taus[l]);
      v[i][INIT_N_CLUS + l] = -1;
      t[i][INIT_N_CLUS + l] = l;
    }
  }
  m = std::vector<int>(taus.size(), 0);

  for (int i = 0; i < ngroups; i++) {
    n_by_theta_star[i] = std::vector<int>(theta_star[i].size(), 0);
    for (int j = 0; j < n_by_group[i]; j++) {
      n_by_theta_star[i][s[i][j]] += 1;
    }
  }
}

void SemiHdpSampler::update_unique_vals() {
  // Loop through the theta stars, if its equal to some theta_tilde, we
  // update it on the fly. Otherwise we store the the data in a vector
  // and then update all the taus

  std::vector<MatrixXd> data_by_tau(taus.size());
  for (int h = 0; h < taus.size(); h++) {
    data_by_tau[h] = Eigen::MatrixXd(0, 1);
  }

  for (int r = 0; r < ngroups; r++) {
    std::vector<MatrixXd> data_by_theta_star(theta_star[r].size());
    // std::cout << "theta_star[r].size() " << theta_star[r].size() <<
    // std::endl;
    for (int i = 0; i < ngroups; i++) {
      if (c[i] == r) {
        for (int j = 0; j < n_by_group[i]; j++) {
          // std::cout << "s[i][j]: " << s[i][j] << std::endl;
          bayesmix::append_by_row(&data_by_theta_star[s[i][j]],
                                  data[i].row(j));
        }
      }
    }

    for (int l = 0; l < theta_star[r].size(); l++) {
      if (v[r][l] >= 0) {
        if (data_by_theta_star[l].rows() > 0)
          theta_tilde[r][v[r][l]].sample_given_data(data_by_theta_star[l]);
        else
          theta_tilde[r][v[r][l]].draw();
        theta_star[r][l] = theta_tilde[r][v[r][l]];
      } else {
        bayesmix::append_by_row(&data_by_tau[t[r][l]], data_by_theta_star[l]);
      }
    }
  }

  for (int h = 0; h < taus.size(); h++) {
    if (data_by_tau[h].rows() > 0)
      taus[h].sample_given_data(data_by_tau[h]);
    else
      taus[h].draw();
  }

  // reassign stuff to teta_stars
  for (int i = 0; i < ngroups; i++) {
    for (int l = 0; l < theta_star[i].size(); l++) {
      if (t[i][l] >= 0) {
        theta_star[i][l] = taus[t[i][l]];
      }
    }
  }
  // std::cout << "update_unique_vals DONE" << std::endl;
}

void SemiHdpSampler::update_s() {
  // std::cout << "update_s" << std::endl;

  auto& rng = bayesmix::Rng::Instance().get();
  const double logw = std::log(w);
  const double log1mw = std::log(1 - w);
  const double logalpha = std::log(alpha);
  const double loggamma = std::log(gamma);

  // compute counts
  m = std::vector<int>(taus.size(), 0);
  for (int l = 0; l < taus.size(); l++)
    for (int r = 0; r < ngroups; r++) {
      for (int l = 0; l < t[r].size(); l++)
        if (t[r][l] >= 0) {
          // std::cout << "t[r][l]: " << t[r][l] << std::endl;
          m[t[r][l]] += 1;
        }

      n_by_theta_star[r] = std::vector<int>(theta_star[r].size(), 0);
#pragma omp parallel for
      for (int i = 0; i < ngroups; i++) {
        if (c[i] == r) {
          for (int j = 0; j < n_by_group[i]; j++) {
            n_by_theta_star[r][s[i][j]] += 1;
          }
        }
      }
    }
  std::vector<std::vector<int>> log_n_by_theta_star(ngroups);
  for (int r = 0; r < ngroups; r++) {
    for (int n : n_by_theta_star[r]) {
      if (n == 0)
        log_n_by_theta_star[r].push_back(1e-20);
      else
        log_n_by_theta_star[r].push_back(std::log(1.0 * n));
    }
  }
  std::vector<int> log_m;
  for (int l = 0; l < taus.size(); l++) {
    if (m[l] == 0)
      log_m.push_back(1e-20);
    else
      log_m.push_back(std::log(1.0 * m[l]));
  }

  double m_sum = std::accumulate(m.begin(), m.end(), 0) + 1e-20;
  double logmsum = std::log(1.0 * m_sum);

  // cicle through observations
  for (int i = 0; i < ngroups; i++) {
    int r = c[i];
    for (int j = 0; j < n_by_group[i]; j++) {
      int s_old = s[i][j];
      // remove current observation from its allocation
      n_by_theta_star[r][s[i][j]] -= 1;
      log_n_by_theta_star[r][s[i][j]] =
          std::log(1.0 * n_by_theta_star[r][s[i][j]]);
      if (t[r][s_old] >= 0) {
        m[t[r][s_old]] -= 1;
        log_m[t[r][s_old]] = std::log(t[r][s_old]);
      }

      VectorXd probas = VectorXd::Zero(theta_star[r].size() + 1);
#pragma omp parallel for
      for (int l = 0; l < theta_star[r].size(); l++) {
        double log_n = log_n_by_theta_star[r][l];
        // if (n_by_theta_star[r][l] > 0)
        //   log_n = std::log(1.0 * n_by_theta_star[r][l]);
        // else
        //   log_n = 1e-20;

        probas[l] = log_n + theta_star[r][l].like_lpdf(data[i].row(j));
      }

      double margG0 = logw + master_hierarchy.marg_lpdf(data[i].row(j));

      VectorXd hdp_contribs(taus.size() + 1);
#pragma omp parallel for
      for (int h = 0; h < taus.size(); h++) {
        double logm = log_m[h];
        // if (m[h] > 0)
        //   logm = std::log(1.0 * m[h]);
        // else
        //   logm = 1e-20;
        hdp_contribs[h] = logm - logmsum + taus[h].like_lpdf(data[i].row(j));
      }

      hdp_contribs[taus.size()] =
          loggamma - logmsum + master_hierarchy.marg_lpdf(data[i].row(j));
      double margHDP = log1mw + stan::math::log_sum_exp(hdp_contribs);
      VectorXd marg(2);
      marg << margG0, margHDP;
      probas[theta_star[r].size()] = logalpha + stan::math::log_sum_exp(marg);

      int snew = bayesmix::categorical_rng(stan::math::softmax(probas), rng);
      s[i][j] = snew;
      if (snew < theta_star[r].size()) {
        n_by_theta_star[r][snew] += 1;
        log_n_by_theta_star[r][snew] =
            std::log(1.0 * n_by_theta_star[r][snew]);
      } else {
        // std::cout << "creating new theta_star" << std::endl;
        n_by_theta_star[r].push_back(1);
        log_n_by_theta_star[r].push_back(0);
        if (stan::math::uniform_rng(0, 1, rng) < w) {
          // sample from G0, add it to theta_star and theta_tilde and adjust
          // counts and stuff
          NNIGHierarchy hierarchy = master_hierarchy;
          hierarchy.sample_given_data(data[i].row(j));
          theta_tilde[r].push_back(hierarchy);
          theta_star[r].push_back(hierarchy);
          t[r].push_back(-1);
          v[r].push_back(theta_tilde[r].size() - 1);
          n_by_theta_star[r].push_back(1);
          log_n_by_theta_star[r].push_back(1);
        } else {
          // sample from Gtilde
          v[r].push_back(-1);
          int tnew = bayesmix::categorical_rng(
              stan::math::softmax(hdp_contribs), rng);
          t[r].push_back(tnew);
          if (tnew < taus.size()) {
            theta_star[r].push_back(taus[tnew]);
            m[tnew] += 1;
          } else {
            // std::cout << "creating new tau!" << std::endl;
            NNIGHierarchy hierarchy = master_hierarchy;
            hierarchy.sample_given_data(data[i].row(j));
            taus.push_back(hierarchy);
            m.push_back(1);
            log_m.push_back(0);
            theta_star[r].push_back(hierarchy);
          }
        }
      }
    }
  }
  // std::cout << "update_s DONE" << std::endl;
}

void SemiHdpSampler::update_t() {}

void SemiHdpSampler::update_c() {
  // std::cout << "update_c" << std::endl;

  auto& rng = bayesmix::Rng::Instance().get();
  for (int i = 0; i < ngroups; i++) {
    int curr_r = c[i];
    int new_r = curr_r;

    if (c_update == "full") {
      VectorXd probas(ngroups);
// Compute probability for group change
#pragma omp parallel for
      for (int r = 0; r < ngroups; r++)
        probas(r) = lpdf_for_group(i, r) + std::log(omega(r));

      std::cout << "group: " << i << ", probas: " << probas.transpose() << std::endl;
      // sample new group
      probas = stan::math::softmax(probas);
      new_r = bayesmix::categorical_rng(probas, rng);
    } else if (c_update == "metro_base") {
      std::uniform_int_distribution<int> proposal_dens(0, ngroups - 1);
      int prop_r = proposal_dens(rng);
      double num = lpdf_for_group(i, prop_r) + std::log(omega(prop_r));
      double den = lpdf_for_group(i, curr_r) + std::log(omega(curr_r));
      if (std::log(stan::math::uniform_rng(0, 1, rng)) < num - den) {
        new_r = prop_r;
      }
    } else if (c_update == "metro_dist") {
      std::vector<bayesmix::MarginalState::ClusterVal> clus1(
          theta_star[curr_r].size());
      VectorXd weights1(theta_star[curr_r].size());
      for (int l = 0; l < theta_star[curr_r].size(); l++) {
        bayesmix::MarginalState::ClusterVal clus;
        theta_star[curr_r][l].write_state_to_proto(&clus);
        clus1[l] = clus;
        weights1(l) = n_by_theta_star[curr_r][l];
      }
      weights1 = weights1.array() / weights1.sum();

      VectorXd dists(ngroups);

      for (int r = 0; r < ngroups; r++) {
        std::vector<bayesmix::MarginalState::ClusterVal> clus2(
            theta_star[r].size());
        VectorXd weights2(theta_star[r].size());
        for (int l = 0; l < theta_star[r].size(); l++) {
          bayesmix::MarginalState::ClusterVal clus;
          theta_star[r][l].write_state_to_proto(&clus);
          clus2[l] = clus;
          weights2(l) = n_by_theta_star[r][l];
        }
        weights2 = weights2.array() / weights2.sum();

        dists(r) =
            bayesmix::gaussian_mixture_dist(clus1, weights1, clus2, weights2);
      }

      VectorXd proposal_weights = VectorXd::Ones(ngroups);
      for (int r = 0; r < ngroups; r++)
        proposal_weights[r] += 0.1 / (0.0001 + dists(r));
      proposal_weights /= proposal_weights.sum();

      int prop_r = bayesmix::categorical_rng(proposal_weights, rng);
      double num = lpdf_for_group(i, prop_r) + std::log(omega(prop_r));
      double den = lpdf_for_group(i, curr_r) + std::log(omega(curr_r));
      // if (i == 0) {
      //   std::cout << "dists: " << dists.transpose() << std::endl;

      //   std::cout << "proposal_weights: " << proposal_weights.transpose()
      //             << std::endl;
      //   std::cout << "curr_r: " << curr_r << ", prop_r: " << prop_r
      //             << ", num: " << num << ", den: " << den
      //             << ", arate: " << std::exp(num - den) << std::endl;
      // }

      if (std::log(stan::math::uniform_rng(0, 1, rng)) < num - den) {
        new_r = prop_r;
      }
    }

    if (new_r != curr_r) {
      reassign_group(i, new_r, curr_r);
    }
  }
}

void SemiHdpSampler::update_w() {
  // std::cout << "update_w" << std::endl;

  auto& rng = bayesmix::Rng::Instance().get();
  // cnt how many from the iodisincratic
  int cnt = 0;
  int tot = 0;
  for (int r = 0; r < ngroups; r++) {
    if (std::find(c.begin(), c.end(), r) != c.end()) {
      for (auto v : v[r]) {
        tot += 1;
        cnt += 1.0 * (v >= 0);
      }
    }
  }
  w = stan::math::beta_rng(a_w + cnt, b_w + tot - cnt, rng);
  // std::cout << "update_w DONE" << std::endl;
}

void SemiHdpSampler::update_omega() {
  VectorXd cnts = VectorXd::Zero(ngroups);
  for (int i = 0; i < ngroups; i++) {
    cnts[c[i]] += 1;
  }

  omega = stan::math::dirichlet_rng(cnts.array() + 1.0 / ngroups,
                                    bayesmix::Rng::Instance().get());
}

void SemiHdpSampler::relabel() {
  for (int r = 0; r < ngroups; r++) {
    // std::cout << "restaurant: " << r << std::endl;
    std::vector<int> groups;
    std::vector<bool> isused(theta_star[r].size(), false);
    for (int i = 0; i < ngroups; i++) {
      if (c[i] == r) groups.push_back(i);
    }

    for (auto i : groups) {
      for (int j = 0; j < n_by_group[i]; j++) {
        isused[s[i][j]] = true;
      }
    }

    for (int l = isused.size() - 1; l >= 0; l--) {
      if (!isused[l]) {
        // theta_star
        theta_star[r].erase(theta_star[r].begin() + l);

        // maybe theta tilde
        if (v[r][l] >= 0) {
          for (int k = 0; k < v[r].size(); k++) {
            if (v[r][k] > v[r][l]) {
              v[r][k] -= 1;
            }
          }

          theta_tilde[r].erase(theta_tilde[r].begin() + v[r][l]);
        }
        // v variables
        v[r].erase(v[r].begin() + l);

        // t variables
        t[r].erase(t[r].begin() + l);

        for (auto i : groups) {
          // cluster allocations
          for (int j = 0; j < n_by_group[i]; j++) {
            if (s[i][j] >= l) {
              s[i][j] -= 1;
            }
          }
          int max_s = *std::max_element(s[i].begin(), s[i].end());
          if (max_s >= theta_star[r].size()) {
            throw std::invalid_argument("max s greater than theta_star size");
          }
        }
      }
    }
  }

  // find the taus which are not allocated and delete them
  std::vector<bool> isused_tau(taus.size(), false);
  for (int i = 0; i < ngroups; i++) {
    for (int l = 0; l < t[i].size(); l++) {
      if (t[i][l] >= 0) {
        isused_tau[t[i][l]] = true;
      }
    }
  }

  for (int l = isused_tau.size() - 1; l >= 0; l--) {
    if (!isused_tau[l]) {
      taus.erase(taus.begin() + l);
      for (int i = 0; i < t.size(); i++) {
        for (int k = 0; k < t[i].size(); k++) {
          if (t[i][k] >= l) {
            t[i][k] -= 1;
          }
        }
      }
    }
  }

  for (int r = 0; r < ngroups; r++) {
    n_by_theta_star[r] = std::vector<int>(theta_star[r].size(), 0);
    for (int i = 0; i < ngroups; i++) {
      if (c[i] == r) {
        for (int j = 0; j < n_by_group[i]; j++) {
          n_by_theta_star[r][s[i][j]] += 1;
        }
      }
    }
  }
}

void SemiHdpSampler::sample_pseudo_prior() {
  // std::cout << "sample_pseudo_prior" << std::endl;

  auto& rng = bayesmix::Rng::Instance().get();
  VectorXd probas = VectorXd::Ones(pseudo_iter).array() / (1.0 * pseudo_iter);
  int iter = bayesmix::categorical_rng(probas, rng);
  for (int r = 0; r < ngroups; r++) {
    MarginalState state = pseudoprior_collectors[r].get_state(iter);
    // compute the cardinalities
    int nclus = state.cluster_vals_size();
    VectorXd cards = VectorXd::Zero(nclus);
    for (size_t j = 0; j < state.cluster_allocs_size(); j++) {
      cards[state.cluster_allocs(j)] += 1;
    }

    // generate a multinomial random variable with the weights given by
    // the (normalized) cardinalities of the pseudoprior
    // std::cout << "cards before: " << cards.transpose() << std::endl;

    cards = cards.array() / cards.sum();
    VectorXd cards_perturb =
        0.5 * cards.array() +
        0.5 * VectorXd::Ones(cards.size()).array() / cards.size();
    n_by_theta_star_pseudo[r] =
        stan::math::multinomial_rng(cards_perturb, n_by_group[r], rng);

    // n_by_theta_star_pseudo[r] = {cards.data(), cards.data() + cards.size()};

    // std::cout << "cards perturb: ";
    // for (int k: n_by_theta_star_pseudo[r]) std::cout << k << ", ";
    // std::cout << std::endl;

    theta_star_pseudo[r].resize(0);
    for (int l = 0; l < state.cluster_vals_size(); l++) {
      MarginalState::ClusterVal clusval = state.cluster_vals(l);
      // perturb(&clusval);
      NNIGHierarchy curr_clus = master_hierarchy;
      curr_clus.set_state(clusval);
      theta_star_pseudo[r].push_back(curr_clus);
    }
  }
}

void SemiHdpSampler::perturb(MarginalState::ClusterVal* out) {
  auto& rng = bayesmix::Rng::Instance().get();
  if (out->has_univ_ls_state()) {
    double m =
        out->univ_ls_state().mean() + stan::math::normal_rng(0, 1.5, rng);
    double curr_sd = out->univ_ls_state().sd();
    double sd =
        curr_sd + stan::math::uniform_rng(-curr_sd / 4, curr_sd / 4, rng);
    out->mutable_univ_ls_state()->set_mean(m);
    out->mutable_univ_ls_state()->set_sd(sd);
  } else {
    throw std::invalid_argument("Case not implemented yet!");
  }
}

double SemiHdpSampler::semihdp_marg_lpdf(const VectorXd& datum) {}

double SemiHdpSampler::lpdf_for_group(int i, int r) {
  VectorXd lpdf_data(n_by_group[i]);
  MatrixXd lpdf_local;
  if (is_used_c[r]) {
    int nr = std::accumulate(n_by_theta_star[r].begin(),
                             n_by_theta_star[r].end(), 0);
    lpdf_local.resize(n_by_group[i], theta_star[r].size());
    for (int h = 0; h < theta_star[r].size(); h++) {
      lpdf_local.col(h) = log(1.0 * n_by_theta_star[r][h] / (alpha + nr)) +
                          theta_star[r][h].like_lpdf_grid(data[i]).array();
    }

  } else {
    int nr = std::accumulate(n_by_theta_star_pseudo[r].begin(),
                             n_by_theta_star_pseudo[r].end(), 0);
    lpdf_local.resize(n_by_group[i], theta_star_pseudo[r].size());
    for (int h = 0; h < theta_star_pseudo[r].size(); h++) {
      lpdf_local.col(h) =
          log(1.0 * n_by_theta_star_pseudo[r][h] / (alpha + nr)) +
          theta_star_pseudo[r][h].like_lpdf_grid(data[i]).array();
    }
  }
  for (int j = 0; j < n_by_group[i]; j++)
    lpdf_data(j) = stan::math::log_sum_exp(lpdf_local.row(j));

  return lpdf_data.sum();
}

void SemiHdpSampler::reassign_group(int i, int new_r, int old_r) {
  // std::cout << "changing c, group: " << i << ", old_r: " << old_r
  //           << ", new_r: " << new_r << std::endl;
  auto& rng = bayesmix::Rng::Instance().get();
  c[i] = new_r;
  is_used_c[old_r] = (std::find(c.begin(), c.end(), old_r) != c.end());

  if (!is_used_c[new_r]) {
    // std::cout << "Copying from pseudoprior" << std::endl;
    theta_star[new_r] = theta_star_pseudo[new_r];
    n_by_theta_star[new_r] = n_by_theta_star_pseudo[new_r];
    theta_tilde[new_r] = theta_star_pseudo[new_r];
    t[new_r] = std::vector<int>(theta_star[new_r].size(), -1);
    v[new_r].resize(theta_star[new_r].size());
    for (int l = 0; l < theta_star[new_r].size(); l++) v[new_r][l] = l;
    is_used_c[new_r] = true;
  }

  for (int j = 0; j < n_by_group[i]; j++) {
    VectorXd probas = VectorXd::Zero(theta_star[new_r].size());
    for (int l = 0; l < theta_star[new_r].size(); l++) {
      double log_n;
      if (n_by_theta_star[new_r][l] > 0)
        log_n = std::log(1.0 * n_by_theta_star[new_r][l]);
      else
        log_n = 1e-20;

      probas[l] = log_n + theta_star[new_r][l].like_lpdf(data[i].row(j));
    }
    s[i][j] = bayesmix::categorical_rng(stan::math::softmax(probas), rng);
  }
  // std::cout << "done" << std::endl;
}

bayesmix::SemiHdpState SemiHdpSampler::get_state_as_proto() {
  bayesmix::SemiHdpState state;
  for (int i = 0; i < ngroups; i++) {
    bayesmix::SemiHdpState::RestaurantState curr_restaurant;

    for (int l = 0; l < theta_star[i].size(); l++) {
      bayesmix::ClusterVal clusval;
      theta_star[i][l].write_state_to_proto(&clusval);
      curr_restaurant.add_theta_stars()->CopyFrom(clusval);
    }
    *curr_restaurant.mutable_n_by_clus() = {n_by_theta_star[i].begin(),
                                            n_by_theta_star[i].end()};
    *curr_restaurant.mutable_table_to_shared() = {t[i].begin(), t[i].end()};
    *curr_restaurant.mutable_table_to_idio() = {v[i].begin(), v[i].end()};

    state.add_restaurants()->CopyFrom(curr_restaurant);

    bayesmix::SemiHdpState::GroupState curr_group;
    *curr_group.mutable_cluster_allocs() = {s[i].begin(), s[i].end()};
    state.add_groups()->CopyFrom(curr_group);

    for (int l = 0; l < taus.size(); l++) {
      bayesmix::ClusterVal clusval;
      taus[l].write_state_to_proto(&clusval);
      state.add_taus()->CopyFrom(clusval);
    }

    *state.mutable_c() = {c.begin(), c.end()};
    state.set_w(w);
  }
  return state;
}

void SemiHdpSampler::print_debug_string() {
  std::cout << "c: ";
  for (auto& k : c) std::cout << k << ", ";
  std::cout << std::endl << std::endl;

  std::cout << "w: " << w << std::endl;

  for (int r = 0; r < ngroups; r++) {
    std::cout << "**** RESTAURANT: " << r << " *****" << std::endl;
    std::vector<MatrixXd> data_by_theta_star(theta_star[r].size());
    std::cout << "theta_star[r].size(): " << theta_star[r].size() << std::endl;
    for (int i = 0; i < ngroups; i++) {
      if (c[i] == r) {
        for (int j = 0; j < n_by_group[i]; j++) {
          //   std::cout << "s[i][j]" << s[i][j] << std::endl;
          //   std::cout << "currdata: " <<
          //   data_by_theta_star[s[i][j]].transpose() << std::endl;
          //   std::cout
          //   << "appending: " << data[i].row(j) << std::endl;
          bayesmix::append_by_row(&data_by_theta_star[s[i][j]],
                                  data[i].row(j));
        }
      }
    }
    for (int l = 0; l < theta_star[r].size(); l++) {
      std::cout << "THETA STAR (" << r << ", " << l
                << "): m=" << theta_star[r][l].get_mean()
                << ", sd: " << theta_star[r][l].get_sd();

      std::cout << ", DATA: " << data_by_theta_star[l].transpose()
                << std::endl;
    }
  }
}

void SemiHdpSampler::check() {
  // make sure there are no holes in v and t
  int min_t = INT_MAX;
  for (int i = 0; i < ngroups; i++) {
    std::vector<int> v_sorted(v[i]);
    std::sort(v_sorted.begin(), v_sorted.end());

    auto it = std::upper_bound(v_sorted.begin(), v_sorted.end(), -1);
    if (it != v[i].end()) {
      int min_v = *it;
      assert(min_v == 0);
    }

    std::vector<int> t_sorted(t[i]);
    std::sort(t_sorted.begin(), t_sorted.end());

    auto it2 = std::upper_bound(t_sorted.begin(), t_sorted.end(), -1);
    if (it2 != t[i].end()) {
      int min_t_temp = *it2;
      min_t = std::min(min_t, min_t_temp);
    }
  }
  assert(min_t == 0);

  for (int r = 0; r < ngroups; r++) {
    if (n_by_theta_star[r].size() != theta_star[r].size()) {
      throw "Error";
    }
  }

  for (int i = 0; i < ngroups; i++) {
    int r = c[i];
    // std::cout << "i: " << i << ", r: " << r << std::endl;
    int max_s = *std::max_element(s[i].begin(), s[i].end());
    if (max_s >= theta_star[r].size()) {
      throw std::invalid_argument("max s greater than theta_star size");
    }
  }
}
