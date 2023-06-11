#include "private_neal2.h"

#include "src/hierarchies/likelihoods/uni_norm_likelihood.h"

void PrivateNeal2::set_public_data(const Eigen::MatrixXd &public_data_,
                                   bool random_init) {
  if (!random_init) {
    data = privacy_channel->get_candidate_private_data(public_data_);
  } else {
    auto ex_datum = unique_values[0]->get_likelihood()->sample();
    data.resize(public_data_.rows(), ex_datum.size());
    for (int i = 0; i < public_data_.rows(); i++) {
      data.row(i) = unique_values[0]->get_likelihood()->sample();
    }
  }
  private_data = data;
  this->public_data = public_data_;
}

void PrivateNeal2::print_startup_message() const {
  std::string msg = "Running PrivateNeal2 algorithm with " +
                    bayesmix::HierarchyId_Name(unique_values[0]->get_id()) +
                    " hierarchies, " +
                    bayesmix::MixingId_Name(mixing->get_id()) + " mixing...";
  std::cout << msg << std::endl;
}

void PrivateNeal2::sample_allocations() {
  int ndata = data.rows();

  auto &rng = bayesmix::Rng::Instance().get();

  for (int i = 0; i < ndata; i++) {
    // Sample the private datum from the CRP
    unsigned int c_old = allocations[i];
    std::shared_ptr<AbstractHierarchy> singleton_val =
        unique_values[0]->clone();
    bool singleton = (unique_values[c_old]->get_card() <= 1);
    if (singleton) {
      // std::cout << "removing singleton" << std::endl;
      // TODO: if more than one aux hierarchy, remember to save this value
      // to the first aux param
      bayesmix::AlgorithmState::ClusterState curr_val;
      unique_values[allocations[i]]->write_state_to_proto(&curr_val);
      curr_val.set_cardinality(0);
      singleton_val->set_state_from_proto(curr_val);
      remove_singleton(c_old);
    } else {
      unique_values[c_old]->remove_datum(i, private_data.row(i),
                                         update_hierarchy_params(),
                                         hier_covariates.row(i));
    }

    Eigen::VectorXd probs = stan::math::softmax(get_cluster_prior_mass(i));
    int c_new = bayesmix::categorical_rng(probs, rng, 0);

    std::shared_ptr<AbstractHierarchy> new_unique = unique_values[0]->clone();

    // Sample the private datum
    Eigen::VectorXd new_private_datum;

    unsigned int n_clust = unique_values.size();
    if (c_new == n_clust) {
      new_unique->sample_prior();
      new_private_datum = new_unique->get_likelihood()->sample();
    } else {
      new_private_datum = unique_values[c_new]->get_likelihood()->sample();
    }
    // compute the acceptance rate
    double log_a_rate =
        privacy_channel->lpdf(public_data.row(i), new_private_datum) -
        privacy_channel->lpdf(public_data.row(i), private_data.row(i));

    n_prop += 1;
    if (std::log(stan::math::uniform_rng(0, 1, rng)) < log_a_rate) {
      // std::cout << "ACCEPTED" << std::endl;
      // std::cout << "old private: " << private_data(i, 0)
      //           << ", new_private: " << new_private_datum(0) << std::endl;

      private_data.row(i) = new_private_datum;
      n_acc += 1;
      if (c_new == n_clust) {
        new_unique->add_datum(i, private_data.row(i),
                              update_hierarchy_params(),
                              hier_covariates.row(i));
        unique_values.push_back(new_unique);
        allocations[i] = unique_values.size() - 1;
        new_unique->sample_full_cond(!update_hierarchy_params());
      } else {
        allocations[i] = c_new;
        unique_values[c_new]->add_datum(i, private_data.row(i),
                                        update_hierarchy_params(),
                                        hier_covariates.row(i));
      }
    } else {
      if (singleton) {
        singleton_val->add_datum(i, private_data.row(i),
                                 update_hierarchy_params(),
                                 hier_covariates.row(i));
        unique_values.push_back(singleton_val);
        allocations[i] = unique_values.size() - 1;
      } else {
        allocations[i] = c_old;
        unique_values[c_old]->add_datum(i, private_data.row(i),
                                        update_hierarchy_params(),
                                        hier_covariates.row(i));
      }
    }
  }
  // std::cout << std::endl << std::endl << std::endl;
  // for (int h=0; h < unique_values.size(); h++) {
  //   std::cout << "Cluster: " << h << std::endl;
  //   std::cout << "Params: " <<
  //   unique_values[h]->get_state_proto()->DebugString(); std::cout << "Data:
  //   "; for (auto& j: unique_values[h]->get_data_idx()) {
  //     std::cout << private_data(j,0) << ", ";
  //   }
  //   std::cout << std::endl;
  // }
}

void PrivateNeal2::sample_unique_values() {
  for (auto &un : unique_values) {
    un->sample_full_cond(!update_hierarchy_params());
  }
}

Eigen::VectorXd PrivateNeal2::lpdf_marginal_component(
    const std::shared_ptr<AbstractHierarchy> hier, const Eigen::MatrixXd &grid,
    const Eigen::RowVectorXd &covariate) const {
  if (hier->is_conjugate()) {
    return hier->prior_pred_lpdf_grid(grid, covariate);
  }

  int n_mc = 100;
  unsigned int n_grid = grid.rows();
  Eigen::VectorXd lpdf_(n_grid);
  Eigen::MatrixXd lpdf_temp(n_grid, n_mc);

  for (size_t i = 0; i < n_mc; i++) {
    hier->sample_prior();
    lpdf_temp.col(i) = hier->like_lpdf_grid(grid, covariate);
  }
  for (size_t i = 0; i < n_grid; i++) {
    lpdf_(i) = stan::math::log_sum_exp(lpdf_temp.row(i));
  }
  return lpdf_.array() - log(1.0 * n_mc);
}

Eigen::VectorXd PrivateNeal2::get_cluster_prior_mass(
    const unsigned int data_idx) const {
  unsigned int n_data = data.rows();
  unsigned int n_clust = unique_values.size();
  Eigen::VectorXd logprior(n_clust + 1);
  for (size_t j = 0; j < n_clust; j++) {
    // Probability of being assigned to an already existing cluster
    logprior(j) = mixing->get_mass_existing_cluster(
        n_data - 1, n_clust, true, true, unique_values[j],
        mix_covariates.row(data_idx));
  }
  // Further update with marginal component
  logprior(n_clust) = mixing->get_mass_new_cluster(
      n_data - 1, n_clust, true, true, mix_covariates.row(data_idx));

  return logprior;
}
