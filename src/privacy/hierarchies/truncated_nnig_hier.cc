#include "truncated_nnig_hier.h"

Eigen::MatrixXd eval_private_nnig_lpdf(
    const std::shared_ptr<BaseAlgorithm> algo, BaseCollector *const collector,
    const Eigen::MatrixXd &grid, double var_l, int njobs) {
  bayesmix::AlgorithmState base_state;
  std::vector<std::shared_ptr<google::protobuf::Message>> chain =
      collector->get_whole_chain(&base_state);
  auto chain_shards = bayesmix::internal::gen_even_slices(chain, njobs);
  std::vector<Eigen::MatrixXd> lpdfs(njobs);
  Eigen::RowVectorXd fake_cov;

#pragma omp parallel for
  for (int i = 0; i < njobs; i++) {
    std::shared_ptr<BaseAlgorithm> curr_algo = algo->clone();
    Eigen::MatrixXd curr_lpdfs(chain_shards[i].size(), grid.rows());
    for (int j = 0; j < chain_shards[i].size(); j++) {
      // HACK HERE: we modify by hand the variance of the state to take into
      // account the privacy shift
      auto state_cast = std::dynamic_pointer_cast<bayesmix::AlgorithmState>(
          chain_shards[i][j]);
      for (int h = 0; h < state_cast->cluster_states_size(); h++) {
        double curr_var =
            state_cast->mutable_cluster_states(h)->uni_ls_state().var();
        state_cast->mutable_cluster_states(h)->mutable_uni_ls_state()->set_var(
            curr_var - var_l);
      }

      curr_algo->set_state_proto(chain_shards[i][j]);
      Eigen::VectorXd lpdf_eval =
          curr_algo->lpdf_from_state(grid, fake_cov, fake_cov);
      curr_lpdfs.row(j) = lpdf_eval.transpose();
    }
    lpdfs[i] = curr_lpdfs;
  }
  collector->reset();
  Eigen::MatrixXd out = bayesmix::vstack(lpdfs);
  return out;
}
