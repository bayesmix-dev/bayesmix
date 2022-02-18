#include "eval_like.h"

namespace bayesmix {

Eigen::MatrixXd eval_lpdf_parallel(
    const std::shared_ptr<BaseAlgorithm> algo, BaseCollector *const collector,
    const Eigen::MatrixXd &grid, const Eigen::RowVectorXd &hier_covariate,
    const Eigen::RowVectorXd &mix_covariate, const bool low_memory /*= false*/,
    const int njobs /*= 4*/, const int chunk_size /*= 100*/) {
  if (low_memory) {
    return bayesmix::internal::eval_lpdf_parallel_lowmemory(
        algo, collector, grid, hier_covariate, mix_covariate, chunk_size);
  } else {
    return bayesmix::internal::eval_lpdf_parallel_fullmemory(
        algo, collector, grid, hier_covariate, mix_covariate, njobs);
  }
}

Eigen::MatrixXd internal::eval_lpdf_parallel_lowmemory(
    const std::shared_ptr<BaseAlgorithm> algo, BaseCollector *const collector,
    const Eigen::MatrixXd &grid, const Eigen::RowVectorXd &hier_covariate,
    const Eigen::RowVectorXd &mix_covariate, const int chunk_size /*= 100*/) {
  std::vector<Eigen::VectorXd> lpdfs;
  bool keep = true;
  do {
    bayesmix::AlgorithmState base_state;
    std::vector<std::shared_ptr<google::protobuf::Message>> states =
        collector->get_chunk(chunk_size, keep, &base_state);
    std::vector<Eigen::VectorXd> curr_lpdfs(states.size());
#pragma omp parallel for
    for (int i = 0; i < states.size(); i++) {
      std::shared_ptr<BaseAlgorithm> curr_algo = algo->clone();
      curr_algo->set_state_proto(states[i]);
      curr_lpdfs[i] =
          curr_algo->lpdf_from_state(grid, hier_covariate, mix_covariate);
    }
    lpdfs.insert(lpdfs.end(), curr_lpdfs.begin(), curr_lpdfs.end());
  } while (keep);
  collector->reset();
  Eigen::MatrixXd out = bayesmix::stack_vectors(lpdfs);
  return out;
}

Eigen::MatrixXd internal::eval_lpdf_parallel_fullmemory(
    const std::shared_ptr<BaseAlgorithm> algo, BaseCollector *const collector,
    const Eigen::MatrixXd &grid, const Eigen::RowVectorXd &hier_covariate,
    const Eigen::RowVectorXd &mix_covariate, const int njobs /*= 4*/) {
  bayesmix::AlgorithmState base_state;
  std::vector<std::shared_ptr<google::protobuf::Message>> chain =
      collector->get_whole_chain(&base_state);
  auto chain_shards = bayesmix::internal::gen_even_slices(chain, njobs);
  std::vector<Eigen::MatrixXd> lpdfs(njobs);

#pragma omp parallel for
  for (int i = 0; i < njobs; i++) {
    std::shared_ptr<BaseAlgorithm> curr_algo = algo->clone();
    Eigen::MatrixXd curr_lpdfs(chain_shards[i].size(), grid.rows());
    for (int j = 0; j < chain_shards[i].size(); j++) {
      curr_algo->set_state_proto(chain_shards[i][j]);
      curr_lpdfs.row(j) =
          curr_algo->lpdf_from_state(grid, hier_covariate, mix_covariate);
    }
    lpdfs[i] = curr_lpdfs;
  }
  collector->reset();
  Eigen::MatrixXd out = bayesmix::vstack(lpdfs);
  return out;
}

}  // namespace bayesmix
