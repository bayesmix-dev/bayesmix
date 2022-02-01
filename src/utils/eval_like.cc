#include "eval_like.h"

namespace bayesmix {
Eigen::MatrixXd eval_lpdf_parallel(std::shared_ptr<BaseAlgorithm> algo,
                                   BaseCollector *const collector,
                                   const Eigen::MatrixXd &grid,
                                   int chunk_size) {
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
      curr_lpdfs[i] = curr_algo->lpdf_from_state(grid, Eigen::RowVectorXd(0),
                                                 Eigen::RowVectorXd(0));
    }
    lpdfs.insert(lpdfs.end(), curr_lpdfs.begin(), curr_lpdfs.end());
  } while (keep);
  collector->reset();
  Eigen::MatrixXd out = bayesmix::stack_vectors(lpdfs);
  return out;
}
}  // namespace bayesmix
