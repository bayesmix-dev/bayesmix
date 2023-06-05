#include "utils.h"

std::shared_ptr<PrivateNeal2> get_algo1d(std::string hier_params,
                                         std::string mix_params,
                                         std::string algo_params,
                                         std::string hierarchy) {
  std::shared_ptr<PrivateNeal2> algo(new PrivateNeal2());
  auto& factory_hier = HierarchyFactory::Instance();
  auto& factory_mixing = MixingFactory::Instance();
  auto hier = factory_hier.create_object(hierarchy);
  auto mixing = factory_mixing.create_object("DP");

  bayesmix::read_proto_from_file(mix_params, mixing->get_mutable_prior());
  bayesmix::read_proto_from_file(hier_params, hier->get_mutable_prior());

  std::cout << "HIERARCHY PARAMS: "
            << hier->get_mutable_prior()->DebugString();

  bayesmix::AlgorithmParams algo_proto;
  bayesmix::read_proto_from_file(algo_params, &algo_proto);
  algo->read_params_from_proto(algo_proto);

  algo->set_mixing(mixing);
  algo->set_hierarchy(hier);
  return algo;
}

Eigen::MatrixXi get_cluster_mat(BaseCollector* coll, int ndata) {
  Eigen::MatrixXi clusterings(coll->get_size(), ndata);
  for (int i = 0; i < coll->get_size(); i++) {
    bayesmix::AlgorithmState state;
    coll->get_next_state(&state);
    for (int j = 0; j < ndata; j++) {
      clusterings(i, j) = state.cluster_allocs(j);
    }
  }
  coll->reset();
  return clusterings;
}

double cluster_entropy(const Eigen::VectorXd& clustering) {
  double out = 0.0;
  for (int i = 0; i < clustering.maxCoeff(); i++) {
    double cnt = (clustering.array() == i).sum();
    out += cnt * std::log(cnt);
  }
  return out;
}
