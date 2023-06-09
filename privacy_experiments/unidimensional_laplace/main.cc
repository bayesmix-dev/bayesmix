#include <chrono>

#include "../utils.h"
#include "src/includes.h"
#include "src/privacy/algorithms/private_conditional.h"
#include "src/privacy/algorithms/private_neal2.h"
#include "src/privacy/channels/laplace_channel.h"

std::string CURR_DIR = "privacy_experiments/unidimensional_laplace/";
std::string OUT_DIR = CURR_DIR + "out/";
std::string PARAM_DIR = CURR_DIR + "params/";

std::pair<Eigen::MatrixXd, Eigen::VectorXi> simulate_private_data(int ndata) {
  Eigen::VectorXd probs = Eigen::VectorXd::Ones(3) / 3.0;
  Eigen::VectorXd means(3);
  means << -5, 0, 5;

  Eigen::MatrixXd out(ndata, 1);
  Eigen::VectorXi clus(ndata);
  auto& rng = bayesmix::Rng::Instance().get();
  for (int i = 0; i < ndata; i++) {
    int c_alloc = bayesmix::categorical_rng(probs, rng, 0);
    clus(i) = c_alloc;
    double x = -100;
    while ((x < -10) || (x > 10)) {
      x = stan::math::normal_rng(means[c_alloc], 1.0, rng);
    }
    out(i, 0) = x;
  }
  return std::make_pair(out, clus);
}

Eigen::VectorXd eval_true_dens(Eigen::VectorXd xgrid) {
  Eigen::VectorXd out(xgrid.size());
  for (int i = 0; i < xgrid.size(); i++) {
    double x = xgrid[i];
    out(i) = 1.0 / 3.0 *
             (std::exp(stan::math::normal_lpdf(x, -5.0, 1.0)) +
              std::exp(stan::math::normal_lpdf(x, 0.0, 1.0)) +
              std::exp(stan::math::normal_lpdf(x, 5.0, 1.0)));
  }
  return out;
}

void save_stuff_to_file(int ndata, double alpha, int repnum,
                        Eigen::MatrixXd private_data, BaseCollector* coll,
                        std::shared_ptr<BaseAlgorithm> algo,
                        std::string algo_name) {
  std::string base_fname =
      OUT_DIR + algo_name + "_ndata_" + std::to_string(ndata) + "_alpha_" +
      std::to_string(alpha) + +"_rep_" + std::to_string(repnum) + "_";

  Eigen::VectorXd grid = Eigen::VectorXd::LinSpaced(1000, -10.0, 10.0);
  Eigen::MatrixXd dens = bayesmix::eval_lpdf_parallel(algo, coll, grid);
  bayesmix::write_matrix_to_file(dens, base_fname + "eval_dens.csv");

  Eigen::MatrixXi clus_allocs = get_cluster_mat(coll, ndata);
  // bayesmix::write_matrix_to_file(clus_allocs, base_fname +
  // "clus_chain.csv");

  Eigen::VectorXi best_clus = bayesmix::cluster_estimate(clus_allocs);
  bayesmix::write_matrix_to_file(best_clus, base_fname + "best_clus.csv");

  // chains for assessing MCMC mixing
  Eigen::VectorXi nclus_chain(clus_allocs.rows());
  for (int j = 0; j < clus_allocs.rows(); j++) {
    Eigen::VectorXi curr_clus = clus_allocs.row(j);
    std::set<int> uniqs{curr_clus.data(), curr_clus.data() + curr_clus.size()};
    nclus_chain(j) = uniqs.size();
  }
  bayesmix::write_matrix_to_file(nclus_chain, base_fname + "nclus_chain.csv");

  Eigen::VectorXd entropy_chain(clus_allocs.rows());
  for (int i = 0; i < clus_allocs.rows(); i++) {
    entropy_chain(i) = cluster_entropy(clus_allocs.row(i).transpose());
  }
  bayesmix::write_matrix_to_file(entropy_chain,
                                 base_fname + "entropy_chain.csv");

  Eigen::VectorXd loglik_chain =
      bayesmix::eval_lpdf_parallel(algo, coll, private_data).rowwise().sum();
  bayesmix::write_matrix_to_file(loglik_chain,
                                 base_fname + "loglik_chain.csv");

  Eigen::VectorXd true_dens = eval_true_dens(grid);
  Eigen::VectorXd error_chain =
      (dens.array().exp().matrix().rowwise() - true_dens.transpose())
          .rowwise()
          .squaredNorm() *
      (grid[1] - grid[0]);
  bayesmix::write_matrix_to_file(error_chain,
                                 base_fname + "l2error_chain.csv");

  Eigen::MatrixXd arate(1, 1);
  if (algo_name == "neal2") {
    arate(0, 0) =
        std::dynamic_pointer_cast<PrivateNeal2>(algo)->get_acceptance_rate();
  } else if (algo_name == "slice") {
    arate(0, 0) =
        std::dynamic_pointer_cast<PrivateConditionalAlgorithm<SliceSampler>>(
            algo)
            ->get_acceptance_rate();
  } else if (algo_name == "blockedgibbs") {
    arate(0, 0) = std::dynamic_pointer_cast<
                      PrivateConditionalAlgorithm<BlockedGibbsAlgorithm>>(algo)
                      ->get_acceptance_rate();
  } else {
    throw std::runtime_error("Unknown algorithm name");
  }

  bayesmix::write_matrix_to_file(arate, base_fname + "acceptance_rate.csv");
}

void run_experiment(int ndata, int repnum) {
  auto [private_data, clus] = simulate_private_data(ndata);
  bayesmix::write_matrix_to_file(
      clus, OUT_DIR + "ndata_" + std::to_string(ndata) + "_rep_" +
                std::to_string(repnum) + "trueclus.csv");

  auto& factory_hier = HierarchyFactory::Instance();
  auto& factory_mixing = MixingFactory::Instance();

  std::vector<double> priv_levels = {1.0, 2.0, 5.0, 10.0, 50.0};
  std::vector<double> epsilons;
  for (double alpha : priv_levels) {
    epsilons.push_back(20.0 / alpha);
  }

  for (int k = 0; k < epsilons.size(); k++) {
    double eps = epsilons[k];
    double alpha = priv_levels[k];
    std::shared_ptr<LaplaceChannel> channel(new LaplaceChannel(eps));
    Eigen::MatrixXd sanitized_data = channel->sanitize(private_data);

    bayesmix::AlgorithmParams algo_proto;
    bayesmix::read_proto_from_file(PARAM_DIR + "algo.asciipb", &algo_proto);

    auto neal2algo = std::make_shared<PrivateNeal2>();
    auto slicealgo =
        std::make_shared<PrivateConditionalAlgorithm<SliceSampler>>();
    auto bgalgo =
        std::make_shared<PrivateConditionalAlgorithm<BlockedGibbsAlgorithm>>();

    std::map<std::string, std::shared_ptr<BaseAlgorithm>> algos;
    algos.insert(std::make_pair("neal2", neal2algo));
    algos.insert(std::make_pair("slice", slicealgo));
    // algos.insert(std::make_pair("blockedgibbs", bgalgo));

    for (auto& [name, algo] : algos) {
      auto hier = factory_hier.create_object("NNIG");
      std::shared_ptr<AbstractMixing> mixing;
      if (name == "neal2") {
        mixing = factory_mixing.create_object("DP");
        bayesmix::read_proto_from_file(PARAM_DIR + "dp_gamma.asciipb",
                                       mixing->get_mutable_prior());
        auto algo_cast = std::dynamic_pointer_cast<PrivateNeal2>(algo);
        algo_cast->set_channel(channel);
        algo_cast->set_public_data(sanitized_data);
      } else if (name == "slice") {
        mixing = factory_mixing.create_object("TruncSB");
        bayesmix::read_proto_from_file(PARAM_DIR + "truncsb.asciipb",
                                       mixing->get_mutable_prior());
        auto algo_cast = std::dynamic_pointer_cast<
            PrivateConditionalAlgorithm<SliceSampler>>(algo);
        algo_cast->set_channel(channel);
        algo_cast->set_public_data(sanitized_data);
      } else if (name == "blockedgibbs") {
        mixing = factory_mixing.create_object("TruncSB");
        bayesmix::read_proto_from_file(PARAM_DIR + "truncsb.asciipb",
                                       mixing->get_mutable_prior());
        auto algo_cast = std::dynamic_pointer_cast<
            PrivateConditionalAlgorithm<BlockedGibbsAlgorithm>>(algo);
        algo_cast->set_channel(channel);
        algo_cast->set_public_data(sanitized_data);
      } else {
        throw std::runtime_error("Unknown algorithm name");
      }
      bayesmix::read_proto_from_file(PARAM_DIR + "nnig_ngg.asciipb",
                                     hier->get_mutable_prior());
      algo->set_hierarchy(hier);
      algo->set_mixing(mixing);
      algo->set_verbose(false);

      BaseCollector* coll = new MemoryCollector();

      algo->read_params_from_proto(algo_proto);
      auto start = std::chrono::high_resolution_clock::now();
      algo->run(coll);
      auto end = std::chrono::high_resolution_clock::now();
      Eigen::MatrixXd time(1, 1);
      time(0, 0) =
          std::chrono::duration_cast<std::chrono::milliseconds>(end - start)
              .count();

      std::string fname = OUT_DIR + name + "_ndata_" + std::to_string(ndata) +
                          "_alpha_" + std::to_string(alpha) + "_rep_" +
                          std::to_string(repnum) + "_time.csv";
      bayesmix::write_matrix_to_file(time, fname);

      save_stuff_to_file(ndata, alpha, repnum, private_data, coll, algo, name);
      delete coll;
    }
  }
}

int main() {
  std::vector<int> ndata = {50, 100, 200, 500, 1000};
  // std::vector<int> ndata = {50, 100};
  int nrep = 48;

#pragma omp parallel for
  for (int i = 0; i < nrep; i++) {
    {
#pragma omp critical
      std::cout << "repnum: " << i << std::endl;
    }
    for (int j = 0; j < ndata.size(); j++) {
      run_experiment(ndata[j], i);
    }
  }
}
