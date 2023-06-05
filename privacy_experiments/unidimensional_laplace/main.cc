#include "../utils.h"
#include "src/algorithms/private_conditional.h"
#include "src/algorithms/private_neal2.h"
#include "src/includes.h"
#include "src/privacy/laplace_channel.h"

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
  return std::make_pair<out, clus>;
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

void save_stuff_to_file(int ndata, double eps, int repnum, int j,
                        Eigen::MatrixXd private_data, BaseCollector* coll,
                        std::shared_ptr<BaseAlgorithm> algo,
                        std::string algo_name) {
  std::string base_fname =
      OUT_DIR + "_" + algo_name + "_ndata_" + std::to_string(ndata) + "_eps_" +
      std::to_string(eps) + +"_rep_" + std::to_string(repnum) + "_";

  Eigen::VectorXd grid = Eigen::VectorXd::LinSpaced(1000, 0.0, 1.0);
  Eigen::MatrixXd dens = bayesmix::eval_lpdf_parallel(algo, coll, grid);
  bayesmix::write_matrix_to_file(dens, base_fname + "eval_dens.csv");

  Eigen::MatrixXi clus_allocs = get_cluster_mat(coll, ndata);
  // bayesmix::write_matrix_to_file(clus_allocs, base_fname +
  // "clus_chain.csv");

  Eigen::VectorXi best_clus = bayesmix::cluster_estimate(clusterings);
  bayesmix::write_matrix_to_file(best_clus, base_fname + "best_clus.csv");

  // chains for assessing MCMC mixing
  Eigen::VectorXi nclus_chain(clus_allocs.rows());
  for (int j = 0; j < clus_allocs.rows(); j++) {
    Eigen::VectorXi curr_clus = clus_allocs.row(j);
    std::set<int> uniqs{curr_clus.data(), curr_clus.data() + curr_clus.size()};
    nclus_chain(i) = uniqs.size();
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
  arate(0, 0) = algo->get_acceptance_rate();
  bayesmix::write_matrix_to_file(arate, base_fname + "acceptance_rate.csv");
}

void run_experiment(int ndata, int repnum) {
  auto [private_data, clus] = simulate_private_data(ndata);
  bayesmix::write_matrix_to_file(clus, OUT_DIR + "ndata_" +
                                           std::to_string(ndata) + "_rep_" +
                                           std::to_string(repnum) + ".csv")

      auto& factory_hier = HierarchyFactory::Instance();
  auto& factory_mixing = MixingFactory::Instance();

  std::vector priv_levels = {0.1, 0.3, 1.0, 3.0, 10.0, 100.0};
  std::vector epsilons;
  for (alpha : priv_levels) {
    epsilons.push_back(20 / alpha);
  }

  for (auto& eps : epsilons) {
    std::shared_ptr<LaplaceChannel> channel(new LaplaceChannel(eps));
    Eigen::MatrixXd sanitized_data = channel->sanitize(private_data);

    bayesmix::AlgorithmParams algo_proto;
    bayesmix::read_proto_from_file(PARAM_DIR + "algo.asciipb", &algo_proto);

    auto neal2algo = std::make_shared<PrivateNeal2>();
    auto slicealgo =
        std::make_shared<PrivateConditionalAlgorithm<SliceSampler>>();
    auto bgalgo =
        std::make_shared<PrivateConditionalAlgorithm<BlockedGibbsAlgorithm>>();

    std::map<string, std::shared_ptr<BaseAlgorithm>> algos;
    algos.insert(std::make_pair("neal2", neal2algo));
    algos.insert(std::make_pair("slice", slicealgo));
    algos.insert(std::make_pair("blockedgibbs", bgalgo));

    for (auto& [name, algo] : algos) {
      auto hier = factory_hier.create_object(hierarchy);
      std::shared_ptr<AbstractMixing> mixing;
      if (name == "neal2") {
        mixing = factory_mixing.create_object("DP");
        bayesmix::read_proto_from_file(PARAM_DIR + "dp_gamma.asciipb",
                                       mixing->get_mutable_prior());
      } else {
        mixing = factory_mixing.create_object("TruncSB");
        bayesmix::read_proto_from_file(PARAM_DIR + "truncsb.asciipb",
                                       mixing->get_mutable_prior());
      }

      bayesmix::read_proto_from_file(PARAM_DIR + "nnig_ngg.asciipb",
                                     hier->get_mutable_prior());
      algo->set_hierarchy(hier);
      algo->set_mixing(mixing);
      algo->set_channel(channel);
      algo->set_public_data(sanitized_data);
      algo->set_verbose(true);

      BaseCollector* coll = new MemoryCollector();
      algo->run(coll);
      save_stuff_to_file(ndata, eps, repnum, private_data, coll, algo, name);
      delete coll;
    }
  }

  // Save stuff to file
  std::string base_fname = OUT_DIR + "ndata_" + std::to_string(ndata) +
                           "_eps_" + std::to_string(eps) + "_rep_" +
                           std::to_string(repnum) + "_";
  Eigen::VectorXd grid = Eigen::VectorXd::LinSpaced(1000, -15, 15);
  Eigen::MatrixXd dens = bayesmix::eval_lpdf_parallel(algo, coll, grid);
  bayesmix::write_matrix_to_file(dens, base_fname + "eval_dens.csv");

  Eigen::MatrixXd clus_allocs = get_cluster_mat(coll, ndata);
  bayesmix::write_matrix_to_file(clus_allocs, base_fname + "clus.csv");

  // chains for assessing MCMC mixing
  Eigen::VectorXi nclus_chain = clus_allocs.rowwise().maxCoeff();
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
          .squaredNorm() /
      (grid[1] - grid[0]);

  bayesmix::write_matrix_to_file(error_chain,
                                 base_fname + "l2error_chain.csv");

  Eigen::MatrixXd arate(1, 1);
  arate(0, 0) = algo->get_acceptance_rate();

  bayesmix::write_matrix_to_file(arate, base_fname + "acceptance_rate.csv");
}

int main() {
  std::vector<int> ndata = {50, 250, 500, 1000, 5000, 10000};
  int nrep = 25;

#pragma omp parallel for collapse(2)
  for (int i = 0; i < nrep; i++) {
    for (auto& n : ndata) {
      run_experiment(n, eps, i);
    }
  }
}
