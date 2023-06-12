#include <chrono>

#include "../utils.h"
#include "src/includes.h"
#include "src/privacy/algorithms/private_conditional.h"
#include "src/privacy/algorithms/private_neal2.h"
#include "src/privacy/channels/gaussian_channel.h"
#include "src/privacy/hierarchies/truncated_nnig_hier.h"

std::string CURR_DIR = "privacy_experiments/gaussian/";
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

double get_sigma(double alpha, double delta) {
  return 20 / alpha * std::sqrt(2 * std::log(1.25 / delta));
}

void save_stuff_to_file(int ndata, double alpha, double delta, int repnum,
                        Eigen::MatrixXd private_data, BaseCollector* coll,
                        std::shared_ptr<BaseAlgorithm> algo,
                        std::string algo_name, double algo_is_private) {
  std::string base_fname =
      OUT_DIR + algo_name + "_ndata_" + std::to_string(ndata) + "_alpha_" +
      std::to_string(alpha) + "_delta_" + std::to_string(delta) + "_rep_" +
      std::to_string(repnum) + "_";

  Eigen::VectorXd grid = Eigen::VectorXd::LinSpaced(1000, -10.0, 10.0);
  Eigen::MatrixXd dens;
  if (algo_is_private) {
    Eigen::RowVectorXd fake_covariate;
    dens = bayesmix::eval_lpdf_parallel(algo, coll, grid, fake_covariate,
                                        fake_covariate, false, 1);
    Eigen::MatrixXd arate(1, 1);
    arate(0, 0) =
        std::dynamic_pointer_cast<PrivateNeal2>(algo)->get_acceptance_rate();
    bayesmix::write_matrix_to_file(arate, base_fname + "acceptance_rate.csv");
  } else {
    double eta = get_sigma(alpha, delta);
    dens = eval_private_nnig_lpdf(algo, coll, grid, eta * eta, 1);
  }

  if (repnum == 0) {
    bayesmix::write_matrix_to_file(dens, base_fname + "eval_dens.csv");
  }

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
}

std::shared_ptr<PrivateNeal2> get_private_neal2() {
  auto& factory_hier = HierarchyFactory::Instance();
  auto hier = factory_hier.create_object("NNIG");
  bayesmix::read_proto_from_file(PARAM_DIR + "nnig_ngg.asciipb",
                                 hier->get_mutable_prior());
  hier->initialize();

  auto& factory_mixing = MixingFactory::Instance();
  auto mixing = factory_mixing.create_object("DP");
  bayesmix::read_proto_from_file(PARAM_DIR + "dp_gamma.asciipb",
                                 mixing->get_mutable_prior());

  auto algo = std::make_shared<PrivateNeal2>();
  bayesmix::AlgorithmParams algo_proto;
  bayesmix::read_proto_from_file(PARAM_DIR + "algo.asciipb", &algo_proto);
  algo->read_params_from_proto(algo_proto);

  algo->set_hierarchy(hier);
  algo->set_mixing(mixing);
  algo->set_verbose(false);
  return algo;
}

std::shared_ptr<Neal2Algorithm> get_neal2(double eps_sq) {
  auto hier = std::make_shared<PrivateNIGHier>();
  bayesmix::read_proto_from_file(PARAM_DIR + "nnig_ngg.asciipb",
                                 hier->get_mutable_prior());
  hier->set_var_bounds(eps_sq);
  hier->initialize();

  auto& factory_mixing = MixingFactory::Instance();
  auto mixing = factory_mixing.create_object("DP");

  bayesmix::read_proto_from_file(PARAM_DIR + "dp_gamma.asciipb",
                                 mixing->get_mutable_prior());
  auto algo = std::make_shared<Neal2Algorithm>();
  bayesmix::AlgorithmParams algo_proto;
  bayesmix::read_proto_from_file(PARAM_DIR + "algo.asciipb", &algo_proto);
  algo->read_params_from_proto(algo_proto);

  algo->set_hierarchy(hier);
  algo->set_mixing(mixing);
  algo->set_verbose(false);
  return algo;
}

std::shared_ptr<Neal2Algorithm> get_neal3(double eps_sq) {
  auto hier = std::make_shared<PrivateNIGHier>();
  bayesmix::read_proto_from_file(PARAM_DIR + "nnig_ngg.asciipb",
                                 hier->get_mutable_prior());
  hier->set_var_bounds(eps_sq);
  hier->initialize();

  auto& factory_mixing = MixingFactory::Instance();
  auto mixing = factory_mixing.create_object("DP");

  bayesmix::read_proto_from_file(PARAM_DIR + "dp_gamma.asciipb",
                                 mixing->get_mutable_prior());
  auto algo = std::make_shared<Neal3Algorithm>();
  bayesmix::AlgorithmParams algo_proto;
  bayesmix::read_proto_from_file(PARAM_DIR + "algo.asciipb", &algo_proto);
  algo->read_params_from_proto(algo_proto);

  algo->read_params_from_proto(algo_proto);
  algo->set_hierarchy(hier);
  algo->set_mixing(mixing);
  algo->set_verbose(false);
  return algo;
}

void run_experiment(int ndata, int repnum) {
  Eigen::MatrixXd private_data;
  Eigen::VectorXi clus;
  auto temp = simulate_private_data(ndata);
  private_data = std::get<0>(temp);
  clus = std::get<1>(temp);
  bayesmix::write_matrix_to_file(
      clus, OUT_DIR + "ndata_" + std::to_string(ndata) + "_rep_" +
                std::to_string(repnum) + "trueclus.csv");

  auto& factory_hier = HierarchyFactory::Instance();
  auto& factory_mixing = MixingFactory::Instance();

  std::vector<double> priv_levels = {50.0, 25.0, 10.0, 5.0};
  std::vector<double> deltas = {0.25, 0.1, 0.01};

  for (int k = 0; k < priv_levels.size(); k++) {
    for (int h = 0; h < deltas.size(); h++) {
      double alpha = priv_levels[k];
      double delta = deltas[h];

      double eps = get_sigma(alpha, delta);
      Eigen::MatrixXd time(1, 1);
      std::shared_ptr<GaussianChannel> channel(new GaussianChannel(eps));
      Eigen::MatrixXd sanitized_data = channel->sanitize(private_data);

      auto neal2 = get_neal2(eps * eps);
      neal2->set_data(sanitized_data);
      BaseCollector* neal2coll = new MemoryCollector();
      neal2->run(neal2coll);
      save_stuff_to_file(ndata, alpha, delta, repnum, private_data, neal2coll,
                         neal2, "neal2", false);
      delete neal2coll;

      auto neal3 = get_neal3(eps * eps);
      neal3->set_data(sanitized_data);
      BaseCollector* neal3coll = new MemoryCollector();
      neal3->run(neal3coll);
      auto end = std::chrono::high_resolution_clock::now();
      save_stuff_to_file(ndata, alpha, delta, repnum, private_data, neal3coll,
                         neal3, "neal3", false);
      delete neal3coll;

      // Private Version of the Algorithm
      auto privateneal2 = get_private_neal2();
      privateneal2->set_channel(channel);
      privateneal2->set_public_data(sanitized_data);
      BaseCollector* privneal2coll = new MemoryCollector();
      privateneal2->run(privneal2coll);
      save_stuff_to_file(ndata, alpha, delta, repnum, private_data,
                         privneal2coll, privateneal2, "privateneal2", true);
      delete privneal2coll;
    }
  }
}

int main() {
  int ndata = 250;
  int nrep = 48;
#pragma omp parallel for
  for (int i = 0; i < nrep; i++) {
    {
#pragma omp critical
      std::cout << "repnum: " << i << std::endl;
    }
    run_experiment(ndata, i);
  }
}
