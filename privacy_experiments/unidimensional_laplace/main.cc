#include "../utils.h"
#include "src/algorithms/private_neal2.h"
#include "src/includes.h"
#include "src/privacy/laplace_channel.h"

std::string CURR_DIR = "privacy_experiments/unidimensional_laplace/";
std::string OUT_DIR = CURR_DIR + "out/";
std::string PARAM_DIR = CURR_DIR + "params/";

Eigen::MatrixXd simulate_private_data(int ndata) {
  Eigen::VectorXd probs = Eigen::VectorXd::Ones(3) / 3.0;
  Eigen::VectorXd means(3);
  means << -5, 0, 5;

  Eigen::MatrixXd out(ndata, 1);
  auto& rng = bayesmix::Rng::Instance().get();
  for (int i = 0; i < ndata; i++) {
    int c_alloc = bayesmix::categorical_rng(probs, rng, 0);
    out(i, 0) = stan::math::normal_rng(means[c_alloc], 1.0, rng);
  }
  return out;
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

void run_experiment(int ndata, double eps, int repnum) {
  std::shared_ptr<LaplaceChannel> channel(new LaplaceChannel(eps));
  Eigen::MatrixXd private_data = simulate_private_data(ndata);
  Eigen::MatrixXd sanitized_data = channel->sanitize(private_data);

  std::shared_ptr<PrivateNeal2> algo =
      get_algo1d(PARAM_DIR + "nnig_ngg.asciipb",
                 PARAM_DIR + "dp_gamma.asciipb", PARAM_DIR + "algo.asciipb");

  algo->set_public_data(sanitized_data);
  algo->set_channel(channel);
  algo->set_verbose(false);

  BaseCollector* coll = new MemoryCollector();
  algo->run(coll);

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
  int ndata = 250;
  int nrep = 100;
  std::vector epsilons = {2.0, 1.0, 0.5, 0.25, 0.1, 0.05, 0.01};

  for (auto eps : epsilons) {
    std::cout << "eps: " << eps << std::endl;
#pragma omp parallel for
    for (int i = 0; i < nrep; i++) {
      run_experiment(ndata, eps, i);
    }
  }
}
