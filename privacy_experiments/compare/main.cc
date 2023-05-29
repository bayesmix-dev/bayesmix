#include "../utils.h"
#include "src/algorithms/private_neal2.h"
#include "src/includes.h"
#include "src/privacy/laplace_channel.h"
#include "src/privacy/wavelet_channel.h"

std::string CURR_DIR = "privacy_experiments/compare/";
std::string OUT_DIR = CURR_DIR + "out/";
std::string PARAM_DIR = CURR_DIR + "params/";

Eigen::MatrixXd simulate_private_data(int ndata) {
  Eigen::VectorXd probs = Eigen::VectorXd::Ones(3) / 3.0;
  Eigen::VectorXd a_params(3);
  Eigen::VectorXd b_params(3);
  a_params << 5.0, 50.0, 50.0;
  b_params << 50.0, 50.0, 5.0;

  Eigen::MatrixXd out(ndata, 1);
  auto& rng = bayesmix::Rng::Instance().get();
  for (int i = 0; i < ndata; i++) {
    int c_alloc = bayesmix::categorical_rng(probs, rng, 0);
    out(i, 0) =
        stan::math::beta_rng(a_params[c_alloc], b_params[c_alloc], rng);
  }
  return out;
}

Eigen::VectorXd eval_true_dens(Eigen::VectorXd xgrid) {
  Eigen::VectorXd out(xgrid.size());
  for (int i = 0; i < xgrid.size(); i++) {
    double x = xgrid[i];
    if ((x <= 0) || (x >= 1)) {
      out(i) = 0;
    } else {
      out(i) = 1.0 / 3.0 *
               (std::exp(stan::math::beta_lpdf(x, 5.0, 50.0)) +
                std::exp(stan::math::beta_lpdf(x, 50.0, 50.0)) +
                std::exp(stan::math::beta_lpdf(x, 50.0, 5.0)));
    }
  }
  return out;
}

void save_stuff_to_file(int ndata, double alpha, int repnum, int j,
                        Eigen::MatrixXd private_data, BaseCollector* coll,
                        std::shared_ptr<PrivateNeal2> algo,
                        std::string privacy_channel) {
  std::string base_fname;
  if (privacy_channel == "laplace") {
    base_fname = OUT_DIR + privacy_channel + "_ndata_" +
                 std::to_string(ndata) + "_alpha_" + std::to_string(alpha) +
                 "_rep_" + std::to_string(repnum) + "_";
  } else {
    base_fname = OUT_DIR + privacy_channel + "_ndata_" +
                 std::to_string(ndata) + "_alpha_" + std::to_string(alpha) +
                 "_j_" + std::to_string(j) + "_rep_" + std::to_string(repnum) +
                 "_";
  }

  Eigen::VectorXd grid = Eigen::VectorXd::LinSpaced(1000, 0.0, 1.0);
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
          .squaredNorm() *
      (grid[1] - grid[0]);

  bayesmix::write_matrix_to_file(error_chain,
                                 base_fname + "l2error_chain.csv");

  Eigen::MatrixXd arate(1, 1);
  arate(0, 0) = algo->get_acceptance_rate();

  bayesmix::write_matrix_to_file(arate, base_fname + "acceptance_rate.csv");
}

void run_experiment(int ndata, double alpha, int repnum) {
  Eigen::MatrixXd private_data = simulate_private_data(ndata);
  std::vector<int> Js = {2, 4, 6};

  // Laplace channel
  double lap_eps = 1.0 / alpha;
  std::shared_ptr<LaplaceChannel> lap_channel(new LaplaceChannel(lap_eps));
  std::shared_ptr<PrivateNeal2> algo = get_algo1d(
      PARAM_DIR + "bgg_params.asciipb", PARAM_DIR + "dp_gamma.asciipb",
      PARAM_DIR + "algo.asciipb", "BetaGG");
  algo->set_verbose(false);
  algo->set_channel(lap_channel);

  Eigen::MatrixXd sanitized_data = lap_channel->sanitize(private_data);
  algo->set_public_data(sanitized_data);

  BaseCollector* lap_coll = new MemoryCollector();
  algo->run(lap_coll);
  save_stuff_to_file(ndata, alpha, repnum, -1, private_data, lap_coll, algo,
                     "laplace");
  delete lap_coll;

  // Wavelet channel
  for (int& j : Js) {
    double wavelet_eps = 12.0 / alpha * 3.414 * std::pow(2, 0.5 * j);

    algo = get_algo1d(PARAM_DIR + "bgg_params.asciipb",
                      PARAM_DIR + "dp_gamma.asciipb",
                      PARAM_DIR + "algo.asciipb", "BetaGG");

    algo->set_verbose(false);
    std::shared_ptr<WaveletChannel> wav_channel(
        new WaveletChannel(j, wavelet_eps));
    algo->set_channel(wav_channel);

    Eigen::MatrixXd sanitized_data = wav_channel->sanitize(private_data);
    algo->set_public_data(sanitized_data);

    BaseCollector* coll = new MemoryCollector();
    algo->run(coll);
    save_stuff_to_file(ndata, alpha, repnum, j, private_data, coll, algo,
                       "haar");
    delete coll;
  }
}

int main() {
  std::vector<int> ndata = {50, 100, 250, 500, 1000, 5000, 10000, 50000};
  std::vector<double> alphas = {10.0, 5.0, 2.5, 1.0, 0.5, 0.1, 0.05};
  int nrep = 20;

  for (auto alpha : alphas) {
    for (auto n : ndata) {
#pragma omp parallel for
      for (int i = 0; i < nrep; i++) {
        run_experiment(n, alpha, i);
      }
    }
  }
}
