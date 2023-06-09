#include <chrono>

#include "../utils.h"
#include "src/includes.h"
#include "src/privacy/algorithms/private_conditional.h"
#include "src/privacy/algorithms/private_neal2.h"
#include "src/privacy/channels/laplace_channel.h"
#include "src/privacy/channels/wavelet_channel.h"

std::string CURR_DIR = "privacy_experiments/wavelets/";
std::string OUT_DIR = CURR_DIR + "out/";
std::string PARAM_DIR = CURR_DIR + "params/";

std::pair<Eigen::MatrixXd, Eigen::VectorXi> simulate_private_data(int ndata) {
  Eigen::VectorXd probs = Eigen::VectorXd::Ones(3) / 3.0;
  Eigen::VectorXd a_params(3);
  Eigen::VectorXd b_params(3);
  a_params << 5.0, 50.0, 50.0;
  b_params << 50.0, 50.0, 5.0;

  Eigen::MatrixXd out(ndata, 1);
  Eigen::VectorXi clus(ndata);
  auto& rng = bayesmix::Rng::Instance().get();
  for (int i = 0; i < ndata; i++) {
    int c_alloc = bayesmix::categorical_rng(probs, rng, 0);
    clus(i) = c_alloc;
    out(i, 0) =
        stan::math::beta_rng(a_params[c_alloc], b_params[c_alloc], rng);
  }
  return std::make_pair(out, clus);
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

void save_stuff_to_file(int ndata, double alpha, int repnum,
                        Eigen::MatrixXd time, Eigen::MatrixXd private_data,
                        BaseCollector* coll,
                        std::shared_ptr<PrivateNeal2> algo,
                        std::string channel_name, int j) {
  std::string base_fname;
  if (channel_name == "laplace") {
    base_fname = OUT_DIR + "laplace" + "_ndata_" + std::to_string(ndata) +
                 "_alpha_" + std::to_string(alpha) + +"_rep_" +
                 std::to_string(repnum) + "_";
  } else {
    base_fname = OUT_DIR + "wavelet_j_" + std::to_string(j) + "_ndata_" +
                 std::to_string(ndata) + "_alpha_" + std::to_string(alpha) +
                 +"_rep_" + std::to_string(repnum) + "_";
  }

  Eigen::VectorXd grid = Eigen::VectorXd::LinSpaced(1000, 0.0, 1.0);
  Eigen::MatrixXd dens = bayesmix::eval_lpdf_parallel(algo, coll, grid);
  bayesmix::write_matrix_to_file(dens, base_fname + "eval_dens.csv");

  Eigen::MatrixXd clus_allocs = get_cluster_mat(coll, ndata);
  // bayesmix::write_matrix_to_file(clus_allocs, base_fname + "clus.csv");

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
  arate(0, 0) = algo->get_acceptance_rate();

  bayesmix::write_matrix_to_file(arate, base_fname + "acceptance_rate.csv");

  bayesmix::write_matrix_to_file(time, base_fname + "time.csv");
}

void run_experiment(int ndata, int repnum) {
  auto [private_data, clus] = simulate_private_data(ndata);
  bayesmix::write_matrix_to_file(
      clus, OUT_DIR + "ndata_" + std::to_string(ndata) + "_rep_" +
                std::to_string(repnum) + "trueclus.csv");

  std::vector<double> priv_levels = {1.0, 2.0, 5.0, 10.0, 50.0};
  std::vector<int> Js = {2, 4, 6};

  for (int k = 0; k < priv_levels.size(); k++) {
    double alpha = priv_levels[k];
    std::shared_ptr<PrivateNeal2> lap_algo = get_algo1d(
        PARAM_DIR + "bgg_params.asciipb", PARAM_DIR + "dp_gamma.asciipb",
        PARAM_DIR + "algo.asciipb", "BetaGG");

    // LAPLACE CHANNEL
    double lap_sd = 1.0 / alpha;
    std::shared_ptr<LaplaceChannel> lap_channel(new LaplaceChannel(lap_sd));
    Eigen::MatrixXd lap_sanitized_data = lap_channel->sanitize(private_data);
    lap_algo->set_channel(lap_channel);
    lap_algo->set_public_data(lap_sanitized_data);

    BaseCollector* lap_coll = new MemoryCollector();
    auto start = std::chrono::high_resolution_clock::now();
    lap_algo->run(lap_coll);
    auto end = std::chrono::high_resolution_clock::now();
    Eigen::MatrixXd time(1, 1);
    time(0, 0) =
        std::chrono::duration_cast<std::chrono::milliseconds>(end - start)
            .count();
    save_stuff_to_file(ndata, alpha, repnum, time, private_data, lap_coll,
                       lap_algo, "laplace", -1);

    delete lap_coll;

    for (int& j : Js) {
      std::shared_ptr<PrivateNeal2> wavelet_algo = get_algo1d(
          PARAM_DIR + "bgg_params.asciipb", PARAM_DIR + "dp_gamma.asciipb",
          PARAM_DIR + "algo.asciipb", "BetaGG");

      // LAPLACE CHANNEL
      double wave_sd = 12. / alpha * 3.41 * std::pow(2, 0.5 * j);
      std::shared_ptr<WaveletChannel> wav_channel(
          new WaveletChannel(j, wave_sd));
      Eigen::MatrixXd wav_sanitized_data = wav_channel->sanitize(private_data);
      wavelet_algo->set_channel(wav_channel);
      wavelet_algo->set_public_data(wav_sanitized_data);

      BaseCollector* wav_coll = new MemoryCollector();
      auto start = std::chrono::high_resolution_clock::now();
      wavelet_algo->run(wav_coll);
      auto end = std::chrono::high_resolution_clock::now();
      time(0, 0) =
          std::chrono::duration_cast<std::chrono::milliseconds>(end - start)
              .count();
      save_stuff_to_file(ndata, alpha, repnum, time, private_data, wav_coll,
                         wavelet_algo, "wavelet", j);
      delete wav_coll;
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
