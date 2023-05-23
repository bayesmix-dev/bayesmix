#include "../utils.h"
#include "src/algorithms/private_neal2.h"
#include "src/includes.h"
#include "src/privacy/laplace_channel.h"
#include "src/privacy/wavelet_channel.h"

std::string CURR_DIR = "privacy_experiments/wavelets/";
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

void save_stuff_to_file(int ndata, double eps, int repnum, int j,
                        Eigen::MatrixXd private_data, BaseCollector* coll,
                        std::shared_ptr<PrivateNeal2> algo) {
  bool random_init = true;

  std::string base_fname;

  if (random_init) {
    base_fname = OUT_DIR + "random_init_ndata_" + std::to_string(ndata) +
                 "_eps_" + std::to_string(eps) + "_j_" + std::to_string(j) +
                 "_rep_" + std::to_string(repnum) + "_";
  } else {
    base_fname = OUT_DIR + "ndata_" + std::to_string(ndata) + "_eps_" +
                 std::to_string(eps) + "_j_" + std::to_string(j) + "_rep_" +
                 std::to_string(repnum) + "_";
  }
  Eigen::VectorXd grid = Eigen::VectorXd::LinSpaced(1000, -0.5, 1.5);
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

void run_experiment(int ndata, double eps, int repnum) {
  Eigen::MatrixXd private_data = simulate_private_data(ndata);
  std::vector<int> Js = {2, 4, 6};
  // std::vector<int> Js = {4};

  for (int& j : Js) {
    std::shared_ptr<PrivateNeal2> algo =
        get_algo1d(PARAM_DIR + "nnig_ngg.asciipb",
                   PARAM_DIR + "dp_gamma.asciipb", PARAM_DIR + "algo.asciipb");

    algo->set_verbose(false);
    std::shared_ptr<WaveletChannel> channel(new WaveletChannel(j, eps));
    algo->set_channel(channel);

    Eigen::MatrixXd sanitized_data = channel->sanitize(private_data);
    algo->set_public_data(sanitized_data);

    BaseCollector* coll = new MemoryCollector();
    algo->run(coll);
    save_stuff_to_file(ndata, eps, repnum, j, private_data, coll, algo);
    delete coll;
  }
}

int main() {
  int ndata = 250;
  int nrep = 1;

  std::vector<double> epsilons = {2.0, 1.0, 0.5, 0.25, 0.1, 0.05, 0.01};
  // std::vector<double> epsilons = {0.05};

#pragma omp parallel for
  for (auto eps : epsilons) {
    std::cout << "eps: " << eps << std::endl;
    for (int i = 0; i < nrep; i++) {
      run_experiment(ndata, eps, i);
    }
  }

  // std::shared_ptr<WaveletChannel> channel(new WaveletChannel(5, 0.5));
  // std::cout << "basis_size: " << channel->get_basis_size() << std::endl;
  // Eigen::MatrixXd private_data = simulate_private_data(ndata);
  // Eigen::MatrixXd san_data = channel->sanitize(private_data);

  // Eigen::MatrixXd eval_basis(ndata, channel->get_basis_size());
  // for (int i=0; i < ndata; i++) {
  //   eval_basis.row(i) = channel->eval_haar_basis(private_data(i, 0));
  // }

  // std::cout << "data: " << private_data.transpose() << std::endl;
  // std::cout << "rec1: " <<
  // channel->get_candidate_private_data(eval_basis).transpose() << std::endl;
  // std::cout << "data: " <<
  // channel->get_candidate_private_data(san_data).transpose() << std::endl;
}
