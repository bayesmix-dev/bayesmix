// This scripts runs the simulations with four populations (Section 6.1)

#include <Eigen/Dense>
#include <chrono>
#include <src/algorithms/neal2_algorithm.hpp>
#include <src/algorithms/semihdp_sampler.hpp>
#include <src/collectors/file_collector.hpp>
#include <src/collectors/memory_collector.hpp>
#include <src/includes.hpp>
#include <src/utils/rng.hpp>
#include <stan/math/prim.hpp>
#include <vector>

using Eigen::MatrixXd;

MatrixXd generate_mixture(double m1, double s1, double m2, double s2, double w,
                          int n) {
  auto& rng = bayesmix::Rng::Instance().get();
  MatrixXd out(n, 1);
  for (int i = 0; i < n; i++) {
    if (stan::math::uniform_rng(0, 1, rng) < w) {
      out(i, 0) = stan::math::normal_rng(m1, s1, rng);
    } else {
      out(i, 0) = stan::math::normal_rng(m2, s2, rng);
    }
  }
  return out;
}

void run_semihdp(const std::vector<MatrixXd> data, std::string chainfile,
                 std::string update_c = "full") {
  // Collect pseudo priors
  std::vector<MemoryCollector<bayesmix::MarginalState>> pseudoprior_collectors;
  pseudoprior_collectors.resize(data.size());
  bayesmix::DPPrior mix_prior;
  double totalmass = 1.0;
  mix_prior.mutable_fixed_value()->set_value(totalmass);
  for (int i = 0; i < data.size(); i++) {
    auto mixing = std::make_shared<DirichletMixing>();
    mixing->set_prior(mix_prior);
    auto hier = std::make_shared<NNIGHierarchy>();
    hier->set_mu0(data[i].mean());
    hier->set_lambda0(0.1);
    hier->set_alpha0(2.0);
    hier->set_beta0(2.0);

    Neal2Algorithm sampler;
    sampler.set_maxiter(2000);
    sampler.set_burnin(1000);
    sampler.set_mixing(mixing);
    sampler.set_data_and_initial_clusters(data[i], hier, 5);
    sampler.run(&pseudoprior_collectors[i]);
  }

  auto start = std::chrono::high_resolution_clock::now();
  int nburn = 10000;
  int niter = 10000;
  MemoryCollector<bayesmix::SemiHdpState> collector;
  SemiHdpSampler sampler(data, update_c);
  sampler.initialize();
  sampler.check();
  sampler.run(nburn, nburn, niter, 5, &collector, pseudoprior_collectors);
  collector.write_to_file(chainfile);
  auto end = std::chrono::high_resolution_clock::now();
  auto duration =
      std::chrono::duration_cast<std::chrono::seconds>(end - start).count();
  std::cout << "Finished running, duration: " << duration << std::endl;
}

int main() {
  // Scenario IV std::vector<MatrixXd> data1(4);
  std::vector<MatrixXd> data1(4);
  std::cout << "data1.size(): " << data1.size() << std::endl;
  for (int i = 0; i < 4; i++) {
    data1[i] = MatrixXd::Zero(100, 1);
  }
  std::cout << data1[0].transpose() << std::endl;
  std::cout << "assigning stuff to data1" << std::endl;

  for (int j = 0; j < 100; j++) {
    for (int i = 0; i < 3; i++) {
      auto& rng = bayesmix::Rng::Instance().get();
      data1[i](j, 0) = stan::math::normal_rng(0.0, 1.0, rng);
    }
    auto& rng = bayesmix::Rng::Instance().get();
    data1[3](j, 0) = stan::math::skew_normal_rng(0.0, 1.0, 1.0, rng);
  }

  std::cout << data1[0].transpose() << std::endl;
  std::cout << data1[1].transpose() << std::endl;
  std::cout << data1[2].transpose() << std::endl;
  std::cout << data1[3].transpose() << std::endl;

  std::cout << "Data1 OK" << std::endl;
  run_semihdp(data1, "/home/mario/dev/bayesmix/s2e1_full.recordio");
  run_semihdp(data1, "/home/mario/dev/bayesmix/s2e1_metro_base.recordio",
              "metro_base");
  run_semihdp(data1, "/home/mario/dev/bayesmix/s2e1_metro_dist.recordio",
              "metro_dist");

  // // Scenario V
  // auto& rng = bayesmix::Rng::Instance().get();
  // std::vector<MatrixXd> data2(4);
  // for (int i = 0; i < 4; i++) {
  //   data2[i].resize(100, 1);
  // }

  // for (int j = 0; j < 100; j++) {
  //   data2[0](j, 0) = stan::math::normal_rng(0, 1, rng);
  //   data2[3](j, 0) = stan::math::normal_rng(0, 1, rng);
  //   data2[1](j, 0) = stan::math::normal_rng(0, std::sqrt(2.25), rng);
  //   data2[2](j, 0) = stan::math::normal_rng(0, std::sqrt(0.25), rng);
  // }

  // std::cout << "Data2 OK" << std::endl;
  // run_semihdp(data2, "/home/mario/dev/bayesmix/s2e2.recordio");

  // // // Scenario VI
  // std::vector<MatrixXd> data3(4);
  // data3[0] = generate_mixture(0, 1, 5, 1, 0.5, 100);
  // data3[1] = generate_mixture(0, 1, 5, 1, 0.5, 100);
  // data3[2] = generate_mixture(0, 1, -5, 1, 0.5, 100);
  // data3[3] = generate_mixture(-5, 1, 5, 1, 0.5, 100);

  // std::cout << "Data3 OK" << std::endl;

  // run_semihdp(data3, "/home/mario/dev/bayesmix/s2e3.recordio");
}