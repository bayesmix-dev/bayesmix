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
                          int n, std::mt19937_64& rng) {
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
  int nburn = 5000;
  int niter = 5000;
  MemoryCollector<bayesmix::SemiHdpState> collector;
  SemiHdpSampler sampler(data, update_c);
  sampler.run(500, nburn, niter, 5, &collector, pseudoprior_collectors);
  collector.write_to_file(chainfile);
  auto end = std::chrono::high_resolution_clock::now();
  auto duration =
      std::chrono::duration_cast<std::chrono::seconds>(end - start).count();
  std::cout << "Finished running, duration: " << duration << std::endl;
}

int main() {
  // Scenario VII 
  std::vector<MatrixXd> data(100);
  std::cout << "data.size(): " << data.size() << std::endl;

//   for (int i = 0; i < 10; i++) {
//     data[i] = generate_mixture(0.0, 2.0, 1.0, 0.8, 0.5, 100);
//   }
//   for (int i = 10; i < 20; i++) {
//     data[i] = generate_mixture(-1.0, 1.0, 1.0, 2.0, 0.8, 100);
//   }
  auto &rng = bayesmix::Rng::Instance().get();

  for (int i=0; i < 20; i++) {
    // auto rng = bayesmix::Rng::Instance().get();
    data[i] = generate_mixture(-5, 1.0, 5, 1.0, 0.5, 100, rng);
  }
  for (int i = 20; i < 40; i++) {
    // auto rng = bayesmix::Rng::Instance().get();
    data[i] = generate_mixture(-5.0, 1.0, 0.0, 1.0, 0.5, 100, rng);
  }
  for (int i = 40; i < 60; i++) {
    // auto rng = bayesmix::Rng::Instance().get();
    data[i] = generate_mixture(0.0, 1.0, 5.0, 0.1, 0.5, 100, rng);
  }
  for (int i = 60; i < 80; i++) {
    // auto rng = bayesmix::Rng::Instance().get();
    data[i] = generate_mixture(-10, 1.0, 0.0, 1.0, 0.5, 100, rng);
  }
  for (int i = 80; i < 100; i++) {
    // auto rng = bayesmix::Rng::Instance().get();
    data[i] = generate_mixture(-10, 1.0, 0.0, 1.0, 0.1, 100, rng);
  }

  run_semihdp(data, "s100.recordio", "full");
}