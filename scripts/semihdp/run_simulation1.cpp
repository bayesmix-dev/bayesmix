// This scripts runs the simulations with two populations (Section 6.1)

#include <Eigen/Dense>
#include <src/algorithms/neal2_algorithm.hpp>
#include <src/algorithms/semihdp_sampler.hpp>
#include <src/collectors/file_collector.hpp>
#include <src/collectors/memory_collector.hpp>
#include <src/utils/rng.hpp>
#include <stan/math/prim.hpp>
#include <vector>

#include <src/includes.hpp>

using Eigen::MatrixXd;

std::vector<MatrixXd> simulate_data(double m1, double s1, double m2, double s2,
                                    double w1, double m3, double s3, double m4,
                                    double s4, double w2, int n1, int n2) {
  auto& rng = bayesmix::Rng::Instance().get();
  std::vector<MatrixXd> out(2);
  out[0] = MatrixXd::Zero(n1, 1);
  out[1] = MatrixXd::Zero(n2, 1);

  for (int i = 0; i < n1; i++) {
    if (stan::math::uniform_rng(0, 1, rng) < w1) {
      out[0](i, 0) = stan::math::normal_rng(m1, s1, rng);
    } else {
      out[0](i, 0) = stan::math::normal_rng(m2, s2, rng);
    }
  }

  for (int i = 0; i < n2; i++) {
    if (stan::math::uniform_rng(0, 1, rng) < w2) {
      out[1](i, 0) = stan::math::normal_rng(m3, s3, rng);
    } else {
      out[1](i, 0) = stan::math::normal_rng(m4, s4, rng);
    }
  }
  return out;
}

void run_semihdp(const std::vector<MatrixXd> data, std::string chainfile) {
  
  // Collect pseudo priors
  std::vector<MemoryCollector<bayesmix::MarginalState>> pseudoprior_collectors;
  pseudoprior_collectors.resize(data.size());
  bayesmix::DPPrior mix_prior;
  double totalmass = 1.0;
  mix_prior.mutable_fixed_value()->set_value(totalmass);
  for (int i=0; i < data.size(); i++) {
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

    std::cout << "collector.size(): " << pseudoprior_collectors[i].get_size() << std::endl;
  }

  int nburn = 10000;
  int niter = 10000;
  MemoryCollector<bayesmix::SemiHdpState> collector;
  SemiHdpSampler sampler(data);
  sampler.initialize();
  sampler.check();
  sampler.run(nburn, nburn, niter, 5, &collector, pseudoprior_collectors);
  collector.write_to_file(chainfile);
}

int main() {
  // Scenario I
  std::vector<MatrixXd> data1 = simulate_data(0.0, 1.0, 5.0, 1.0, 0.5, 0.0,
                                              1.0, 5.0, 1.0, 0.5, 100, 100);

  // Scenario II
  std::vector<MatrixXd> data2 = simulate_data(5.0, 0.6, 10.0, 0.6, 0.9, 5.0,
                                              0.6, 0.0, 0.6, 0.1, 100, 100);

  // Scenario III
  std::vector<MatrixXd> data3 = simulate_data(0.0, 1.0, 5.0, 1.0, 0.8, 0.0,
                                              1.0, 5.0, 1.0, 0.2, 100, 100);

  std::cout << data1[0].transpose() << std::endl;
  std::cout << data1[1].transpose() << std::endl;
  data1[1] = data1[0];

  run_semihdp(data1, "/home/mario/dev/bayesmix/s1e1_new.recordio");

  // MemoryCollector<bayesmix::SemiHdpState> collector1;
  // SemiHdpSampler sampler1(data1);
  // sampler1.initialize();
  // sampler1.check();
  // sampler1.run(1000, 1000, nburn, niter, 5, &collector1);
  // collector1.write_to_file("/home/mario/dev/bayesmix/s1e1.recordio");

  // MemoryCollector<bayesmix::SemiHdpState> collector2;
  // SemiHdpSampler sampler2(data2);
  // sampler2.initialize();
  // sampler2.check();
  // sampler2.run(1000, 1000, nburn, niter, 5, &collector2);
  // collector2.write_to_file("/home/mario/dev/bayesmix/s1e2.recordio");

  // MemoryCollector<bayesmix::SemiHdpState> collector3;
  // SemiHdpSampler sampler3(data3);
  // sampler3.initialize();
  // sampler3.check();
  // sampler3.run(1000, 1000, nburn, niter, 5, &collector3);
  // collector3.write_to_file("/home/mario/dev/bayesmix/s1e3.recordio");
}