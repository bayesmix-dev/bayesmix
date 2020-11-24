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

std::vector<MatrixXd> read_data(std::string filename) {
  std::ifstream infile(filename);

  std::vector<std::vector<double>> out(3);

  int group;
  float grade;
  std::string line;
  char delim;
  // skip header
  std::getline(infile, line);
  while (std::getline(infile, line)) {
    std::istringstream iss(line);
    if (!(iss >> group >> delim >> grade)) {
      break;
    }
    out[group - 1].push_back(grade);
  }
  std::cout << "data: \n";
  for (auto& group: out) {
      for (auto& k: group) {
          std::cout << k << ", ";
      }
      std::cout << std::endl;
  }

  std::vector<MatrixXd> grades(3);
  for (int g = 0; g < 3; g++) {
    grades[g] = MatrixXd::Zero(out[g].size(), 1);
    for (int j = 0; j < out[g].size(); j++) {
      grades[g](j, 0) = out[g][j];
    }
  }

  return grades;
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
    sampler.set_maxiter(10000);
    sampler.set_burnin(9000);
    sampler.set_mixing(mixing);
    sampler.set_data_and_initial_clusters(data[i], hier, 5);
    sampler.run(&pseudoprior_collectors[i]);
  }

  auto start = std::chrono::high_resolution_clock::now();
  int nburn = 5000;
  int niter = 5000;
  MemoryCollector<bayesmix::SemiHdpState> collector;
  SemiHdpSampler sampler(data, update_c);
  sampler.run(nburn, nburn, niter, 5, &collector, pseudoprior_collectors);
  collector.write_to_file(chainfile);
  auto end = std::chrono::high_resolution_clock::now();
  auto duration =
      std::chrono::duration_cast<std::chrono::seconds>(end - start).count();
  std::cout << "Finished running, duration: " << duration << std::endl;
}

int main() {
  // Scenario VII
  std::vector<MatrixXd> data = read_data(
      "/home/mario/PhD/exchangeability/data/grades_chile_norm_jit.csv");
  
  std::cout <<" DATA " << std::endl;
  std::cout << data[0].transpose() << std::endl;
  std::cout << data[1].transpose() << std::endl;
  std::cout << data[2].transpose() << std::endl;

  run_semihdp(data, "chile.recordio", "full");
}