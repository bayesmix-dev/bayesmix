// This scripts runs the simulations with two populations (Section 6.1)

#include <Eigen/Dense>
#include <src/algorithms/semihdp_sampler.hpp>
#include <src/utils/rng.hpp>
#include <stan/math/prim.hpp>
#include <vector>

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

  SemiHdpSampler sampler1(data1);
  sampler1.initialize();

  int nburn = 10000;
  int niter = 10000;

  sampler1.relabel();

  for (int i = 0; i < nburn; i++) sampler1.pseudo_step();

  sampler1.print_debug_string();

  for (int i = 0; i < niter; i++) sampler1.step();

  sampler1.print_debug_string();
}