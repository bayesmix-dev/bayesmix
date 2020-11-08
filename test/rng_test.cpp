#include <gtest/gtest.h>
#include <Eigen/Dense>

#include "../src/utils/distributions.hpp"
#include "../src/utils/rng.hpp"

TEST(rng, test1) {
  int rounds = 1000;
  std::vector<double> p = {0.8, 0.1, 0.1};

  int dim = p.size();
  Eigen::Map<Eigen::VectorXd> probas(p.data(), dim);
  std::vector<int> counts(dim);
  for(int i = 0; i < rounds; i++) {
    auto rng = bayesmix::Rng::Instance().get();
    int c = bayesmix::categorical_rng(probas, rng);
    counts[c]++;
  }
  std::cout << "[          ] >>>>> ";
  for(int i = 0; i < dim; i++) {
    std::cout << counts[i] << " ";
  }
  std::cout << std::endl;
}

TEST(rng, test2) {
  int dim = 50;
  std::vector<double> p(dim, 1.0);
  p[0] = 10.0;
  Eigen::Map<Eigen::VectorXd> probas(p.data(), dim);
  probas = probas/probas.sum();

  auto rng = bayesmix::Rng::Instance().get();
  int c1 = bayesmix::categorical_rng(probas, rng);

  auto rng2 = bayesmix::Rng::Instance().get();
  int c2 = bayesmix::categorical_rng(probas, rng2);

  std::cout << "[          ] >>>>> ";
  std::cout << c1 << " " << c2 << " ";

  if (true) {
    auto rng = bayesmix::Rng::Instance().get();
    int c3 = bayesmix::categorical_rng(probas, rng);
    std::cout << c3 << " ";
  }
  if (true) {
    auto rng = bayesmix::Rng::Instance().get();
    int c4 = bayesmix::categorical_rng(probas, rng);
    std::cout << c4 << " " << std::endl;
  } 
}
