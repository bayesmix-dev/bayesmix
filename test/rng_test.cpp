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
    int c = stan::math::categorical_rng(probas, rng) - 1;
    counts[c]++;
  }
  for(int i = 0; i < dim; i++) {
    std::cout << counts[i] << " ";
  }
  std::cout << std::endl;
}
