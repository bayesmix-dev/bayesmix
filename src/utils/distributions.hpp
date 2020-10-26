#ifndef DISTRIBUTIONS_HPP
#define DISTRIBUTIONS_HPP

#include <Eigen/Dense>
#include <stan/math/prim/prob.hpp>

namespace bayesmix {
	int categorical_rng(Eigen::VectorXd probas, std::mt19937_64 rng,
		                  int start = 0) {
    return stan::math::categorical_rng(probas, rng) - (start+1);
  }
}

#endif  // DISTRIBUTIONS_HPP
