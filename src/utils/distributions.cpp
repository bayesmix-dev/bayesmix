#include "distributions.hpp"

int bayesmix::categorical_rng(Eigen::VectorXd probas, std::mt19937_64 rng,
                              int start /*= 0*/) {
    return stan::math::categorical_rng(probas, rng) - (start+1);
}
