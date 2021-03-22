#include <iostream>

#include "lib/math/lib/eigen_3.3.9/Eigen/Dense"
#include "lib/math/stan/math/prim.hpp"

// compile with:
// g++ -D_REENTRANT -I{lib/math,lib/math/lib/eigen_3.3.9,lib/math/lib/boost_1.72.0} stan.cc

int main() {
  using namespace Eigen;
  using namespace stan::math;
  VectorXd mu(2); mu << 0, 0;
  VectorXd x(2); x << 1, 1;
  double v = 3;
  double dim = 2;
  auto mat = 2 * MatrixXd::Identity(2, 2);
  auto mat_inv = 0.5 * MatrixXd::Identity(2, 2);
  double logdet = log(4.0); // log(mat.diagonal().array()).sum();
  double lpdf = -0.5*(v + dim) *
                log(1 + ((x-mu).transpose() * mat * (x-mu) / v)[0]);
  double logZ = stan::math::lgamma(0.5*(v + dim)) * 0.5 * logdet
             - stan::math::lgamma(0.5*v)
             - 0.5 * dim * log(v * 0.5 * TWO_PI);
  lpdf -= logZ;
  std::cout << lpdf << std::endl;
  std::cout << multi_student_t_lpdf(x, v, mu, mat) << std::endl;
  std::cout << multi_student_t_lpdf(x, v, mu, mat_inv) << std::endl;
  return 0;
}
