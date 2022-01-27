#include <gtest/gtest.h>

#include <stan/math/rev.hpp>

class fbase {
 public:
  virtual double lpdf(const Eigen::VectorXd& x) = 0;
};

class f1 : public fbase {
 protected:
  double y;

 public:
  f1() = default;
  f1(double y) : y(y) {}

  template <typename T>
  T lpdf(const Eigen::Matrix<T, Eigen::Dynamic, 1>& x) const {
    return 0.5 * x.squaredNorm() * y;
  }

  double lpdf(const Eigen::Matrix<double, Eigen::Dynamic, 1>& x) {
    return this->lpdf<double>(x);
  }
};

template <class F>
struct target_lpdf {
  F f;

  template <typename T>
  T operator()(const Eigen::Matrix<T, Eigen::Dynamic, 1>& x) const {
    return f.lpdf(x);
  }
};

TEST(gradient, quadratic_function) {
  Eigen::VectorXd out;
  Eigen::VectorXd x(5);
  x << 1.0, 2.0, 3.0, 4.0, 5.0;
  target_lpdf<f1> target_function;
  target_function.f = f1(5.0);
  double y;
  stan::math::gradient(target_function, x, y, out);

  for (int i = 0; i < 5; i++) {
    ASSERT_DOUBLE_EQ(out(i), 5 * x(i));
  }
}
