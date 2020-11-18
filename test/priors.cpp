#include <gtest/gtest.h>

//#include <Eigen/Dense>
// #include <stan/math/prim/fun.hpp>
// #include <stan/math/prim/prob.hpp>

//#include "../proto/cpp/marginal_state.pb.h"
//#include "../src/hierarchies/nnig_hierarchy.hpp"
#include "../src/mixings/dirichlet_mixing.hpp"


TEST(mixing, fixed_value) {
  DirichletMixing mix;
  bayesmix::DPState state;
  double m = 2.0;
  state.mutable_fixed_value()->set_value(m);
  double m_state = state.mutable_fixed_value()->value();
  ASSERT_FLOAT_EQ(m, m_state);
  mix.set_state(&state);
  double m_mix = mix.get_totalmass();
  ASSERT_FLOAT_EQ(m, m_mix);
}
