#include <gtest/gtest.h>

#include <memory>

#include "../src/hierarchies/base_hierarchy.hpp"
#include "../src/mixings/dirichlet_mixing.hpp"

TEST(mixing, fixed_value) {
  DirichletMixing mix;
  bayesmix::DPState state;
  double m = 2.0;
  state.mutable_fixed_value()->set_value(m);
  double m_state = state.fixed_value().value();
  ASSERT_FLOAT_EQ(m, m_state);
  mix.set_state(state);
  double m_mix = mix.get_totalmass();
  ASSERT_FLOAT_EQ(m, m_mix);

  std::vector<std::shared_ptr<BaseHierarchy>> hiers(5);
  mix.update_hypers(hiers, 100);
  double m_mix_after = mix.get_totalmass();
  ASSERT_FLOAT_EQ(m, m_mix_after);
}

TEST(mixing, gamma_prior) {
  DirichletMixing mix;
  bayesmix::DPState state;
  double alpha = 1.0;
  double beta = 2.0;
  double m_prior = alpha / beta;
  state.mutable_gamma_prior()->set_alpha(alpha);
  state.mutable_gamma_prior()->set_beta(beta);
  mix.set_state(state);
  double m_mix = mix.get_totalmass();
  ASSERT_FLOAT_EQ(m_prior, m_mix);

  std::vector<std::shared_ptr<BaseHierarchy>> hiers(5);
  mix.update_hypers(hiers, 100);
  double m_mix_after = mix.get_totalmass();

  std::cout << "[          ] after = " << m_mix_after << std::endl;
  ASSERT_TRUE(m_mix != m_mix_after);
}
