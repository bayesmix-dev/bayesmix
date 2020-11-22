#include <gtest/gtest.h>

#include <memory>

#include "../src/hierarchies/base_hierarchy.hpp"
#include "../src/mixings/dirichlet_mixing.hpp"

TEST(mixing, fixed_value) {
  DirichletMixing mix;
  bayesmix::DPPrior prior;
  double m = 2.0;
  prior.mutable_fixed_value()->set_value(m);
  double m_state = prior.fixed_value().value();
  ASSERT_FLOAT_EQ(m, m_state);
  mix.set_prior(prior);
  double m_mix = mix.get_totalmass();
  ASSERT_FLOAT_EQ(m, m_mix);

  std::vector<std::shared_ptr<BaseHierarchy>> hiers(100);
  unsigned int n_data = 1000;
  mix.update_hypers(hiers, n_data);
  double m_mix_after = mix.get_totalmass();
  ASSERT_FLOAT_EQ(m, m_mix_after);
}

TEST(mixing, gamma_prior) {
  DirichletMixing mix;
  bayesmix::DPPrior prior;
  double alpha = 1.0;
  double beta = 2.0;
  double m_prior = alpha / beta;
  prior.mutable_gamma_prior()->set_shape(alpha);
  prior.mutable_gamma_prior()->set_rate(beta);
  mix.set_prior(prior);
  double m_mix = mix.get_totalmass();
  ASSERT_FLOAT_EQ(m_prior, m_mix);

  std::vector<std::shared_ptr<BaseHierarchy>> hiers(100);
  unsigned int n_data = 1000;
  mix.update_hypers(hiers, n_data);
  double m_mix_after = mix.get_totalmass();

  std::cout << "[          ] after = " << m_mix_after << std::endl;
  ASSERT_TRUE(m_mix_after > m_mix);
}
