#include <google/protobuf/stubs/casts.h>
#include <gtest/gtest.h>

#include <memory>

#include "../proto/cpp/marginal_state.pb.h"
#include "../src/hierarchies/nnig_hierarchy.hpp"
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

TEST(hierarchies, priors) {
  bayesmix::NNIGPrior prior;
  bayesmix::NNIGPrior prior_out;
  prior.mutable_fixed_values()->set_mean(5.0);
  prior.mutable_fixed_values()->set_var_scaling(0.1);
  prior.mutable_fixed_values()->set_shape(2.0);
  prior.mutable_fixed_values()->set_scale(2.0);

  auto hier = std::make_shared<NNIGHierarchy>();
  hier->set_prior(prior);
  hier->initialize();

  std::vector<std::shared_ptr<BaseHierarchy>> unique_values;
  std::vector<bayesmix::MarginalState::ClusterState> states;

  // Check equality before update
  unique_values.push_back(hier);
  for (size_t i = 1; i < 4; i++) {
    unique_values.push_back(hier->clone());
    unique_values[i]->write_hypers_to_proto(&prior_out);
    ASSERT_EQ(prior.DebugString(), prior_out.DebugString());
  }

  // Check equality after update
  unique_values[0]->update_hypers(states);
  unique_values[0]->write_hypers_to_proto(&prior);
  for (size_t i = 1; i < 4; i++) {
    unique_values[i]->write_hypers_to_proto(&prior_out);
    ASSERT_EQ(prior.DebugString(), prior_out.DebugString());
  }
}
