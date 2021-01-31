#include <google/protobuf/stubs/casts.h>
#include <gtest/gtest.h>

#include <memory>

#include "marginal_state.pb.h"
#include "src/hierarchies/nnig_hierarchy.hpp"
#include "src/hierarchies/nnw_hierarchy.hpp"
#include "src/mixings/dirichlet_mixing.hpp"
#include "src/utils/proto_utils.hpp"

TEST(mixing, fixed_value) {
  DirichletMixing mix;
  bayesmix::DPPrior prior;
  double m = 2.0;
  prior.mutable_fixed_value()->set_totalmass(m);
  double m_state = prior.fixed_value().totalmass();
  ASSERT_DOUBLE_EQ(m, m_state);
  mix.set_prior(prior);
  double m_mix = mix.get_state().totalmass;
  ASSERT_DOUBLE_EQ(m, m_mix);

  std::vector<std::shared_ptr<BaseHierarchy>> hiers(100);
  unsigned int n_data = 1000;
  mix.update_state(hiers, n_data);
  double m_mix_after = mix.get_state().totalmass;
  ASSERT_DOUBLE_EQ(m, m_mix_after);
}

TEST(mixing, gamma_prior) {
  DirichletMixing mix;
  bayesmix::DPPrior prior;
  double alpha = 1.0;
  double beta = 2.0;
  double m_prior = alpha / beta;
  prior.mutable_gamma_prior()->mutable_totalmass_prior()->set_shape(alpha);
  prior.mutable_gamma_prior()->mutable_totalmass_prior()->set_rate(beta);
  mix.set_prior(prior);
  double m_mix = mix.get_state().totalmass;
  ASSERT_DOUBLE_EQ(m_prior, m_mix);

  std::vector<std::shared_ptr<BaseHierarchy>> hiers(100);
  unsigned int n_data = 1000;
  mix.update_state(hiers, n_data);
  double m_mix_after = mix.get_state().totalmass;

  std::cout << "             after = " << m_mix_after << std::endl;
  ASSERT_TRUE(m_mix_after > m_mix);
}

TEST(hierarchies, fixed_values) {
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

TEST(hierarchies, normal_mean_prior) {
  bayesmix::NNWPrior prior;
  bayesmix::NNWPrior prior_out;
  Eigen::Vector2d mu00;
  mu00 << 0.0, 0.0;
  auto ident = Eigen::Matrix2d::Identity();

  prior.mutable_normal_mean_prior()->set_var_scaling(0.1);
  bayesmix::to_proto(
      mu00,
      prior.mutable_normal_mean_prior()->mutable_mean_prior()->mutable_mean());
  bayesmix::to_proto(
      ident,
      prior.mutable_normal_mean_prior()->mutable_mean_prior()->mutable_var());
  prior.mutable_normal_mean_prior()->set_deg_free(2.0);
  bayesmix::to_proto(ident,
                     prior.mutable_normal_mean_prior()->mutable_scale());

  std::vector<bayesmix::MarginalState::ClusterState> states(4);
  for (int i = 0; i < states.size(); i++) {
    double mean = 9.0 + i;
    Eigen::Vector2d vec;
    vec << mean, mean;
    bayesmix::to_proto(vec,
                       states[i].mutable_multi_ls_state()->mutable_mean());
    bayesmix::to_proto(ident,
                       states[i].mutable_multi_ls_state()->mutable_prec());
  }

  NNWHierarchy hier;
  hier.set_prior(prior);
  hier.initialize();

  hier.update_hypers(states);
  hier.write_hypers_to_proto(&prior_out);
  Eigen::Vector2d mean_out =
      bayesmix::to_eigen(prior_out.fixed_values().mean());
  std::cout << "             after = " << mean_out(0) << " " << mean_out(1)
            << std::endl;
  assert(mu00(0) < mean_out(0) && mu00(1) < mean_out(1));
}
