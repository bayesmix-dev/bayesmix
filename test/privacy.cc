#include <gtest/gtest.h>

#include <memory>
#include <stan/math/rev.hpp>
#include <vector>

#include "src/includes.h"
#include "src/privacy/algorithms/private_conditional.h"
#include "src/privacy/channels/base_channel.h"
#include "src/privacy/channels/laplace_channel.h"

// TEST(Algo, can_create) {
//     auto hier = std::make_shared<NNIGHierarchy>();
//     bayesmix::NNIGPrior hier_prior;
//     hier_prior.mutable_fixed_values()->set_mean(0.0);
//     hier_prior.mutable_fixed_values()->set_var_scaling(0.1);
//     hier_prior.mutable_fixed_values()->set_shape(2.0);
//     hier_prior.mutable_fixed_values()->set_scale(2.0);
//     hier->get_mutable_prior()->CopyFrom(hier_prior);
//     hier->initialize();

//     auto mix = std::make_shared<TruncatedSBMixing>();
//     bayesmix::TruncSBPrior prior;
//     prior.mutable_dp_prior()->set_totalmass(2.0);
//     prior.set_num_components(10);
//     mix->get_mutable_prior()->CopyFrom(prior);
//     mix->initialize();

//     std::shared_ptr<LaplaceChannel> channel(new LaplaceChannel(2.0));

//     PrivateConditionalAlgorithm<BlockedGibbsAlgorithm> private_algo;
//     private_algo.set_mixing(mix);
//     private_algo.set_hierarchy(hier);
//     private_algo.set_channel(channel);

//     Eigen::MatrixXd private_data = Eigen::MatrixXd::Ones(30, 1);
//     Eigen::MatrixXd public_data = channel->sanitize(private_data);
//     private_algo.set_public_data(public_data);

//     ASSERT_GT(private_algo.get_acceptance_rate(), 0.0);
//     ASSERT_TRUE(true);
// }

class AlgoTest : public PrivateConditionalAlgorithm<BlockedGibbsAlgorithm>,
                 public ::testing::Test {
 public:
  std::shared_ptr<AbstractHierarchy> get_hierarchy() {
    auto hier = std::make_shared<NNIGHierarchy>();
    bayesmix::NNIGPrior hier_prior;
    hier_prior.mutable_fixed_values()->set_mean(0.0);
    hier_prior.mutable_fixed_values()->set_var_scaling(0.1);
    hier_prior.mutable_fixed_values()->set_shape(2.0);
    hier_prior.mutable_fixed_values()->set_scale(2.0);
    hier->get_mutable_prior()->CopyFrom(hier_prior);
    hier->initialize();
    return hier;
  }

  std::shared_ptr<TruncatedSBMixing> get_mixing(int num_components) {
    auto mix = std::make_shared<TruncatedSBMixing>();
    bayesmix::TruncSBPrior prior;
    prior.mutable_dp_prior()->set_totalmass(2.0);
    prior.set_num_components(num_components);
    mix->get_mutable_prior()->CopyFrom(prior);
    mix->initialize();
    return mix;
  }

  std::shared_ptr<LaplaceChannel> get_channel(double eps) {
    std::shared_ptr<LaplaceChannel> channel(new LaplaceChannel(eps));
    return channel;
  }

  void setup(int num_components = 10) {
    auto hier = get_hierarchy();
    auto mix = get_mixing(num_components);
    bayesmix::AlgorithmParams algo_proto;
    bayesmix::read_proto_from_file(
        "../resources/benchmarks/default_algo_params.asciipb", &algo_proto);
    SliceSampler algo;
    read_params_from_proto(algo_proto);
    set_mixing(mix);
    set_hierarchy(hier);

    set_channel(get_channel(2.0));
    Eigen::MatrixXd private_data = Eigen::MatrixXd::Ones(30, 1);
    Eigen::MatrixXd public_data = privacy_channel->sanitize(private_data);
    set_public_data(public_data);
  }
};

TEST_F(AlgoTest, initialize) {
  setup();
  initialize();
  ASSERT_TRUE(true);
}

TEST_F(AlgoTest, update_private_data) {
  setup();
  initialize();
  update_private_data();
  ASSERT_GT(get_acceptance_rate(), 0.0);
  update_private_data();
  ASSERT_GT(get_acceptance_rate(), 0.0);
}

TEST_F(AlgoTest, sample_uniqs) {
  setup();
  initialize();
  sample_unique_values();
  for (int h = 0; h < unique_values.size(); h++) {
    std::cout << unique_values[h]->get_state_proto()->DebugString()
              << std::endl;
  }
}

TEST_F(AlgoTest, step) {
  setup();
  initialize();
  step();
  ASSERT_GT(get_acceptance_rate(), 0.0);
}
