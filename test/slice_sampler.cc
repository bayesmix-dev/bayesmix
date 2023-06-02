#include "src/algorithms/slice_sampler.h"

#include <gtest/gtest.h>

#include <memory>
#include <stan/math/rev.hpp>
#include <vector>

#include "semihdp.pb.h"
#include "src/includes.h"
#include "src/utils/eigen_utils.h"

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

class SliceSamplerTest : public SliceSampler, public ::testing::Test {
 public:
  void setup(int num_components = 10) {
    Eigen::MatrixXd data = Eigen::MatrixXd::Ones(30, 1);
    auto hier = get_hierarchy();
    auto mix = get_mixing(num_components);
    bayesmix::AlgorithmParams algo_proto;
    bayesmix::read_proto_from_file(
        "../resources/benchmarks/default_algo_params.asciipb", &algo_proto);
    SliceSampler algo;
    read_params_from_proto(algo_proto);
    set_mixing(mix);
    set_hierarchy(hier);
    set_data(data);
  }
};

TEST_F(SliceSamplerTest, initialize) {
  setup();
  initialize();
  ASSERT_TRUE(true);
}

TEST_F(SliceSamplerTest, sample_weights) {
  setup(2);
  initialize();
  sample_slice();
  sample_weights();
  Eigen::VectorXd weights = mixing->get_mixing_weights(false, false);
  ASSERT_GT(weights(0), weights(2));
  ASSERT_GT(weights(1), weights(2));
  ASSERT_GT(mixing->get_num_components(), 2);
  ASSERT_LE(weights.sum(), 1.0);
}
