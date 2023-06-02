#ifndef BAYESMIX_ALGORITHMS_SLICE_SAMPLER_H_
#define BAYESMIX_ALGORITHMS_SLICE_SAMPLER_H_

#include "algorithm_id.pb.h"
#include "conditional_algorithm.h"
#include "src/mixings/truncated_sb_mixing.h"

//! This class implement the original slice sampler from [1].
//!
//! [1] Walker, S.G.: Sampling the Dirichlet mixture model with slices.
//!     Commun. Stat., Simul. Comput. 36, 45–54 (2007)

class SliceSampler : public ConditionalAlgorithm {
 public:
  SliceSampler() = default;
  ~SliceSampler() = default;

  void initialize() override;

  void step() override;

  bayesmix::AlgorithmId get_id() const override {
    return bayesmix::AlgorithmId::Slice;
  }

  std::shared_ptr<BaseAlgorithm> clone() const override {
    auto out = std::make_shared<SliceSampler>(*this);
    out->set_mixing(mixing->clone());
    out->set_hierarchy(unique_values[0]->deep_clone());
    return out;
  }

  double sample_truncated_beta(double a, double b, double l, double u);

  void sample_slice();

 protected:
  void print_startup_message() const override;

  void sample_allocations() override;

  void sample_unique_values() override;

  void sample_weights() override;

  Eigen::VectorXd slice_u;

  std::shared_ptr<TruncatedSBMixing> mixing;
};

#endif  // BAYESMIX_ALGORITHMS_SLICE_SAMPLER_H_
