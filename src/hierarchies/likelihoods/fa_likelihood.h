#ifndef BAYESMIX_HIERARCHIES_LIKELIHOODS_FA_LIKELIHOOD_H_
#define BAYESMIX_HIERARCHIES_LIKELIHOODS_FA_LIKELIHOOD_H_

#include <google/protobuf/stubs/casts.h>

#include <memory>
#include <stan/math/prim.hpp>
#include <stan/math/rev.hpp>
#include <vector>

#include "algorithm_state.pb.h"
#include "base_likelihood.h"
#include "states/includes.h"

class FALikelihood : public BaseLikelihood<FALikelihood, State::FA> {
 public:
  FALikelihood() = default;
  ~FALikelihood() = default;
  bool is_multivariate() const override { return true; };
  bool is_dependent() const override { return false; };
  void set_state_from_proto(const google::protobuf::Message& state_,
                            bool update_card = true) override;
  void clear_summary_statistics() override;
  void set_dim(unsigned int dim_) {
    dim = dim_;
    clear_summary_statistics();
  };
  unsigned int get_dim() const { return dim; };
  Eigen::VectorXd get_data_sum() const { return data_sum; };

  std::shared_ptr<bayesmix::AlgorithmState::ClusterState> get_state_proto()
      const override;

  void compute_wood_factors(
      Eigen::MatrixXd& cov_wood, double& cov_logdet,
      const Eigen::MatrixXd& lambda,
      const Eigen::DiagonalMatrix<double, Eigen::Dynamic>& psi_inverse);

 protected:
  double compute_lpdf(const Eigen::RowVectorXd& datum) const override;
  void update_sum_stats(const Eigen::RowVectorXd& datum, bool add) override;

  unsigned int dim;
  Eigen::VectorXd data_sum;
};

#endif  // BAYESMIX_HIERARCHIES_LIKELIHOODS_FA_LIKELIHOOD_H_
