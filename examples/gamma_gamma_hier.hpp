#ifndef BAYESMIX_HIERARCHIES_GAMMAGAMMA_HIERARCHY_H_
#define BAYESMIX_HIERARCHIES_GAMMAGAMMA_HIERARCHY_H_

#include <google/protobuf/stubs/casts.h>

#include <Eigen/Dense>
#include <memory>
#include <vector>

#include <src/hierarchies/base_hierarchy.h>
#include <src/hierarchies/conjugate_hierarchy.h>


namespace GammaGamma {
//! Custom container for State values
struct State {
  double rate;
};

//! Custom container for Hyperparameters values
struct Hyperparams {
  double shape, rate_alpha, rate_beta;
};

}; // namespace GammaGamma

class GammaGammaHierarchy
    : public ConjugateHierarchy<GammaGammaHierarchy, GammaGamma::State, 
                                GammaGamma::Hyperparams, bayesmix::EmptyPrior> {
 public:
  GammaGammaHierarchy() = default;
  ~GammaGammaHierarchy() = default;

  double like_lpdf(const Eigen::RowVectorXd &datum,
                   const Eigen::RowVectorXd &covariate = Eigen::RowVectorXd(0)) const override {
        return stan::math::gamma_lpdf(datum(0), hypers->shape, state.rate);
    }

  GammaGamma::State draw(const GammaGamma::Hyperparams &params) { 
      return GammaGamma::State{stan::math::gamma_rng(
          params.rate_alpha, params.rate_beta, bayesmix::Rng::Instance().get())};
  }

  void update_summary_statistics(const Eigen::RowVectorXd &datum,
                                 const Eigen::RowVectorXd &covariate,
                                 bool add) {
        if (add) { data_sum += datum(0); ndata += 1; } 
        else { data_sum -= datum(0); ndata -= 1; }
  }

  //! Computes and return posterior hypers given data currently in this cluster
  GammaGamma::Hyperparams get_posterior_parameters() {
      GammaGamma::Hyperparams out;
      out.shape = hypers->shape;
      out.rate_alpha = hypers->rate_alpha + hypers->shape * ndata;
      out.rate_beta = hypers->rate_beta + data_sum;
      return out;
  }

  void initialize_state() {state.rate = hypers->rate_alpha / hypers->rate_beta;}

  void set_hypers(double shape, double rate_alpha, double rate_beta) {
    this->shape = shape;
    this->rate_alpha = rate_alpha;
    this->rate_beta = rate_beta; 
  }

  void initialize_hypers() { 
      hypers->shape = shape;
      hypers->rate_alpha = rate_alpha;
      hypers->rate_beta = rate_beta; 
    }

  void update_hypers(const std::vector<bayesmix::AlgorithmState::ClusterState>
                     &states) override { return; }

  //! Removes every data point from this cluster
  void clear_data() {
      data_sum = 0;
      ndata = 0;
  }

  bool is_multivariate() const override { return false; }

  void set_state_from_proto(const google::protobuf::Message &state_) override {
      auto &statecast = google::protobuf::internal::down_cast<
        const bayesmix::AlgorithmState::ClusterState &>(state_);

    state.rate = statecast.general_state().data()[0];
  }

  void write_state_to_proto(google::protobuf::Message *out) const override {
      bayesmix::Vector state_;
      state_.mutable_data()->Add(state.rate);
      auto *out_cast = google::protobuf::internal::down_cast<
        bayesmix::AlgorithmState::ClusterState *>(out);
      out_cast->mutable_general_state()->CopyFrom(state_);
      out_cast->set_cardinality(card);
      std::cout << "card: " << card << std::endl;
  }

  void write_hypers_to_proto(google::protobuf::Message *out) const override { return; }

  bayesmix::HierarchyId get_id() const override {
    return bayesmix::HierarchyId::UNKNOWN_HIERARCHY;
  }

  double marg_lpdf(
      const GammaGamma::Hyperparams &params, const Eigen::RowVectorXd &datum,
      const Eigen::RowVectorXd &covariate = Eigen::RowVectorXd(0)) const {
        throw std::runtime_error("Not implemented");
        return 0;
    }


 protected:
  double data_sum = 0;
  int ndata = 0;

  double shape, rate_alpha, rate_beta;
};

#endif  // BAYESMIX_HIERARCHIES_GAMMAGAMMA_HIERARCHY_H_
