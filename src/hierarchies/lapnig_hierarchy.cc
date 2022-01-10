#include "lapnig_hierarchy.h"
#include <google/protobuf/stubs/casts.h>

#include <Eigen/Dense>
#include <stan/math/prim/prob.hpp>
#include <vector>
#include <cmath>
#include <algorithm>

#include "algorithm_state.pb.h"
#include "hierarchy_prior.pb.h"
#include "ls_state.pb.h"
#include "src/utils/rng.h"

unsigned int LapNIGHierarchy::accepted_ = 0;
unsigned int LapNIGHierarchy::iter_ = 0;


void LapNIGHierarchy::set_state_from_proto(
    const google::protobuf::Message &state_) {
  auto &statecast = downcast_state(state_);
  state.mean = statecast.uni_ls_state().mean();
  state.scale = statecast.uni_ls_state().var(); // ??? changed from scale to var
  set_card(statecast.cardinality());
}

void LapNIGHierarchy::set_hypers_from_proto(
    const google::protobuf::Message &hypers_) {
  auto &hyperscast = downcast_hypers(hypers_).lapnig_state();
  hypers->mean = hyperscast.mean();
  hypers->var = hyperscast.var();
  hypers->scale = hyperscast.scale();
  hypers->shape = hyperscast.shape();
}

double LapNIGHierarchy::like_lpdf(const Eigen::RowVectorXd &datum) const {
  return stan::math::double_exponential_lpdf(datum(0), state.mean,
                                             state.scale);
}

std::shared_ptr<bayesmix::AlgorithmState::ClusterState>
LapNIGHierarchy::get_state_proto() const {
  bayesmix::UniLSState state_;
  state_.set_mean(state.mean);
  state_.set_var(state.scale);

  auto out = std::make_shared<bayesmix::AlgorithmState::ClusterState>();
  out->mutable_uni_ls_state()->CopyFrom(state_);
  return out;
}

std::shared_ptr<bayesmix::AlgorithmState::HierarchyHypers>
LapNIGHierarchy::get_hypers_proto() const {
  bayesmix::LapNIGState hypers_;
  hypers_.set_mean(hypers->mean);
  hypers_.set_var(hypers->var);
  hypers_.set_shape(hypers->shape);
  hypers_.set_scale(hypers->scale);

  auto out = std::make_shared<bayesmix::AlgorithmState::HierarchyHypers>();
  out->mutable_lapnig_state()->CopyFrom(hypers_);
  return out;
}

void LapNIGHierarchy::clear_summary_statistics() {
  cluster_data_values.clear();
}

void LapNIGHierarchy::initialize_hypers() {
  if (prior->has_fixed_values()) {
    // Set values
    hypers->mean = prior->fixed_values().mean();
    hypers->var = prior->fixed_values().var();
    hypers->shape = prior->fixed_values().shape();
    hypers->scale = prior->fixed_values().scale();
    // Check validity
    if (hypers->var <= 0) {
      throw std::invalid_argument("Variance-scaling parameter must be > 0");
    }
    if (hypers->shape <= 0) {
      throw std::invalid_argument("Shape parameter must be > 0");
    }
    if (hypers->scale <= 0) {
      throw std::invalid_argument("Scale parameter must be > 0");
    }
  }
  else {
    throw std::invalid_argument("Unrecognized hierarchy prior ghjk");
  }
}

void LapNIGHierarchy::initialize_state() {
  state.mean = hypers->mean;
  state.scale = 1./hypers->scale / (hypers->shape + 1);
}

void LapNIGHierarchy::update_hypers(
    const std::vector<bayesmix::AlgorithmState::ClusterState> &states) {
  auto &rng = bayesmix::Rng::Instance().get();

  if (prior->has_fixed_values()) {
    return;
  }

  else {
    throw std::invalid_argument("Unrecognized hierarchy prior asdf");
  }
}

LapNIG::State LapNIGHierarchy::draw(const LapNIG::Hyperparams &params) {
  auto &rng = bayesmix::Rng::Instance().get();
  LapNIG::State out{};
  out.scale = stan::math::inv_gamma_rng(params.shape, 1./params.scale, rng);
  out.mean = stan::math::normal_rng(params.mean,
                                    sqrt(params.var), rng);
  return out;
}

void LapNIGHierarchy::update_summary_statistics(const Eigen::RowVectorXd &datum,
                                              bool add) {
  if (add) {

    cluster_data_values.push_back(datum); // add the datum to cluster_data_values, needed to compute the full_cond

  } else {

    auto it = std::find(cluster_data_values.begin(), cluster_data_values.end(), datum);
    cluster_data_values.erase(it);

  }
}


void LapNIGHierarchy::sample_full_cond(bool update_params) {

  if (this->card == 0) {
      // No posterior update possible
      this->sample_prior();
  } else {
      // Number of iterations to compute the acceptance rate
      ++iter_;

      // Random generator
      auto &rng = bayesmix::Rng::Instance().get();

      // Candidate mean and candidate log_scale
      double candidate_mean_step_var{2*hypers->var}; // 0.5 * Variance of the mean distribution

      double candidate_log_scale_step_var{10*boost::math::trigamma(2*hypers->shape)};  // 0.5 * Variance of log-gamma distribution

      double candidate_mean = state.mean + stan::math::normal_rng(0,sqrt(candidate_mean_step_var), rng);

      double candidate_log_scale = std::log(state.scale) +
                                   stan::math::normal_rng(0,sqrt(candidate_log_scale_step_var), rng);

      double candidate_scale = std::exp(candidate_log_scale);

      //MH step

      double pi_current, pi_candidate; // posterior of current state and candidate state

      double candidate_sum = 0; // Sum of absolute values of data - candidate_mean
      for(auto & elem: cluster_data_values){
        candidate_sum += std::abs(elem(0,0) - candidate_mean); // MODIFY IN THE FUTURE FOR MULTIVARIATE CASE!!!
      }

      pi_candidate = 0.5/candidate_scale *
                std::exp(-0.5 / candidate_scale * candidate_sum) *
                std::exp(-0.5 / hypers->var / hypers->var *
                         (candidate_mean - hypers->mean) *
                         (candidate_mean - hypers->mean)) *
                std::pow(1./ candidate_scale, hypers->shape) *
                std::exp(-1./hypers->scale / candidate_scale);

      double current_sum = 0; // Sum of absolute values of data - candidate_mean

      double current_mean = state.mean;
      double current_scale = state.scale;

      for(auto & elem: cluster_data_values){
        current_sum += std::abs(elem(0,0) - current_mean); // MODIFY IN THE FUTURE FOR MULTIVARIATE CASE!!!
      }
      pi_current = 0.5/current_scale * std::exp(-0.5/current_scale*current_sum) *
                std::exp(-0.5/hypers->var/hypers->var
                         *(current_mean-hypers->mean)*(current_mean-hypers->mean))*
                std::pow(1./current_scale,hypers->shape) * std::exp(-1./hypers->scale/current_scale);

      double alpha;
      if(pi_current!=0) {
        alpha = std::min<double>((pi_candidate/pi_current), 1);
      }
      else{
        alpha = 1;
      }

      //std::cout << "alpha: " << alpha << std::endl;

      bool accept = stan::math::bernoulli_rng(alpha,rng);

      if(accept){
        state.mean = candidate_mean;
        state.scale = candidate_scale;
        // To compute the acceptance rate
        ++accepted_;
      }
      else{
      }
  }
}
