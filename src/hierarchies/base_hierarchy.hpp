#ifndef BAYESMIX_HIERARCHIES_BASE_HIERARCHY_HPP_
#define BAYESMIX_HIERARCHIES_BASE_HIERARCHY_HPP_

#include <google/protobuf/message.h>

#include <Eigen/Dense>
#include <memory>
#include <random>
#include <map>
#include <stan/math/prim.hpp>

#include "../../proto/cpp/marginal_state.pb.h"
#include "../utils/rng.hpp"

//! Abstract base template class for a hierarchy object.

//! This template class represents a hierarchy object in a generic iterative
//! BNP algorithm, that is, a single set of unique values with their own prior
//! distribution attached to it. These values are part of the Markov chain's
//! state chain (which includes multiple hierarchies) and are simply referred
//! to as the state of the hierarchy. This object also corresponds to a single
//! cluster in the algorithm, in the sense that its state is the set of
//! parameters for the distribution of the data points that belong to it. Since
//! the prior distribution for the state is often the same across multiple
//! different hierarchies, the hyperparameters object is accessed via a shared
//! pointer. Lastly, any hierarchy that inherits from this class contains
//! multiple ways of updating the state, either via prior or posterior
//! distributions, and of evaluating the distribution of the data, either its
//! likelihood (whose parameters are the state) or its marginal distribution.

class BaseHierarchy {
 protected:
  // map: data_id -> datum
  std::map<int, std::shared_ptr<Eigen::EigenXd>> data_map;
  std::map<int, std::shared_ptr<Eigen::EigenXd>> covariate_map;
  int card = 0;
  double log_card = stan::math::NEGATIVE_INFTY;

  virtual void update_summary_statistics(const Eigen::VectorXd &datum,
                                         bool add) = 0;

  void set_card(int card_) {
    card = card_;
    log_card = std::log(card_);
  }

 public:
  void add_datum(const int id, const Eigen::VectorXd *datum,
    const Eigen::VectorXd *cov = nullptr) {
    auto it = data_map.find(id);
    assert(it == data_map.end());
    card += 1;
    log_card = std::log(card);
    data_map[id] = datum;
    covariate_map[id] = cov;
    update_summary_statistics(datum, true);
    data_map.insert(id);
  }

  void remove_datum(const int id) {
    update_summary_statistics(data_map[id], false);
    card -= 1;
    log_card = (card == 0) ? stan::math::NEGATIVE_INFTY : std::log(card);
    // Datum
    auto it = data_map.find(id);
    assert(it != data_map.end());
    data_map.erase(it);
    // Covariate
    auto it2 = covariates_map.find(id);
    assert(it2 != covariates_map.end());
    covariates_map.erase(it2);
  }

  int get_card() const { return card; }
  double get_log_card() const { return log_card; }

  std::set<int> get_data_idx() {}  // TODO we can implement it if needed

  virtual void initialize() = 0;
  //! Returns true if the hierarchy models multivariate data
  virtual bool is_multivariate() const = 0;

  virtual void update_hypers(
      const std::vector<bayesmix::MarginalState::ClusterState> &states) = 0;

  // DESTRUCTOR AND CONSTRUCTORS
  virtual ~BaseHierarchy() = default;
  BaseHierarchy() = default;
  virtual std::shared_ptr<BaseHierarchy> clone() const = 0;

  // EVALUATION FUNCTIONS
  //! Evaluates the log-likelihood of data in a single point
  virtual double like_lpdf(const int idx) const = 0;
  //! Evaluates the log-likelihood of data in the given points
  virtual Eigen::VectorXd like_lpdf_grid(
      const std::vector<int> &idxs) const = 0;
  //! Evaluates the log-marginal distribution of data in a single point
  virtual double marg_lpdf(const int idx) const = 0;
  //! Evaluates the log-marginal distribution of data in the given points
  virtual Eigen::VectorXd marg_lpdf_grid(
      const std::vector<int> &idxs) const = 0;

  // SAMPLING FUNCTIONS
  //! Generates new values for state from the centering prior distribution
  virtual void draw() = 0;
  //! Generates new values for state from the centering posterior distribution
  virtual void sample_given_data() = 0;
  virtual void sample_given_data(const Eigen::MatrixXd &data) = 0;  // TODO
                                                                    // needed?

  // GETTERS AND SETTERS
  virtual void write_state_to_proto(google::protobuf::Message *out) const = 0;
  virtual void write_hypers_to_proto(google::protobuf::Message *out) const = 0;

  //! \param state_ State value to set
  //! \param check  If true, a state validity check occurs after assignment
  virtual void set_state_from_proto(
      const google::protobuf::Message &state_) = 0;

  virtual void set_prior(const google::protobuf::Message &prior_) = 0;

  virtual std::string get_id() const = 0;
};

#endif  // BAYESMIX_HIERARCHIES_BASE_HIERARCHY_HPP_
