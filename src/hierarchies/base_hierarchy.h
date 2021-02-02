#ifndef BAYESMIX_HIERARCHIES_BASE_HIERARCHY_H_
#define BAYESMIX_HIERARCHIES_BASE_HIERARCHY_H_

#include <google/protobuf/message.h>

#include <Eigen/Dense>
#include <memory>
#include <random>
#include <set>
#include <stan/math/prim.hpp>

#include "marginal_state.pb.h"
#include "src/utils/rng.h"

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
  std::set<int> cluster_data_idx;
  int card = 0;
  double log_card = stan::math::NEGATIVE_INFTY;

  virtual void update_summary_statistics(const Eigen::VectorXd &datum,
                                         const Eigen::VectorXd &covariate,
                                         bool add) = 0;

  virtual double like_lpdf(const Eigen::RowVectorXd &datum, const Eigen::RowVectorXd &covariate) const = 0;
  virtual double marg_lpdf(const Eigen::RowVectorXd &datum, const Eigen::RowVectorXd &covariate, const bool posterior) const = 0;

 public:
  void add_datum(const int id, const Eigen::VectorXd &datum,
                 const Eigen::VectorXd &covariate);

  void remove_datum(const int id, const Eigen::VectorXd &datum,
                    const Eigen::VectorXd &covariate);

  virtual void clear_data() = 0;

  int get_card() const { return card; }
  double get_log_card() const { return log_card; }

  std::set<int> get_data_idx() { return cluster_data_idx; }

  virtual void initialize() = 0;
  //! Returns true if the hierarchy models multivariate data
  virtual bool is_multivariate() const = 0;
  //! Returns true if the hierarchy has covariates i.e. is a dependent model
  virtual bool is_dependent() const = 0;
  //! Returns true if the hierarchy is conjugate i.e. has a marginal lpdf
  virtual bool is_conjugate() const { return true; }

  virtual void update_hypers(
      const std::vector<bayesmix::MarginalState::ClusterState> &states) = 0;

  // DESTRUCTOR AND CONSTRUCTORS
  virtual ~BaseHierarchy() = default;
  BaseHierarchy() = default;
  virtual std::shared_ptr<BaseHierarchy> clone() const = 0;

  // EVALUATION FUNCTIONS FOR SINGLE POINTS
  double get_like_lpdf(const Eigen::RowVectorXd &datum,
                       const Eigen::RowVectorXd &covariate = Eigen::MatrixXd(0, 0)) const;
  double get_marg_lpdf(const Eigen::RowVectorXd &datum,
                       const Eigen::RowVectorXd &covariate = Eigen::MatrixXd(0, 0),
                       const bool posterior) const;

  // SAMPLING FUNCTIONS
  //! Generates new values for state from the centering prior distribution
  virtual void draw() = 0;
  //! Generates new values for state from the centering posterior distribution
  virtual void sample_given_data() = 0;
  virtual void sample_given_data(const Eigen::MatrixXd &data,
                                 const Eigen::MatrixXd &covariates) = 0;

  // GETTERS AND SETTERS
  virtual void write_state_to_proto(google::protobuf::Message *out) const = 0;
  virtual void write_hypers_to_proto(google::protobuf::Message *out) const = 0;
  virtual void set_state_from_proto(
      const google::protobuf::Message &state_) = 0;
  virtual void set_prior(const google::protobuf::Message &prior_) = 0;
  void set_card(const int card_) {
    card = card_;
    log_card = std::log(card_);
  }
  virtual std::string get_id() const = 0;
};

#endif  // BAYESMIX_HIERARCHIES_BASE_HIERARCHY_H_
