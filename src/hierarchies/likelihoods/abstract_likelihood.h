#ifndef BAYESMIX_HIERARCHIES_ABSTRACT_LIKELIHOOD_H_
#define BAYESMIX_HIERARCHIES_ABSTRACT_LIKELIHOOD_H_

#include <google/protobuf/message.h>

#include <Eigen/Dense>
#include <memory>

// #include <random>
// #include <set>
// #include <stan/math/prim.hpp>

#include "algorithm_state.pb.h"
// #include "hierarchy_id.pb.h"
// #include "src/utils/rng.h"

class AbstractLikelihood {
 public:
  virtual ~AbstractLikelihood() = default;

  // IMPLEMENTED in BaseLikelihood
  virtual std::shared_ptr<AbstractLikelihood> clone() const = 0;

  double lpdf(const Eigen::RowVectorXd &datum,
              const Eigen::RowVectorXd &covariate) const {
    if (is_dependent()) {
      return compute_lpdf(datum, covariate);
    } else {
      return compute_lpdf(datum);
    }
  }

  virtual Eigen::VectorXd lpdf_grid(
      const Eigen::MatrixXd &data,
      const Eigen::MatrixXd &covariates = Eigen::MatrixXd(0, 0)) const = 0;

  virtual bool is_multivariate() const = 0;

  virtual bool is_dependent() const = 0;

  virtual void set_state_from_proto(
      const google::protobuf::Message &state_) = 0;

  // IMPLEMENTED in BaseLikelihood
  virtual void write_state_to_proto(google::protobuf::Message *out) const = 0;

  // IMPLEMENTED in BaseLikelihood
  virtual void add_datum(
      const int id, const Eigen::RowVectorXd &datum,
      const Eigen::RowVectorXd &covariate = Eigen::RowVectorXd(0)) = 0;

  // IMPLEMENTED in BaseLikelihood
  virtual void remove_datum(
      const int id, const Eigen::RowVectorXd &datum,
      const Eigen::RowVectorXd &covariate = Eigen::RowVectorXd(0)) = 0;

  void update_summary_statistics(const Eigen::RowVectorXd &datum,
                                 const Eigen::RowVectorXd &covariate,
                                 bool add) {
    if (is_dependent()) {
      return update_sum_stats(datum, covariate, add);
    } else {
      return update_sum_stats(datum, add);
    }
  }

  virtual void clear_summary_statistics() = 0;

 protected:
  virtual std::shared_ptr<bayesmix::AlgorithmState::ClusterState>
  get_state_proto() const = 0;

  virtual double compute_lpdf(const Eigen::RowVectorXd &datum) const {
    if (is_dependent()) {
      throw std::runtime_error(
          "Cannot call this function from a dependent likelihood");
    } else {
      throw std::runtime_error("Not implemented");
    }
  }

  virtual double compute_lpdf(const Eigen::RowVectorXd &datum,
                              const Eigen::RowVectorXd &covariate) const {
    if (!is_dependent()) {
      throw std::runtime_error(
          "Cannot call this function from a non-dependent likelihood");
    } else {
      throw std::runtime_error("Not implemented");
    }
  }

  virtual void update_sum_stats(const Eigen::RowVectorXd &datum, bool add) {
    if (is_dependent()) {
      throw std::runtime_error(
          "Cannot call this function from a dependent hierarchy");
    } else {
      throw std::runtime_error("Not implemented");
    }
  }

  virtual void update_sum_stats(const Eigen::RowVectorXd &datum,
                                const Eigen::RowVectorXd &covariate,
                                bool add) {
    if (!is_dependent()) {
      throw std::runtime_error(
          "Cannot call this function from a non-dependent hierarchy");
    } else {
      throw std::runtime_error("Not implemented");
    }
  }
};

#endif  // BAYESMIX_HIERARCHIES_ABSTRACT_LIKELIHOOD_H_