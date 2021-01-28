#ifndef BAYESMIX_HIERARCHIES_DEPENDENT_HIERARCHY_HPP_
#define BAYESMIX_HIERARCHIES_DEPENDENT_HIERARCHY_HPP_

#include <google/protobuf/message.h>

#include <Eigen/Dense>
#include <memory>
#include <random>
#include <stan/math/prim.hpp>

#include "marginal_state.pb.h"
#include "../utils/rng.hpp"
#include "base_hierarchy.hpp"

class DependentHierarchy : public BaseHierarchy {
 public:
  void add_datum(const int id, const Eigen::VectorXd &datum) override {
    return;
  }

  void remove_datum(const int id, const Eigen::VectorXd &datum) override {
    return;
  }

  void add_datum(const int id, const Eigen::VectorXd &datum,
                 const Eigen::VectorXd &covariate);

  void remove_datum(const int id, const Eigen::VectorXd &datum,
                    const Eigen::VectorXd &covariate);

  //! Returns true if the hierarchy has covariates i.e. is a dependent model
  bool is_dependent() const override { return true; }

  void update_summary_statistics(const Eigen::VectorXd &datum,
                                 bool add) override {
    return;
  }

  virtual void update_summary_statistics(const Eigen::VectorXd &datum,
                                         const Eigen::VectorXd &covariate,
                                         bool add) = 0;

  void sample_given_data(const Eigen::MatrixXd &data) override { return; }
  virtual void sample_given_data(const Eigen::MatrixXd &data,
                                 const Eigen::MatrixXd &covariates) = 0;

  // DESTRUCTOR AND CONSTRUCTORS
  virtual ~DependentHierarchy() = default;
  DependentHierarchy() = default;

  // DEPRECATED EVALUATION FUNCTIONS // TODO give warning message?
  double like_lpdf(const Eigen::RowVectorXd &datum) const override {
    return 0;
  };
  Eigen::VectorXd like_lpdf_grid(const Eigen::MatrixXd &data) const override {
    return Eigen::VectorXd::Zero(1);
  };
  double marg_lpdf(const Eigen::RowVectorXd &datum) const override {
    return 0;
  };
  Eigen::VectorXd marg_lpdf_grid(const Eigen::MatrixXd &data) const override {
    return Eigen::VectorXd::Zero(1);
  };

  // EVALUATION FUNCTIONS
  //! Evaluates the log-likelihood of data in a single point
  virtual double like_lpdf(const Eigen::RowVectorXd &datum,
                           const Eigen::RowVectorXd &covariate) const = 0;
  //! Evaluates the log-likelihood of data in the given points
  virtual Eigen::VectorXd like_lpdf_grid(
      const Eigen::MatrixXd &data,
      const Eigen::MatrixXd &covariates) const = 0;
  //! Evaluates the log-marginal distribution of data in a single point
  virtual double marg_lpdf(const Eigen::RowVectorXd &datum,
                           const Eigen::RowVectorXd &covariate) const = 0;
  //! Evaluates the log-marginal distribution of data in the given points
  virtual Eigen::VectorXd marg_lpdf_grid(
      const Eigen::MatrixXd &data,
      const Eigen::MatrixXd &covariates) const = 0;
};

#endif  // BAYESMIX_HIERARCHIES_DEPENDENT_HIERARCHY_HPP_
