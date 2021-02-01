#ifndef BAYESMIX_HIERARCHIES_DEPENDENT_HIERARCHY_H_
#define BAYESMIX_HIERARCHIES_DEPENDENT_HIERARCHY_H_

#include <google/protobuf/message.h>

#include <Eigen/Dense>
#include <memory>
#include <random>
#include <stan/math/prim.hpp>

#include "base_hierarchy.h"
#include "marginal_state.pb.h"
#include "src/utils/rng.h"

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
};

#endif  // BAYESMIX_HIERARCHIES_DEPENDENT_HIERARCHY_H_
