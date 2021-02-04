#include "base_hierarchy.h"

#include <Eigen/Dense>
#include <cassert>
#include <set>
#include <stan/math/prim.hpp>

//! \param id            Index of the given data point
//! \param datum         Data point
//! \param update_params If true, updates the stored posterior hyperparameters
//!                      (for algorithms such as Neal3)
//! \param covariate     Covariate value for dependent hierarchies, leave the
//!                      default value otherwise
void BaseHierarchy::add_datum(
    const int id, const Eigen::VectorXd &datum,
    const bool update_params /*= false*/,
    const Eigen::VectorXd &covariate /*= Eigen::VectorXd(0)*/) {
  auto it = cluster_data_idx.find(id);
  assert(it == cluster_data_idx.end());
  card += 1;
  log_card = std::log(card);
  update_summary_statistics(datum, covariate, true);
  cluster_data_idx.insert(id);
  if (update_params) {
    save_posterior_hypers();
  }
}

//! \param id            Index of the given data point
//! \param datum         Data point
//! \param update_params If true, updates the stored posterior hyperparameters
//!                      (for algorithms such as Neal3)
//! \param covariate     Covariate value for dependent hierarchies, leave the
//!                      default value otherwise
void BaseHierarchy::remove_datum(
    const int id, const Eigen::VectorXd &datum,
    const bool update_params /*= false*/,
    const Eigen::VectorXd &covariate /* = Eigen::VectorXd(0)*/) {
  update_summary_statistics(datum, covariate, false);
  card -= 1;
  log_card = (card == 0) ? stan::math::NEGATIVE_INFTY : std::log(card);
  auto it = cluster_data_idx.find(id);
  assert(it != cluster_data_idx.end());
  cluster_data_idx.erase(it);
  if (update_params) {
    save_posterior_hypers();
  }
}

Eigen::VectorXd BaseHierarchy::like_lpdf_grid(
    const Eigen::MatrixXd &data,
    const Eigen::MatrixXd &covariates /*= Eigen::MatrixXd(0, 0)*/) const {
  Eigen::VectorXd lpdf(data.rows());
  if (covariates.cols() == 0) {
    for (int i = 0; i < data.rows(); i++) {
      // Pass null value as covariate
      lpdf(i) = like_lpdf(data.row(i), Eigen::RowVectorXd(0));
    }
  } else {
    for (int i = 0; i < data.rows(); i++) {
      lpdf(i) = like_lpdf(data.row(i), covariates.row(i));
    }
  }
  return lpdf;
}

Eigen::VectorXd BaseHierarchy::marg_lpdf_grid(
    const bool posterior, const Eigen::MatrixXd &data,
    const Eigen::MatrixXd &covariates /*= Eigen::MatrixXd(0, 0)*/) const {
  Eigen::VectorXd lpdf(data.rows());
  if (covariates.cols() == 0) {
    for (int i = 0; i < data.rows(); i++) {
      // Pass null value as covariate
      lpdf(i) = marg_lpdf(posterior, data.row(i), Eigen::RowVectorXd(0));
    }
  } else {
    for (int i = 0; i < data.rows(); i++) {
      lpdf(i) = marg_lpdf(posterior, data.row(i), covariates.row(i));
    }
  }
  return lpdf;
}

void BaseHierarchy::sample_given_data(
    const Eigen::MatrixXd &data,
    const Eigen::MatrixXd &covariates /*= Eigen::MatrixXd(0, 0)*/) {
  clear_data();
  if (covariates == Eigen::MatrixXd(0, 0)) {
    for (int i = 0; i < data.rows(); i++)
      add_datum(i, data.row(i), false, Eigen::RowVectorXd(0));
  } else {
    for (int i = 0; i < data.rows(); i++)
      add_datum(i, data.row(i), false, covariates.row(i));
  }
  sample_given_data();
}
