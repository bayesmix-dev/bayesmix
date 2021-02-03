#include "base_hierarchy.h"

#include <Eigen/Dense>
#include <cassert>
#include <set>
#include <stan/math/prim.hpp>

void BaseHierarchy::add_datum(
    const int id, const bool save_params, const Eigen::VectorXd &datum,
    const Eigen::VectorXd &covariate /*= Eigen::VectorXd(0)*/) {
  auto it = cluster_data_idx.find(id);
  assert(it == cluster_data_idx.end());
  card += 1;
  log_card = std::log(card);
  update_summary_statistics(datum, covariate, true);
  cluster_data_idx.insert(id);
  if (save_params) {
    save_posterior_hypers();
  }
}

void BaseHierarchy::remove_datum(
    const int id, const bool save_params, const Eigen::VectorXd &datum,
    const Eigen::VectorXd &covariate /* = Eigen::VectorXd(0)*/) {
  update_summary_statistics(datum, covariate, false);
  card -= 1;
  log_card = (card == 0) ? stan::math::NEGATIVE_INFTY : std::log(card);
  auto it = cluster_data_idx.find(id);
  assert(it != cluster_data_idx.end());
  cluster_data_idx.erase(it);
  if (save_params) {
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
