#include "lin_dep_normal_hierarchy.hpp"

void LinDepNormalHierarchy::initialize() {
  int dim = 1;                                    // TODO set dimension somehow
  state.parameters = Eigen::VectorXd::Zero(dim);  // or other starting values

  // ... TODO
}

double LinDepNormalHierarchy::like_lpdf(
    const Eigen::RowVectorXd &datum,
    const Eigen::RowVectorXd &covariate) const {
  return 0;  // TODO
}

Eigen::VectorXd LinDepNormalHierarchy::like_lpdf_grid(
    const Eigen::MatrixXd &data, const Eigen::MatrixXd &covariates) const {
  Eigen::VectorXd result(data.rows());
  for (size_t i = 0; i < data.rows(); i++) {
    result(i) = 0;  // TODO
  }
  return result;
}

double LinDepNormalHierarchy::marg_lpdf(
    const Eigen::RowVectorXd &datum,
    const Eigen::RowVectorXd &covariate) const {
  return 0;  // TODO
}

Eigen::VectorXd LinDepNormalHierarchy::marg_lpdf_grid(
    const Eigen::MatrixXd &data, const Eigen::MatrixXd &covariates) const {
  Eigen::VectorXd result(data.rows());
  for (size_t i = 0; i < data.rows(); i++) {
    result(i) = 0;  // TODO
  }
  return result;
}

void LinDepNormalHierarchy::draw() {
  return;  // TODO
}

void LinDepNormalHierarchy::sample_given_data() {
  return;  // TODO
}

void LinDepNormalHierarchy::sample_given_data(const Eigen::MatrixXd &data) {
  return;  // TODO
}

void LinDepNormalHierarchy::set_state_from_proto(
    const google::protobuf::Message &state_) {
  return;  // TODO
}

void LinDepNormalHierarchy::set_prior(
    const google::protobuf::Message &prior_) {
  return;  // TODO
}

void LinDepNormalHierarchy::write_state_to_proto(
    google::protobuf::Message *out) const {
  return;  // TODO
}

void LinDepNormalHierarchy::write_hypers_to_proto(
    google::protobuf::Message *out) const {
  return;  // TODO
}
