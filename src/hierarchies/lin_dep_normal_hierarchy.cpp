#include "lin_dep_normal_hierarchy.hpp"

void LinDepNormalHierarchy::initialize() {
  state.coefficients = Eigen::VectorXd::Zero(dim);
}

void LinDepNormalHierarchy::clear_data() {
  data_sum = 0.0;
  data_sum_squares = 0.0;
  card = 0;
  cluster_data_idx.clear();
}

void LinDepNormalHierarchy::update_summary_statistics(
  const Eigen::VectorXd &datum, bool add) {
  if (add) {
    data_sum += datum(0);
    data_sum_squares += datum(0) * datum(0);
  } else {
    data_sum -= datum(0);
    data_sum_squares -= datum(0) * datum(0);
  }
}

LinDepNormalHierarchy::Hyperparams LinDepNormalHierarchy::some_update() {
  return LinDepNormalHierarchy::Hyperparams();  // TODO
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

void LinDepNormalHierarchy::update_hypers(
  const std::vector<bayesmix::MarginalState::ClusterState> &states) {
  return;  // TODO
}
