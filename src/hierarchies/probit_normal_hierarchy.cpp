#import "probit_normal_hierarchy.hpp"

ProbitNormalHierarchy::initialize() {
  int dim = covariate_map.begin()->second->cols();  // or it can be done...
                                                    // in the state

  normal_parameters = Eigen::VectorXd::Zero(dim);  // or any other... 
  kernel_parameters = Eigen::VectorXd::Zero(dim);  // starting value

  // ... TODO
}

double ProbitNormalHierarchy::like_lpdf(const int idx) const {
  return 0;  // TODO
}

Eigen::VectorXd ProbitNormalHierarchy::like_lpdf_grid(
	const std::vector<int> &idxs) const {
  Eigen::VectorXd result(idxs.size());
  for (size_t i = 0; i < idxs.size(); i++) {
    result(i) = 0;  // TODO
  }
  return result;
}

double ProbitNormalHierarchy::marg_lpdf(const int idx) const {
  return 0;  // TODO
}

Eigen::VectorXd ProbitNormalHierarchy::marg_lpdf_grid(
	const std::vector<int> &idxs) const {
  Eigen::VectorXd result(idxs.size());
  for (size_t i = 0; i < idxs.size(); i++) {
    result(i) = 0;  // TODO
  }
  return result;
}

void ProbitNormalHierarchy::draw(){
	return;  // TODO
}

void ProbitNormalHierarchy::sample_given_data() {
	return;  // TODO
}

void ProbitNormalHierarchy::sample_given_data(const Eigen::MatrixXd &data) {
	return;  // TODO
}

void ProbitNormalHierarchy::set_state_from_proto(
	const google::protobuf::Message &state_) {
	return;  // TODO
}

void ProbitNormalHierarchy::set_prior(
  const google::protobuf::Message &prior_) {
	return;  // TODO
}

void ProbitNormalHierarchy::write_state_to_proto(
	google::protobuf::Message *out) const {
	return;  // TODO
}

void ProbitNormalHierarchy::write_hypers_to_proto(
	google::protobuf::Message *out) const {
	return;  // TODO
}
