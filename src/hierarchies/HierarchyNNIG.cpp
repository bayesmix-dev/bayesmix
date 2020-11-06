#include "HierarchyNNIG.hpp"

void HierarchyNNIG::check_state_validity() { assert(std > 0); }

//! \param data                       Column vector of data points
//! \param mu0, alpha0, beta0, lambda Original values for hyperparameters
//! \return                           Vector of updated values for hyperpar.s
std::vector<double> HierarchyNNIG::normal_gamma_update(
    const Eigen::VectorXd &data, const double mu0, const double alpha0,
    const double beta0, const double lambda) {
  // Initialize relevant variables
  double mu_post, alpha_post, beta_post, lambda_post;
  unsigned int n = data.rows();

  if (n == 0) {  // no update possible
    return std::vector<double>{mu0, alpha0, beta0, lambda};
  }

  // Compute updated hyperparameters
  double y_bar = data.mean();  // sample mean
  mu_post = (lambda * mu0 + n * y_bar) / (lambda + n);
  alpha_post = alpha0 + 0.5 * n;
  double ss = (data.dot(data)) - n * y_bar * y_bar;  // sum of squares
  beta_post = beta0 + 0.5 * ss +
              0.5 * lambda * n * (y_bar - mu0) * (y_bar - mu0) / (n + lambda);
  lambda_post = lambda + n;

  return std::vector<double>{mu_post, alpha_post, beta_post, lambda_post};
}

//! \param data Column vector of data points
//! \return     Likehood vector evaluated in data
Eigen::VectorXd HierarchyNNIG::like(const Eigen::MatrixXd &data) {
  Eigen::VectorXd result = lpdf(data);
  return result.array().exp();
}

//! \param data Column vector of data points
//! \return     Log-Likehood vector evaluated in data
Eigen::VectorXd HierarchyNNIG::lpdf(const Eigen::MatrixXd &data) {
  Eigen::VectorXd result(data.rows());
  for (size_t i = 0; i < data.rows(); i++) {
    // Compute likelihood for each data point
    result(i) = stan::math::normal_lpdf(data(i, 0), mean, std);
  }
  return result;
}

//! \param data Column vector of data points
//! \return     Marginal distribution vector evaluated in data
Eigen::VectorXd HierarchyNNIG::eval_marg(const Eigen::MatrixXd &data) {
  Eigen::VectorXd result = marg_lpdf(data);
  return result.array().exp();
}

//! \param data Column vector of data points
//! \return     Marginal distribution vector evaluated in data (log)
Eigen::VectorXd HierarchyNNIG::marg_lpdf(const Eigen::MatrixXd &data) {
  // Get values of hyperparameters
  double mu0 = get_mu0();
  double lambda = get_lambda();
  double alpha0 = get_alpha0();
  double beta0 = get_beta0();

  // Compute standard deviation of marginal distribution
  double sig_n = sqrt(beta0 * (lambda + 1) / (alpha0 * lambda));

  Eigen::VectorXd result(data.rows());
  for (size_t i = 0; i < data.rows(); i++) {
    // Compute marginal for each data point
    result(i) = stan::math::student_t_lpdf(data(i, 0), 2 * alpha0, mu0, sig_n);
  }
  return result;
}

void HierarchyNNIG::draw() {
  // Get values of hyperparameters
  double mu0 = get_mu0();
  double lambda = get_lambda();
  double alpha0 = get_alpha0();
  double beta0 = get_beta0();

  // Update state values from their prior centering distribution
  auto rng = bayesmix::Rng::Instance().get();
  std = sqrt(stan::math::inv_gamma_rng(alpha0, beta0, rng));
  mean = stan::math::normal_rng(mu0, std / sqrt(lambda), rng);
}

//! \param data Column vector of data points
void HierarchyNNIG::sample_given_data(const Eigen::MatrixXd &data) {
  // Update values
  std::vector<double> temp =
      normal_gamma_update(data.col(0), mu0, alpha0, beta0, lambda);
  double mu_post = temp[0];
  double alpha_post = temp[1];
  double beta_post = temp[2];
  double lambda_post = temp[3];

  // Update state values from their prior centering distribution
  auto rng = bayesmix::Rng::Instance().get();
  std = sqrt(stan::math::inv_gamma_rng(alpha_post, beta_post, rng));
  mean = stan::math::normal_rng(mu_post, std / sqrt(lambda_post), rng);
}

void HierarchyNNIG::set_state(google::protobuf::Message *curr, bool check) {
  using namespace google::protobuf::internal;
  using namespace bayesmix;

  MarginalState::ClusterVal *currcast =
      down_cast<MarginalState::ClusterVal *>(curr);

  
  std::cout << "Setting values: " << currcast->DebugString() << std::endl;
  mean = currcast->univ_ls_state().mean();
  std = currcast->univ_ls_state().std();
  std::cout << "mean: " << mean << ", std: " << std << std::endl;
  if (check) {
    check_state_validity();
  }
}

void HierarchyNNIG::get_state_as_proto(google::protobuf::Message *out) {
  using namespace google::protobuf::internal;
  using namespace bayesmix;

  UnivLSState state;
  state.set_mean(mean);
  state.set_std(std);

  down_cast<MarginalState::ClusterVal *>(out)
      ->mutable_univ_ls_state()
      ->CopyFrom(state);

  std::cout << "Getting values: \n" << out->DebugString() << std::endl;
}
