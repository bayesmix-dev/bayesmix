#include "private_conditional.h"

void PrivateNeal2::PrivateConditionalAlgorithm(
    const Eigen::MatrixXd &public_data_) {
  data = privacy_channel->get_candidate_private_data(public_data_);
  private_data = data;
  this->public_data = public_data_;
}

void PrivateConditionalAlgorithm::step() {
  algo->step();
  update_private_data();
}

void PrivateConditionalAlgorithm::update_private_data() {
  for (int i = 0; i < data.rows(); i++) {
    n_prop += 1;
    new_private_datum =
        unique_values[allocations[i]]->get_likelihood()->sample();
    double log_a_rate =
        privacy_channel->lpdf(public_data.row(i), new_private_datum) -
        privacy_channel->lpdf(public_data.row(i), private_data.row(i));
    if (std::log(stan::math::uniform_rng(0, 1, rng)) < log_a_rate) {
      n_acc += 1;
      unique_values[allocations[i]]->remove_datum(i, private_data.row(i),
                                                  update_hierarchy_params(),
                                                  hier_covariates.row(i));
      private_data.row(i) = new_private_datum;
      unique_values[allocations[i]]->add_datum(i, private_data.row(i),
                                               update_hierarchy_params(),
                                               hier_covariates.row(i));
    }
  }
}
