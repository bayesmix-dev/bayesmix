#ifndef BAYESMIX_ALGORITHMS_PRIVATE_CONDITIONAL_H_
#define BAYESMIX_ALGORITHMS_PRIVATE_CONDITIONAL_H_

#include "base_algorithm.h"
#include "src/privacy/base_channel.h"

template <class Algo>
class PrivateConditionalAlgorithm : public Algo {
 protected:
  Eigen::MatrixXd public_data;
  Eigen::MatrixXd& private_data = Algo::data;
  int n_acc = 0;
  int n_prop = 0;
  std::shared_ptr<BasePrivacyChannel> privacy_channel;

  void update_private_data();

 public:
  PrivateConditionalAlgorithm() = default;
  ~PrivateConditionalAlgorithm() = default;

  void set_channel(const std::shared_ptr<BasePrivacyChannel> channel) {
    privacy_channel = channel;
  }

  std::shared_ptr<BaseAlgorithm> clone() const override {
    auto out = std::make_shared<PrivateConditionalAlgorithm<Algo>>(*this);
    out->set_mixing(Algo::mixing->clone());
    out->set_hierarchy(Algo::unique_values[0]->deep_clone());
    return out;
  }

  void set_public_data(const Eigen::MatrixXd& public_data_);

  void step() override;

  double get_acceptance_rate() { return (1.0 * n_acc) / n_prop; }
};

template <class Algo>
void PrivateConditionalAlgorithm<Algo>::set_public_data(
    const Eigen::MatrixXd& public_data_) {
  Algo::data = privacy_channel->get_candidate_private_data(public_data_);
  private_data = Algo::data;
  this->public_data = public_data_;
}

template <class Algo>
void PrivateConditionalAlgorithm<Algo>::step() {
  Algo::step();
  update_private_data();
}

template <class Algo>
void PrivateConditionalAlgorithm<Algo>::update_private_data() {
  auto& rng = bayesmix::Rng::Instance().get();
  for (int i = 0; i < Algo::data.rows(); i++) {
    n_prop += 1;
    Eigen::VectorXd new_private_datum =
        Algo::unique_values[Algo::allocations[i]]->get_likelihood()->sample();
    double log_a_rate =
        privacy_channel->lpdf(public_data.row(i), new_private_datum) -
        privacy_channel->lpdf(public_data.row(i), private_data.row(i));
    if (std::log(stan::math::uniform_rng(0, 1, rng)) < log_a_rate) {
      n_acc += 1;
      Algo::unique_values[Algo::allocations[i]]->remove_datum(
          i, private_data.row(i), Algo::update_hierarchy_params(),
          Algo::hier_covariates.row(i));
      private_data.row(i) = new_private_datum;
      Algo::unique_values[Algo::allocations[i]]->add_datum(
          i, private_data.row(i), Algo::update_hierarchy_params(),
          Algo::hier_covariates.row(i));
    }
  }
}

#endif  // BAYESMIX_ALGORITHMS_PRIVATE_CONDITIONAL_H_
