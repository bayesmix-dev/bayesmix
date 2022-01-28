#ifndef BAYESMIX_HIERARCHIES_BASE_HIERARCHY_H_
#define BAYESMIX_HIERARCHIES_BASE_HIERARCHY_H_

#include <google/protobuf/message.h>

#include <memory>
#include <random>
#include <set>
#include <stan/math/prim.hpp>
#include <stan/math/rev.hpp>

#include "abstract_hierarchy.h"
#include "algorithm_state.pb.h"
#include "hierarchy_id.pb.h"
#include "src/hierarchies/updaters/target_lpdf_unconstrained.h"
#include "src/utils/rng.h"

//! Base template class for a hierarchy object.

//! This class is a templatized version of, and derived from, the
//! `AbstractHierarchy` class, and the second stage of the curiously recurring
//! template pattern for `Hierarchy` objects (please see the docs of the parent
//! class for further information). It includes class members and some more
//! functions which could not be implemented in the non-templatized abstract
//! class.
//! See, for instance, `ConjugateHierarchy` and `NNIGHierarchy` to better
//! understand the CRTP patterns.

//! @tparam Derived      Name of the implemented derived class
//! @tparam State        Class name of the container for state values
//! @tparam Hyperparams  Class name of the container for hyperprior parameters
//! @tparam Prior        Class name of the container for prior parameters

template <class Derived, class Likelihood, class PriorModel, class Updater>
class BaseHierarchy : public AbstractHierarchy {
 protected:
  std::shared_ptr<Likelihood> like = std::make_shared<Likelihood>();
  std::shared_ptr<PriorModel> prior = std::make_shared<PriorModel>();
  std::shared_ptr<Updater> updater = std::make_shared<Updater>();

 public:
  using HyperParams = decltype(prior->get_hypers());

  BaseHierarchy(std::shared_ptr<AbstractLikelihood> like_ = nullptr,
                std::shared_ptr<AbstractPriorModel> prior_ = nullptr,
                std::shared_ptr<AbstractUpdater> updater_ = nullptr) {
    if (like_) {
      set_likelihood(like_);
    }
    if (prior_) {
      set_prior(prior_);
    }
    if (updater_) {
      set_updater(updater_);
    }
  }

  ~BaseHierarchy() = default;

  void set_likelihood(std::shared_ptr<AbstractLikelihood> like_) override {
    like = std::static_pointer_cast<Likelihood>(like_);
  }
  void set_prior(std::shared_ptr<AbstractPriorModel> prior_) override {
    prior = std::static_pointer_cast<PriorModel>(prior_);
  }
  void set_updater(std::shared_ptr<AbstractUpdater> updater_) override {
    updater = std::static_pointer_cast<Updater>(updater_);
  };

  std::shared_ptr<AbstractLikelihood> get_likelihood() override {
    return like;
  }
  std::shared_ptr<AbstractPriorModel> get_prior() override { return prior; }

  std::shared_ptr<AbstractHierarchy> clone() const override {
    // Create copy of the hierarchy
    auto out = std::make_shared<Derived>(static_cast<Derived const &>(*this));
    // Cloning each component class
    out->set_likelihood(std::static_pointer_cast<Likelihood>(like->clone()));
    out->set_prior(std::static_pointer_cast<PriorModel>(prior->clone()));
    return out;
  };

  double like_lpdf(const Eigen::RowVectorXd &datum) const override {
    return like->lpdf(datum);
  }

  Eigen::VectorXd like_lpdf_grid(const Eigen::MatrixXd &data,
                                 const Eigen::MatrixXd &covariates =
                                     Eigen::MatrixXd(0, 0)) const override {
    return like->lpdf_grid(data, covariates);
  };

  double get_marg_lpdf(
      const HyperParams &params, const Eigen::RowVectorXd &datum,
      const Eigen::RowVectorXd &covariate /*= Eigen::RowVectorXd(0)*/) const {
    if (this->is_dependent()) {
      return marg_lpdf(params, datum, covariate);
    } else {
      return marg_lpdf(params, datum);
    }
  }

  double prior_pred_lpdf(const Eigen::RowVectorXd &datum,
                         const Eigen::RowVectorXd &covariate =
                             Eigen::RowVectorXd(0)) const override {
    return get_marg_lpdf(prior->get_hypers(), datum, covariate);
  }

  Eigen::VectorXd prior_pred_lpdf_grid(
      const Eigen::MatrixXd &data,
      const Eigen::MatrixXd &covariates /*= Eigen::MatrixXd(0, 0)*/) const {
    Eigen::VectorXd lpdf(data.rows());
    if (covariates.cols() == 0) {
      // Pass null value as covariate
      for (int i = 0; i < data.rows(); i++) {
        lpdf(i) = static_cast<Derived const *>(this)->prior_pred_lpdf(
            data.row(i), Eigen::RowVectorXd(0));
      }
    } else if (covariates.rows() == 1) {
      // Use unique covariate
      for (int i = 0; i < data.rows(); i++) {
        lpdf(i) = static_cast<Derived const *>(this)->prior_pred_lpdf(
            data.row(i), covariates.row(0));
      }
    } else {
      // Use different covariates
      for (int i = 0; i < data.rows(); i++) {
        lpdf(i) = static_cast<Derived const *>(this)->prior_pred_lpdf(
            data.row(i), covariates.row(i));
      }
    }
    return lpdf;
  }

  double conditional_pred_lpdf(const Eigen::RowVectorXd &datum,
                               const Eigen::RowVectorXd &covariate =
                                   Eigen::RowVectorXd(0)) const override {
    return get_marg_lpdf(prior->get_posterior_hypers(), datum, covariate);
  }

  Eigen::VectorXd conditional_pred_lpdf_grid(
      const Eigen::MatrixXd &data,
      const Eigen::MatrixXd &covariates /*= Eigen::MatrixXd(0, 0)*/) const {
    Eigen::VectorXd lpdf(data.rows());
    if (covariates.cols() == 0) {
      // Pass null value as covariate
      for (int i = 0; i < data.rows(); i++) {
        lpdf(i) = static_cast<Derived const *>(this)->conditional_pred_lpdf(
            data.row(i), Eigen::RowVectorXd(0));
      }
    } else if (covariates.rows() == 1) {
      // Use unique covariate
      for (int i = 0; i < data.rows(); i++) {
        lpdf(i) = static_cast<Derived const *>(this)->conditional_pred_lpdf(
            data.row(i), covariates.row(0));
      }
    } else {
      // Use different covariates
      for (int i = 0; i < data.rows(); i++) {
        lpdf(i) = static_cast<Derived const *>(this)->conditional_pred_lpdf(
            data.row(i), covariates.row(i));
      }
    }
    return lpdf;
  }

  void sample_prior() override {
    // int card = like->get_card();
    like->set_state_from_proto(*prior->sample(false), false);
    // like->set_card(card);
  };

  void sample_full_cond(bool update_params = false) override {
    target_lpdf_unconstrained target(this);
    updater->draw(*like, *prior, update_params, target);
  };

  void sample_full_cond(
      const Eigen::MatrixXd &data,
      const Eigen::MatrixXd &covariates = Eigen::MatrixXd(0, 0)) override {
    like->clear_data();
    like->clear_summary_statistics();
    if (covariates.cols() == 0) {
      // Pass null value as covariate
      for (int i = 0; i < data.rows(); i++) {
        static_cast<Derived *>(this)->add_datum(i, data.row(i), false,
                                                Eigen::RowVectorXd(0));
      }
    } else if (covariates.rows() == 1) {
      // Use unique covariate
      for (int i = 0; i < data.rows(); i++) {
        static_cast<Derived *>(this)->add_datum(i, data.row(i), false,
                                                covariates.row(0));
      }
    } else {
      // Use different covariates
      for (int i = 0; i < data.rows(); i++) {
        static_cast<Derived *>(this)->add_datum(i, data.row(i), false,
                                                covariates.row(i));
      }
    }
    static_cast<Derived *>(this)->sample_full_cond(true);
  };

  void update_hypers(const std::vector<bayesmix::AlgorithmState::ClusterState>
                         &states) override {
    prior->update_hypers(states);
  };

  auto get_state() const -> decltype(like->get_state()) {
    return like->get_state();
  };

  int get_card() const override { return like->get_card(); };

  double get_log_card() const override { return like->get_log_card(); };

  std::set<int> get_data_idx() const override { return like->get_data_idx(); };

  google::protobuf::Message *get_mutable_prior() {
    return prior->get_mutable_prior();
  };

  void write_state_to_proto(google::protobuf::Message *out) const override {
    like->write_state_to_proto(out);
  };

  void write_hypers_to_proto(google::protobuf::Message *out) const override {
    prior->write_hypers_to_proto(out);
  };

  void set_state_from_proto(const google::protobuf::Message &state_) override {
    like->set_state_from_proto(state_);
  };

  void set_hypers_from_proto(
      const google::protobuf::Message &state_) override {
    prior->set_hypers_from_proto(state_);
  };

  void add_datum(
      const int id, const Eigen::RowVectorXd &datum,
      const bool update_params = false,
      const Eigen::RowVectorXd &covariate = Eigen::RowVectorXd(0)) override {
    like->add_datum(id, datum, covariate);
    if (update_params) updater->compute_posterior_hypers(*like, *prior);
  };

  void remove_datum(
      const int id, const Eigen::RowVectorXd &datum,
      const bool update_params = false,
      const Eigen::RowVectorXd &covariate = Eigen::RowVectorXd(0)) override {
    like->remove_datum(id, datum, covariate);
    if (update_params) updater->compute_posterior_hypers(*like, *prior);
  };

  void initialize() override {
    prior->initialize();
    if (is_conjugate()) prior->set_posterior_hypers(prior->get_hypers());
    initialize_state();
    like->clear_data();
    like->clear_summary_statistics();
  };

  bool is_multivariate() const override { return like->is_multivariate(); };

  bool is_dependent() const override { return like->is_dependent(); };

  bool is_conjugate() const override { return updater->is_conjugate(); };

 protected:
  virtual void initialize_state() = 0;

  virtual double marg_lpdf(const HyperParams &params,
                           const Eigen::RowVectorXd &datum) const {
    if (!is_conjugate()) {
      throw std::runtime_error(
          "Call marg_lpdf() for a non-conjugate hierarchy");
    } else {
      throw std::runtime_error("marg_lpdf() not yet implemented");
    }
  }

  virtual double marg_lpdf(const HyperParams &params,
                           const Eigen::RowVectorXd &datum,
                           const Eigen::RowVectorXd &covariate) const {
    if (!is_conjugate()) {
      throw std::runtime_error(
          "Call marg_lpdf() for a non-conjugate hierarchy");
    } else {
      throw std::runtime_error("marg_lpdf() not yet implemented");
    }
  }
};

#endif  // BAYESMIX_HIERARCHIES_BASE_HIERARCHY_H_
