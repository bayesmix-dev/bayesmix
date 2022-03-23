#ifndef BAYESMIX_HIERARCHIES_BASE_HIERARCHY_H_
#define BAYESMIX_HIERARCHIES_BASE_HIERARCHY_H_

#include <google/protobuf/message.h>

#include <memory>
#include <random>
#include <set>
// #include <stan/math/prim.hpp>
#include <stan/math/rev.hpp>

#include "abstract_hierarchy.h"
#include "algorithm_state.pb.h"
#include "hierarchy_id.pb.h"
#include "src/utils/rng.h"
#include "updaters/target_lpdf_unconstrained.h"

//! Base template class for a hierarchy object.

//! This class is a templatized version of, and derived from, the
//! `AbstractHierarchy` class, and the second stage of the curiously recurring
//! template pattern for `Hierarchy` objects (please see the docs of the parent
//! class for further information). It includes class members and some more
//! functions which could not be implemented in the non-templatized abstract
//! class.
//! See, for instance, `NNIGHierarchy` to better understand the CRTP patterns.

//! @tparam Derived      Name of the implemented derived class
//! @tparam Likelihood   Class name of the likelihood model for the hierarchy
//! @tparam PriorModel   Class name of the prior model for the hierarchy

template <class Derived, class Likelihood, class PriorModel>
class BaseHierarchy : public AbstractHierarchy {
 protected:
  //! Container for the likelihood of the hierarchy
  std::shared_ptr<Likelihood> like = std::make_shared<Likelihood>();

  //! Container for the prior model of the hierarchy
  std::shared_ptr<PriorModel> prior = std::make_shared<PriorModel>();

  //! Container for the update algorithm
  std::shared_ptr<AbstractUpdater> updater;

 public:
  using HyperParams = decltype(prior->get_hypers());
  using ProtoHypers = AbstractUpdater::ProtoHypers;

  //! Constructor that allows the specification of Likelihood, PriorModel and
  //! Updater for a given Hierarchy
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
    } else {
      static_cast<Derived *>(this)->set_default_updater();
    }
  }

  //! Default destructor
  ~BaseHierarchy() = default;

  //! Sets the likelihood for the current hierarchy
  void set_likelihood(std::shared_ptr<AbstractLikelihood> like_) /*override*/ {
    like = std::static_pointer_cast<Likelihood>(like_);
  }

  //! Sets the prior model for the current hierarchy
  void set_prior(std::shared_ptr<AbstractPriorModel> prior_) /*override*/ {
    prior = std::static_pointer_cast<PriorModel>(prior_);
  }

  //! Sets the update algorithm for the current hierarchy
  void set_updater(std::shared_ptr<AbstractUpdater> updater_) override {
    updater = updater_;
  };

  //! Returns (a pointer to) the likelihood for the current hierarchy
  std::shared_ptr<AbstractLikelihood> get_likelihood() override {
    return like;
  }

  //! Returns (a pointer to) the prior model for the current hierarchy.
  std::shared_ptr<AbstractPriorModel> get_prior() override { return prior; }

  //! Returns an independent, data-less copy of this object
  std::shared_ptr<AbstractHierarchy> clone() const override {
    // Create copy of the hierarchy
    auto out = std::make_shared<Derived>(static_cast<Derived const &>(*this));
    // Cloning each component class
    out->set_likelihood(std::static_pointer_cast<Likelihood>(like->clone()));
    out->set_prior(std::static_pointer_cast<PriorModel>(prior->clone()));
    return out;
  };

  //! Returns an independent, data-less deep copy of this object
  std::shared_ptr<AbstractHierarchy> deep_clone() const override {
    // Create copy of the hierarchy
    auto out = std::make_shared<Derived>(static_cast<Derived const &>(*this));
    // Simple clone for Likelihood is enough
    out->set_likelihood(std::static_pointer_cast<Likelihood>(like->clone()));
    // Deep clone required for PriorModel
    out->set_prior(std::static_pointer_cast<PriorModel>(prior->deep_clone()));
    return out;
  }

  //! Returns an independent, data-less copy of this object
  // std::shared_ptr<AbstractHierarchy> deep_clone() const override {
  //   auto out = std::make_shared<Derived>(static_cast<Derived const
  //   &>(*this));

  //   out->clear_data();
  //   out->clear_summary_statistics();

  //   out->create_empty_prior();
  //   std::shared_ptr<google::protobuf::Message> new_prior(prior->New());
  //   new_prior->CopyFrom(*prior.get());
  //   out->get_mutable_prior()->CopyFrom(*new_prior.get());

  //   out->create_empty_hypers();
  //   auto curr_hypers_proto = get_hypers_proto();
  //   out->set_hypers_from_proto(*curr_hypers_proto.get());
  //   out->initialize();
  //   return out;
  // }

  //! Public wrapper for `like_lpdf()` methods
  double get_like_lpdf(const Eigen::RowVectorXd &datum,
                       const Eigen::RowVectorXd &covariate =
                           Eigen::RowVectorXd(0)) const override {
    return like->lpdf(datum, covariate);
  }

  //! Evaluates the log-likelihood of data in a grid of points
  //! @param data        Grid of points (by row) which are to be evaluated
  //! @param covariates  (Optional) covariate vectors associated to data
  //! @return            The evaluation of the lpdf
  Eigen::VectorXd like_lpdf_grid(const Eigen::MatrixXd &data,
                                 const Eigen::MatrixXd &covariates =
                                     Eigen::MatrixXd(0, 0)) const override {
    return like->lpdf_grid(data, covariates);
  };

  // ADD EXCEPTION HANDLING
  //! Public wrapper for `marg_lpdf()` methods
  double get_marg_lpdf(
      const ProtoHypers &hier_params, const Eigen::RowVectorXd &datum,
      const Eigen::RowVectorXd &covariate /*= Eigen::RowVectorXd(0)*/) const {
    if (this->is_dependent()) {
      return marg_lpdf(hier_params, datum, covariate);
    } else {
      return marg_lpdf(hier_params, datum);
    }
  }

  // ADD EXCEPTION HANDLING
  //! Evaluates the log-prior predictive distribution of data in a single point
  //! @param datum      Point which is to be evaluated
  //! @param covariate  (Optional) covariate vector associated to datum
  //! @return           The evaluation of the lpdf
  double prior_pred_lpdf(const Eigen::RowVectorXd &datum,
                         const Eigen::RowVectorXd &covariate =
                             Eigen::RowVectorXd(0)) const override {
    return get_marg_lpdf(*(prior->get_hypers_proto()), datum, covariate);
  }

  // ADD EXCEPTION HANDLING
  //! Evaluates the log-prior predictive distr. of data in a grid of points
  //! @param data        Grid of points (by row) which are to be evaluated
  //! @param covariates  (Optional) covariate vectors associated to data
  //! @return            The evaluation of the lpdf
  Eigen::VectorXd prior_pred_lpdf_grid(
      const Eigen::MatrixXd &data,
      const Eigen::MatrixXd &covariates /*= Eigen::MatrixXd(0, 0)*/)
      const override {
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

  // ADD EXCEPTION HANDLING
  //! Evaluates the log-conditional predictive distr. of data in a single point
  //! @param datum      Point which is to be evaluated
  //! @param covariate  (Optional) covariate vector associated to datum
  //! @return           The evaluation of the lpdf
  double conditional_pred_lpdf(const Eigen::RowVectorXd &datum,
                               const Eigen::RowVectorXd &covariate =
                                   Eigen::RowVectorXd(0)) const override {
    return get_marg_lpdf(updater->compute_posterior_hypers(*like, *prior),
                         datum, covariate);
  }

  // ADD EXCEPTION HANDLING
  //! Evaluates the log-prior predictive distr. of data in a grid of points
  //! @param data        Grid of points (by row) which are to be evaluated
  //! @param covariates  (Optional) covariate vectors associated to data
  //! @return            The evaluation of the lpdf
  Eigen::VectorXd conditional_pred_lpdf_grid(
      const Eigen::MatrixXd &data,
      const Eigen::MatrixXd &covariates /*= Eigen::MatrixXd(0, 0)*/)
      const override {
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

  //! Generates new state values from the centering prior distribution
  void sample_prior() override {
    auto hypers = prior->get_hypers_proto();
    like->set_state_from_proto(*prior->sample(*hypers), false);
  };

  //! Generates new state values from the centering posterior distribution
  //! @param update_params  Save posterior hypers after the computation?
  void sample_full_cond(bool update_params = false) override {
    updater->draw(*like, *prior, update_params);
  };

  //! Overloaded version of sample_full_cond(bool), mainly used for debugging
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

  //! Updates hyperparameter values given a vector of cluster states
  void update_hypers(const std::vector<bayesmix::AlgorithmState::ClusterState>
                         &states) override {
    prior->update_hypers(states);
  };

  //! Returns the class of the current state
  auto get_state() const -> decltype(like->get_state()) {
    return like->get_state();
  };

  //! Returns the current cardinality of the cluster
  int get_card() const override { return like->get_card(); };

  //! Returns the logarithm of the current cardinality of the cluster
  double get_log_card() const override { return like->get_log_card(); };

  //! Returns the indexes of data points belonging to this cluster
  std::set<int> get_data_idx() const override { return like->get_data_idx(); };

  //! Writes current state to a Protobuf message and return a shared_ptr
  //! New hierarchies have to first modify the field 'oneof val' in the
  //! AlgoritmState::ClusterState message by adding the appropriate type
  std::shared_ptr<bayesmix::AlgorithmState::ClusterState> get_state_proto()
      const override {
    return like->get_state_proto();
  }

  //! Returns a pointer to the Protobuf message of the prior of this cluster
  google::protobuf::Message *get_mutable_prior() override {
    return prior->get_mutable_prior();
  };

  //! Writes current state to a Protobuf message by pointer
  void write_state_to_proto(google::protobuf::Message *out) const override {
    like->write_state_to_proto(out);
  };

  //! Writes current values of the hyperparameters to a Protobuf message by
  //! pointer
  void write_hypers_to_proto(google::protobuf::Message *out) const override {
    prior->write_hypers_to_proto(out);
  };

  //! Read and set state values from a given Protobuf message
  void set_state_from_proto(const google::protobuf::Message &state_) override {
    like->set_state_from_proto(state_);
  };

  //! Read and set hyperparameter values from a given Protobuf message
  void set_hypers_from_proto(
      const google::protobuf::Message &state_) override {
    prior->set_hypers_from_proto(state_);
  };

  //! Adds a datum and its index to the hierarchy
  void add_datum(
      const int id, const Eigen::RowVectorXd &datum,
      const bool update_params = false,
      const Eigen::RowVectorXd &covariate = Eigen::RowVectorXd(0)) override {
    like->add_datum(id, datum, covariate);
    if (update_params) {
      updater->save_posterior_hypers(
          updater->compute_posterior_hypers(*like, *prior));
    }
  };

  //! Removes a datum and its index from the hierarchy
  void remove_datum(
      const int id, const Eigen::RowVectorXd &datum,
      const bool update_params = false,
      const Eigen::RowVectorXd &covariate = Eigen::RowVectorXd(0)) override {
    like->remove_datum(id, datum, covariate);
    if (update_params) {
      updater->save_posterior_hypers(
          updater->compute_posterior_hypers(*like, *prior));
    }
  };

  //! Main function that initializes members to appropriate values
  void initialize() override {
    prior->initialize();
    if (is_conjugate()) {
      updater->save_posterior_hypers(*prior->get_hypers_proto());
    }
    initialize_state();
    like->clear_data();
    like->clear_summary_statistics();
  };

  //! Returns whether the hierarchy models multivariate data or not
  bool is_multivariate() const override { return like->is_multivariate(); };

  //! Returns whether the hierarchy depends on covariate values or not
  bool is_dependent() const override { return like->is_dependent(); };

  //! Returns whether the hierarchy represents a conjugate model or not
  bool is_conjugate() const override { return updater->is_conjugate(); };

  //! Sets the (pointer to the) dataset matrix
  void set_dataset(const Eigen::MatrixXd *const dataset) override {
    like->set_dataset(dataset);
  }

 protected:
  //! Initializes state parameters to appropriate values
  virtual void initialize_state() = 0;

  // ADD EXEPTION HANDLING FOR is_dependent()?
  //! Evaluates the log-marginal distribution of data in a single point
  //! @param params     Container of (prior or posterior) hyperparameter values
  //! @param datum      Point which is to be evaluated
  //! @return           The evaluation of the lpdf
  virtual double marg_lpdf(const ProtoHypers &hier_params,
                           const Eigen::RowVectorXd &datum) const {
    if (!is_conjugate()) {
      throw std::runtime_error(
          "Call marg_lpdf() for a non-conjugate hierarchy");
    } else {
      throw std::runtime_error(
          "marg_lpdf() not implemented for this hierarchy");
    }
  }

  // ADD EXEPTION HANDLING FOR is_dependent()?
  //! Evaluates the log-marginal distribution of data in a single point
  //! @param params     Container of (prior or posterior) hyperparameter values
  //! @param datum      Point which is to be evaluated
  //! @param covariate  Covariate vector associated to datum
  //! @return           The evaluation of the lpdf
  virtual double marg_lpdf(const ProtoHypers &hier_params,
                           const Eigen::RowVectorXd &datum,
                           const Eigen::RowVectorXd &covariate) const {
    if (!is_conjugate()) {
      throw std::runtime_error(
          "Call marg_lpdf() for a non-conjugate hierarchy");
    } else {
      throw std::runtime_error(
          "marg_lpdf() not implemented for this hierarchy");
    }
  }

  // TEMPORANEO!
  // const Eigen::MatrixXd *dataset_ptr;
};

// TODO: Move definitions outside the class to improve code cleaness

#endif  // BAYESMIX_HIERARCHIES_BASE_HIERARCHY_H_
