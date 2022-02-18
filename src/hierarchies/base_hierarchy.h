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

  //! Container for the update algorithm adopted
  std::shared_ptr<AbstractUpdater> updater;

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
    } else {
      static_cast<Derived *>(this)->set_default_updater();
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
    updater = updater_;
  };

  std::shared_ptr<AbstractLikelihood> get_likelihood() override {
    return like;
  }
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

  // NOT SURE THIS IS CORRECT, MAYBE OVERRIDE GET_LIKE_LPDF? OR THIS IS EVEN
  // UNNECESSARY
  double like_lpdf(const Eigen::RowVectorXd &datum) const override {
    return like->lpdf(datum);
  }

  //! Returns an independent, data-less copy of this object
  std::shared_ptr<AbstractHierarchy> deep_clone() const override {
    auto out = std::make_shared<Derived>(static_cast<Derived const &>(*this));

    out->clear_data();
    out->clear_summary_statistics();

    out->create_empty_prior();
    std::shared_ptr<google::protobuf::Message> new_prior(prior->New());
    new_prior->CopyFrom(*prior.get());
    out->get_mutable_prior()->CopyFrom(*new_prior.get());

    out->create_empty_hypers();
    auto curr_hypers_proto = get_hypers_proto();
    out->set_hypers_from_proto(*curr_hypers_proto.get());
    out->initialize();
    return out;
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
  double get_marg_lpdf(
      const HyperParams &params, const Eigen::RowVectorXd &datum,
      const Eigen::RowVectorXd &covariate /*= Eigen::RowVectorXd(0)*/) const {
    if (this->is_dependent()) {
      return marg_lpdf(params, datum, covariate);
    } else {
      return marg_lpdf(params, datum);
    }
  }

  // ADD EXCEPTION HANDLING
  double prior_pred_lpdf(const Eigen::RowVectorXd &datum,
                         const Eigen::RowVectorXd &covariate =
                             Eigen::RowVectorXd(0)) const override {
    return get_marg_lpdf(prior->get_hypers(), datum, covariate);
  }

  // ADD EXCEPTION HANDLING
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
  double conditional_pred_lpdf(const Eigen::RowVectorXd &datum,
                               const Eigen::RowVectorXd &covariate =
                                   Eigen::RowVectorXd(0)) const override {
    return get_marg_lpdf(prior->get_posterior_hypers(), datum, covariate);
  }

  // ADD EXCEPTION HANDLING
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
    like->set_state_from_proto(*prior->sample(false), false);
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

  void set_state_from_proto(const google::protobuf::Message &state_) override {
    like->set_state_from_proto(state_);
  };

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
    if (update_params) updater->compute_posterior_hypers(*like, *prior);
  };

  //! Removes a datum and its index from the hierarchy
  void remove_datum(
      const int id, const Eigen::RowVectorXd &datum,
      const bool update_params = false,
      const Eigen::RowVectorXd &covariate = Eigen::RowVectorXd(0)) override {
    like->remove_datum(id, datum, covariate);
    if (update_params) updater->compute_posterior_hypers(*like, *prior);
  };

  //! Main function that initializes members to appropriate values
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

  //! Sets the (pointer to the) dataset matrix
  void set_dataset(const Eigen::MatrixXd *const dataset) override {
    dataset_ptr = dataset;
  }

 protected:
  //! Initializes state parameters to appropriate values
  virtual void initialize_state() = 0;

  // ADD EXEPTION HANDLING
  virtual double marg_lpdf(const HyperParams &params,
                           const Eigen::RowVectorXd &datum) const {
    if (!is_conjugate()) {
      throw std::runtime_error(
          "Call marg_lpdf() for a non-conjugate hierarchy");
    } else {
      throw std::runtime_error("marg_lpdf() not yet implemented");
    }
  }

  // ADD EXEPTION HANDLING
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

// TODO: Move definitions outside the class to improve code cleaness
// TODO: Move this docs in the right place

//! Returns the struct of the current prior hyperparameters
//   Hyperparams get_hypers() const { return *hypers; }

//! Returns the struct of the current posterior hyperparameters
//   Hyperparams get_posterior_hypers() const { return posterior_hypers; }

//! Raises an error if the prior pointer is not initialized
//   void check_prior_is_set() const {
//     if (prior == nullptr) {
//       throw std::invalid_argument("Hierarchy prior was not provided");
//     }
//   }

//! Re-initializes the prior of the hierarchy to a newly created object
//   void create_empty_prior() { prior.reset(new Prior); }

//! Sets the cardinality of the cluster
//   void set_card(const int card_) {
//     card = card_;
//     log_card = (card_ == 0) ? stan::math::NEGATIVE_INFTY : std::log(card_);
//   }

//! Writes current state to a Protobuf message and return a shared_ptr
//! New hierarchies have to first modify the field 'oneof val' in the
//! AlgoritmState::ClusterState message by adding the appropriate type
//   virtual std::shared_ptr<bayesmix::AlgorithmState::ClusterState>
//   get_state_proto() const = 0;

//! Writes current value of hyperparameters to a Protobuf message and
//! return a shared_ptr.
//! New hierarchies have to first modify the field 'oneof val' in the
//! AlgoritmState::HierarchyHypers message by adding the appropriate type
//   virtual std::shared_ptr<bayesmix::AlgorithmState::HierarchyHypers>
//   get_hypers_proto() const = 0;

//! Initializes hierarchy hyperparameters to appropriate values
//   virtual void initialize_hypers() = 0;

//! Resets cardinality and indexes of data in this cluster
//   void clear_data() {
//     set_card(0);
//     cluster_data_idx = std::set<int>();
//   }

//! Down-casts the given generic proto message to a ClusterState proto
//   bayesmix::AlgorithmState::ClusterState *downcast_state(
//       google::protobuf::Message *state_) const {
//     return google::protobuf::internal::down_cast<
//         bayesmix::AlgorithmState::ClusterState *>(state_);
//   }

//! Down-casts the given generic proto message to a ClusterState proto
//   const bayesmix::AlgorithmState::ClusterState &downcast_state(
//       const google::protobuf::Message &state_) const {
//     return google::protobuf::internal::down_cast<
//         const bayesmix::AlgorithmState::ClusterState &>(state_);
//   }

//! Down-casts the given generic proto message to a HierarchyHypers proto
//   bayesmix::AlgorithmState::HierarchyHypers *downcast_hypers(
//       google::protobuf::Message *state_) const {
//     return google::protobuf::internal::down_cast<
//         bayesmix::AlgorithmState::HierarchyHypers *>(state_);
//   }

//! Down-casts the given generic proto message to a HierarchyHypers proto
//   const bayesmix::AlgorithmState::HierarchyHypers &downcast_hypers(
//       const google::protobuf::Message &state_) const {
//     return google::protobuf::internal::down_cast<
//         const bayesmix::AlgorithmState::HierarchyHypers &>(state_);
//   }

//   //! Container for prior hyperparameters values
//   std::shared_ptr<Hyperparams> hypers;

//   //! Container for posterior hyperparameters values
//   Hyperparams posterior_hypers;

//   //! Pointer to a Protobuf prior object for this class
//   std::shared_ptr<Prior> prior;

//   //! Set of indexes of data points belonging to this cluster
//   std::set<int> cluster_data_idx;

//   //! Current cardinality of this cluster
//   int card = 0;

//   //! Logarithm of current cardinality of this cluster
//   double log_card = stan::math::NEGATIVE_INFTY;
// };

#endif  // BAYESMIX_HIERARCHIES_BASE_HIERARCHY_H_
