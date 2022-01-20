#ifndef BAYESMIX_HIERARCHIES_BASE_HIERARCHY_H_
#define BAYESMIX_HIERARCHIES_BASE_HIERARCHY_H_

#include <google/protobuf/message.h>

#include <Eigen/Dense>
#include <memory>
#include <random>
#include <set>
#include <stan/math/prim.hpp>

#include "abstract_hierarchy.h"
#include "algorithm_state.pb.h"
#include "hierarchy_id.pb.h"
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
  BaseHierarchy() = default;
  ~BaseHierarchy() = default;

  void set_likelihood(std::shared_ptr<Likelihood> like_) { like = like_; };
  void set_prior(std::shared_ptr<PriorModel> prior_) { prior = prior_; };
  void set_updater(std::shared_ptr<Updater> updater_) { updater = updater_; };

  std::shared_ptr<AbstractHierarchy> clone() const override {
    // Create copy of the hierarchy
    auto out = std::make_shared<Derived>(static_cast<Derived const &>(*this));
    // Cloning each component class
    out->set_likelihood(std::static_pointer_cast<Likelihood>(like->clone()));
    out->set_prior(std::static_pointer_cast<PriorModel>(prior->clone()));
    out->set_updater(std::static_pointer_cast<Updater>(updater->clone()));
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
    like->set_state_from_proto(*prior->sample(false));
  };

  void sample_full_cond(bool update_params = false) override {
    updater->draw(*like, *prior);
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

  void initialize() override { updater->initialize(*like, *prior); };

  bool is_multivariate() const override { return like->is_multivariate(); };

  bool is_dependent() const override { return like->is_dependent(); };

  bool is_conjugate() const override { return updater->is_conjugate(); };

 protected:
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

//   //! Returns an independent, data-less copy of this object
//   virtual std::shared_ptr<AbstractHierarchy> clone() const override {
//     auto out = std::make_shared<Derived>(static_cast<Derived const
//     &>(*this)); out->clear_data(); out->clear_summary_statistics(); return
//     out;
//   }

//   //! Evaluates the log-likelihood of data in a grid of points
//   //! @param data        Grid of points (by row) which are to be evaluated
//   //! @param covariates  (Optional) covariate vectors associated to data
//   //! @return            The evaluation of the lpdf
//   virtual Eigen::VectorXd like_lpdf_grid(
//       const Eigen::MatrixXd &data,
//       const Eigen::MatrixXd &covariates = Eigen::MatrixXd(0,
//                                                           0)) const
//                                                           override;

//   //! Generates new state values from the centering prior distribution
//   void sample_prior() override {
//     state = static_cast<Derived *>(this)->draw(*hypers);
//   }

//   //! Overloaded version of sample_full_cond(bool), mainly used for
//   debugging virtual void sample_full_cond(
//       const Eigen::MatrixXd &data,
//       const Eigen::MatrixXd &covariates = Eigen::MatrixXd(0, 0)) override;

//   //! Returns the current cardinality of the cluster
//   int get_card() const override { return card; }

//   //! Returns the logarithm of the current cardinality of the cluster
//   double get_log_card() const override { return log_card; }

//   //! Returns the indexes of data points belonging to this cluster
//   std::set<int> get_data_idx() const override { return cluster_data_idx; }

//   //! Returns a pointer to the Protobuf message of the prior of this cluster
//   virtual google::protobuf::Message *get_mutable_prior() override {
//     if (prior == nullptr) {
//       create_empty_prior();
//     }
//     return prior.get();
//   }

//   //! Writes current state to a Protobuf message by pointer
//   void write_state_to_proto(google::protobuf::Message *out) const override;

//   //! Writes current values of the hyperparameters to a Protobuf message by
//   //! pointer
//   void write_hypers_to_proto(google::protobuf::Message *out) const override;

//   //! Returns the struct of the current state
//   State get_state() const { return state; }

//   //! Returns the struct of the current prior hyperparameters
//   Hyperparams get_hypers() const { return *hypers; }

//   //! Returns the struct of the current posterior hyperparameters
//   Hyperparams get_posterior_hypers() const { return posterior_hypers; }

//   //! Adds a datum and its index to the hierarchy
//   void add_datum(
//       const int id, const Eigen::RowVectorXd &datum,
//       const bool update_params = false,
//       const Eigen::RowVectorXd &covariate = Eigen::RowVectorXd(0)) override;

//   //! Removes a datum and its index from the hierarchy
//   void remove_datum(
//       const int id, const Eigen::RowVectorXd &datum,
//       const bool update_params = false,
//       const Eigen::RowVectorXd &covariate = Eigen::RowVectorXd(0)) override;

//   //! Main function that initializes members to appropriate values
//   void initialize() override {
//     hypers = std::make_shared<Hyperparams>();
//     check_prior_is_set();
//     initialize_hypers();
//     initialize_state();
//     posterior_hypers = *hypers;
//     clear_data();
//     clear_summary_statistics();
//   }

//  protected:
//   //! Raises an error if the prior pointer is not initialized
//   void check_prior_is_set() const {
//     if (prior == nullptr) {
//       throw std::invalid_argument("Hierarchy prior was not provided");
//     }
//   }

//   //! Re-initializes the prior of the hierarchy to a newly created object
//   void create_empty_prior() { prior.reset(new Prior); }

//   //! Sets the cardinality of the cluster
//   void set_card(const int card_) {
//     card = card_;
//     log_card = (card_ == 0) ? stan::math::NEGATIVE_INFTY : std::log(card_);
//   }

//   //! Writes current state to a Protobuf message and return a shared_ptr
//   //! New hierarchies have to first modify the field 'oneof val' in the
//   //! AlgoritmState::ClusterState message by adding the appropriate type
//   virtual std::shared_ptr<bayesmix::AlgorithmState::ClusterState>
//   get_state_proto() const = 0;

//   //! Initializes state parameters to appropriate values
//   virtual void initialize_state() = 0;

//   //! Writes current value of hyperparameters to a Protobuf message and
//   //! return a shared_ptr.
//   //! New hierarchies have to first modify the field 'oneof val' in the
//   //! AlgoritmState::HierarchyHypers message by adding the appropriate type
//   virtual std::shared_ptr<bayesmix::AlgorithmState::HierarchyHypers>
//   get_hypers_proto() const = 0;

//   //! Initializes hierarchy hyperparameters to appropriate values
//   virtual void initialize_hypers() = 0;

//   //! Resets cardinality and indexes of data in this cluster
//   void clear_data() {
//     set_card(0);
//     cluster_data_idx = std::set<int>();
//   }

//   virtual void clear_summary_statistics() = 0;

//   //! Down-casts the given generic proto message to a ClusterState proto
//   bayesmix::AlgorithmState::ClusterState *downcast_state(
//       google::protobuf::Message *state_) const {
//     return google::protobuf::internal::down_cast<
//         bayesmix::AlgorithmState::ClusterState *>(state_);
//   }

//   //! Down-casts the given generic proto message to a ClusterState proto
//   const bayesmix::AlgorithmState::ClusterState &downcast_state(
//       const google::protobuf::Message &state_) const {
//     return google::protobuf::internal::down_cast<
//         const bayesmix::AlgorithmState::ClusterState &>(state_);
//   }

//   //! Down-casts the given generic proto message to a HierarchyHypers proto
//   bayesmix::AlgorithmState::HierarchyHypers *downcast_hypers(
//       google::protobuf::Message *state_) const {
//     return google::protobuf::internal::down_cast<
//         bayesmix::AlgorithmState::HierarchyHypers *>(state_);
//   }

//   //! Down-casts the given generic proto message to a HierarchyHypers proto
//   const bayesmix::AlgorithmState::HierarchyHypers &downcast_hypers(
//       const google::protobuf::Message &state_) const {
//     return google::protobuf::internal::down_cast<
//         const bayesmix::AlgorithmState::HierarchyHypers &>(state_);
//   }

//   //! Container for state values
//   State state;

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

// template <class Derived, typename State, typename Hyperparams, typename
// Prior> void BaseHierarchy<Derived, State, Hyperparams, Prior>::add_datum(
//     const int id, const Eigen::RowVectorXd &datum,
//     const bool update_params /*= false*/,
//     const Eigen::RowVectorXd &covariate /*= Eigen::RowVectorXd(0)*/) {
//   assert(cluster_data_idx.find(id) == cluster_data_idx.end());
//   card += 1;
//   log_card = std::log(card);
//   static_cast<Derived *>(this)->update_ss(datum, covariate, true);
//   cluster_data_idx.insert(id);
//   if (update_params) {
//     static_cast<Derived *>(this)->save_posterior_hypers();
//   }
// }

// template <class Derived, typename State, typename Hyperparams, typename
// Prior> void BaseHierarchy<Derived, State, Hyperparams, Prior>::remove_datum(
//     const int id, const Eigen::RowVectorXd &datum,
//     const bool update_params /*= false*/,
//     const Eigen::RowVectorXd &covariate /* = Eigen::RowVectorXd(0)*/) {
//   static_cast<Derived *>(this)->update_ss(datum, covariate, false);
//   set_card(card - 1);
//   auto it = cluster_data_idx.find(id);
//   assert(it != cluster_data_idx.end());
//   cluster_data_idx.erase(it);
//   if (update_params) {
//     static_cast<Derived *>(this)->save_posterior_hypers();
//   }
// }

// template <class Derived, typename State, typename Hyperparams, typename
// Prior> void BaseHierarchy<Derived, State, Hyperparams,
// Prior>::write_state_to_proto(
//     google::protobuf::Message *out) const {
//   std::shared_ptr<bayesmix::AlgorithmState::ClusterState> state_ =
//       get_state_proto();
//   auto *out_cast = downcast_state(out);
//   out_cast->CopyFrom(*state_.get());
//   out_cast->set_cardinality(card);
// }

// template <class Derived, typename State, typename Hyperparams, typename
// Prior> void BaseHierarchy<Derived, State, Hyperparams,
// Prior>::write_hypers_to_proto(
//     google::protobuf::Message *out) const {
//   std::shared_ptr<bayesmix::AlgorithmState::HierarchyHypers> hypers_ =
//       get_hypers_proto();
//   auto *out_cast = downcast_hypers(out);
//   out_cast->CopyFrom(*hypers_.get());
// }

// template <class Derived, typename State, typename Hyperparams, typename
// Prior> Eigen::VectorXd BaseHierarchy<Derived, State, Hyperparams,
// Prior>::like_lpdf_grid(
//     const Eigen::MatrixXd &data,
//     const Eigen::MatrixXd &covariates /*= Eigen::MatrixXd(0, 0)*/) const {
//   Eigen::VectorXd lpdf(data.rows());
//   if (covariates.cols() == 0) {
//     // Pass null value as covariate
//     for (int i = 0; i < data.rows(); i++) {
//       lpdf(i) = static_cast<Derived const *>(this)->get_like_lpdf(
//           data.row(i), Eigen::RowVectorXd(0));
//     }
//   } else if (covariates.rows() == 1) {
//     // Use unique covariate
//     for (int i = 0; i < data.rows(); i++) {
//       lpdf(i) = static_cast<Derived const *>(this)->get_like_lpdf(
//           data.row(i), covariates.row(0));
//     }
//   } else {
//     // Use different covariates
//     for (int i = 0; i < data.rows(); i++) {
//       lpdf(i) = static_cast<Derived const *>(this)->get_like_lpdf(
//           data.row(i), covariates.row(i));
//     }
//   }
//   return lpdf;
// }

// template <class Derived, typename State, typename Hyperparams, typename
// Prior> void BaseHierarchy<Derived, State, Hyperparams,
// Prior>::sample_full_cond(
//     const Eigen::MatrixXd &data,
//     const Eigen::MatrixXd &covariates /*= Eigen::MatrixXd(0, 0)*/) {
//   clear_data();
//   clear_summary_statistics();
//   if (covariates.cols() == 0) {
//     // Pass null value as covariate
//     for (int i = 0; i < data.rows(); i++) {
//       static_cast<Derived *>(this)->add_datum(i, data.row(i), false,
//                                               Eigen::RowVectorXd(0));
//     }
//   } else if (covariates.rows() == 1) {
//     // Use unique covariate
//     for (int i = 0; i < data.rows(); i++) {
//       static_cast<Derived *>(this)->add_datum(i, data.row(i), false,
//                                               covariates.row(0));
//     }
//   } else {
//     // Use different covariates
//     for (int i = 0; i < data.rows(); i++) {
//       static_cast<Derived *>(this)->add_datum(i, data.row(i), false,
//                                               covariates.row(i));
//     }
//   }
//   static_cast<Derived *>(this)->sample_full_cond(true);
// }

#endif  // BAYESMIX_HIERARCHIES_BASE_HIERARCHY_H_
