#ifndef BAYESMIX_MIXINGS_ABSTRACT_MIXING_H_
#define BAYESMIX_MIXINGS_ABSTRACT_MIXING_H_

#include <google/protobuf/message.h>

#include <Eigen/Dense>
#include <memory>
#include <vector>

#include "mixing_id.pb.h"
#include "src/hierarchies/abstract_hierarchy.h"

//! Abstract base class for a generic mixture model
//!
//! This class is the basis for a curiously recurring template pattern (CRTP)
//! for `Mixing` objects, and is solely composed of interface functions for
//! derived classes to use, similarly to how the `Hierarchy` objects were
//! implemented. For more information about this pattern, please refer to the
//! README.md file included in the `hierarchies` subfolder.
//! This class represents a prior for the mixture weights and the induced
//! exchangeable partition probability function (EPPF). See
//! `ConditionalAlgorithm` and `MarginalAlgorithm` for further details.
//!
//! There are two kinds of `Mixing` objects: marginal and conditional mixings.
//! Any class inheriting from this one must implement the `is_conditional()`
//! flag accordingly, and can only be used with the same type of `Algorithm`
//! object. In a conditional mixing, mixing weights for the clusters are part
//! of the state of the algorithm. Their values are stored in some form in this
//! class, and they can be obtained by calling the `get_weights()` method. In a
//! marginal mixing, the actual mixing weights have been marginalized out of
//! the model, and information related to them translates to probability masses
//! to assign a data point to an existing cluster, or to a new one.
//! Classes inheriting from this one must either implement the `get_weights()`
//! method, or the `mass_existing_cluster()` and `mass_new_cluster()` methods,
//! according to the type of mixing which is being implemented. Other
//! required methods are `update_state()` for a conditional update of the
//! mixing's state (if any) given allocations and unique values coming from the
//! library's `Algorithm` classes, and read-write methods involving Protobuf
//! objects.

class AbstractMixing {
 public:
  AbstractMixing() = default;
  virtual ~AbstractMixing() = default;

  //! Initializes class members to appropriate values
  virtual void initialize() = 0;

  //! Performs conditional update of state, given allocations and unique values
  virtual void update_state(
      const std::vector<std::shared_ptr<AbstractHierarchy>> &unique_values,
      const std::vector<unsigned int> &allocations) = 0;

  //! Returns mixing weights (for conditional mixings only)
  virtual Eigen::VectorXd get_weights(
      const bool log, const bool propto,
      const Eigen::RowVectorXd &covariate = Eigen::RowVectorXd(0)) const {
    if (!is_conditional()) {
      throw std::runtime_error(
          "Cannot call this function from non-conditional mixing");
    } else {
      throw std::runtime_error("Not implemented");
    }
  };

  //! Returns probability mass for an old cluster (for marginal mixings only)
  //! @param n          Total dataset size
  //! @param log        Whether to return logarithm-scale values or not
  //! @param propto     Whether to include normalizing constants or not
  //! @param hier       `Hierarchy` object representing the cluster
  //! @param covariate  (Optional) covariate vector
  //! @return           Probability value
  double get_mass_existing_cluster(
      const unsigned int n, const bool log, const bool propto,
      std::shared_ptr<AbstractHierarchy> hier,
      const Eigen::RowVectorXd &covariate = Eigen::RowVectorXd(0)) const {
    if (is_dependent()) {
      return mass_existing_cluster(n, log, propto, hier, covariate);
    } else {
      return mass_existing_cluster(n, log, propto, hier);
    }
  }

  //! Returns probability mass for a new cluster (for marginal mixings only)
  //! @param n          Total dataset size
  //! @param log        Whether to return logarithm-scale values or not
  //! @param propto     Whether to include normalizing constants or not
  //! @param n_clust    Current number of clusters
  //! @param covariate  (Optional) covariate vector
  //! @return           Probability value
  double get_mass_new_cluster(
      const unsigned int n, const bool log, const bool propto,
      const unsigned int n_clust,
      const Eigen::RowVectorXd &covariate = Eigen::RowVectorXd(0)) const {
    if (is_dependent()) {
      return mass_new_cluster(n, log, propto, n_clust, covariate);
    } else {
      return mass_new_cluster(n, log, propto, n_clust);
    }
  }

  //! Returns current number of clusters of the mixture model
  virtual unsigned int get_num_components() const = 0;

  //! Returns a pointer to the Protobuf message of the prior of this cluster
  virtual google::protobuf::Message *get_mutable_prior() = 0;

  //! Sets current number of clusters of the mixture model
  virtual void set_num_components(const unsigned int num_) = 0;

  //! Sets pointer to the covariate matrix for the mixture model
  virtual void set_covariates(Eigen::MatrixXd *covar) = 0;

  virtual void set_state_from_proto(
      const google::protobuf::Message &state_) = 0;

  virtual void write_state_to_proto(google::protobuf::Message *out) const = 0;

  //! Returns the Protobuf ID associated to this class
  virtual bayesmix::MixingId get_id() const = 0;

  //! Returns whether the mixing is conditional or marginal
  virtual bool is_conditional() const = 0;

  //! Returns whether the mixing depends on covariate values or not
  virtual bool is_dependent() const = 0;

 protected:
  //! Private version of get_mass_existing_cluster()
  virtual double mass_existing_cluster(
      const unsigned int n, const bool log, const bool propto,
      std::shared_ptr<AbstractHierarchy> hier,
      const Eigen::RowVectorXd &covariate) const {
    if (!is_dependent()) {
      throw std::runtime_error(
          "Cannot call this function from non-dependent mixing");
    } else {
      throw std::runtime_error("Not implemented");
    }
  };

  //! Private version of get_mass_existing_cluster(), without covariates
  virtual double mass_existing_cluster(
      const unsigned int n, const bool log, const bool propto,
      std::shared_ptr<AbstractHierarchy> hier) const {
    if (is_dependent()) {
      throw std::runtime_error(
          "Cannot call this function from dependent mixing");
    } else {
      throw std::runtime_error("Not implemented");
    }
  };

  //! Private version of get_mass_new_cluster()
  virtual double mass_new_cluster(const unsigned int n, const bool log,
                                  const bool propto,
                                  const unsigned int n_clust,
                                  const Eigen::RowVectorXd &covariate) const {
    if (!is_dependent()) {
      throw std::runtime_error(
          "Cannot call this function from non-dependent mixing");
    } else {
      throw std::runtime_error("Not implemented");
    }
  };

  //! Private version of get_mass_new_cluster(), without covariates
  virtual double mass_new_cluster(const unsigned int n, const bool log,
                                  const bool propto,
                                  const unsigned int n_clust) const {
    if (is_dependent()) {
      throw std::runtime_error(
          "Cannot call this function from dependent mixing");
    } else {
      throw std::runtime_error("Not implemented");
    }
  };
};

#endif  // BAYESMIX_MIXINGS_ABSTRACT_MIXING_H_
