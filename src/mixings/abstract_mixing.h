#ifndef BAYESMIX_MIXINGS_ABSTRACT_MIXING_H_
#define BAYESMIX_MIXINGS_ABSTRACT_MIXING_H_

#include <google/protobuf/message.h>

#include <Eigen/Dense>
#include <memory>
#include <vector>

#include "mixing_id.pb.h"
#include "src/hierarchies/abstract_hierarchy.h"

//! Abstract base class for a generic mixture model

//! This class represents a mixture model object to be used in a BNP iterative
//! algorithm.
//! TODO update:
//! This class represents a marginal mixture model object to be used in a BNP
//! iterative algorithm. In the context of this library, where a clustering
//! structure is generated on the data, a marginal mixture translates to a
//! certain way of weighing the insertion of data in old clusters vs the
//! creation of new clusters. Therefore any mixture object inheriting from the
//! class must have methods that provide the probabilities for the two
//! aforementioned events. The class will then have its own parameters, and
//! maybe even prior distributions on them.

class AbstractMixing {
 public:
  // DESTRUCTOR AND CONSTRUCTORS
  virtual ~AbstractMixing() = default;
  AbstractMixing() = default;

  virtual void initialize() = 0;
  //! Returns true if the mixing has covariates i.e. is a dependent model
  virtual bool is_dependent() const = 0;
  //!
  virtual bool is_conditional() const = 0;
  //!
  virtual void update_state(
      const std::vector<std::shared_ptr<AbstractHierarchy>> &unique_values,
      const std::vector<unsigned int> &allocations) = 0;
  //!
  virtual Eigen::VectorXd get_weights(
      const bool log, const bool propto,
      const Eigen::RowVectorXd &covariate = Eigen::RowVectorXd(0)) const {
    throw std::runtime_error(
        "Cannot call get_weights() from non-conditional mixing");
  };

  //! @param card Cardinality of the cluster
  //! @param n    Total number of data points
  //! @return     Probability value
  virtual double mass_existing_cluster(
      const unsigned int n, const bool log, const bool propto,
      std::shared_ptr<AbstractHierarchy> hier,
      const Eigen::RowVectorXd &covariate = Eigen::RowVectorXd(0)) const {
    throw std::runtime_error(
        "Cannot call mass_existing_cluster() from non-marginal mixing");
  };

  //! @param n_clust Number of clusters
  //! @param n       Total number of data points
  //! @return        Probability value
  virtual double mass_new_cluster(
      const unsigned int n, const bool log, const bool propto,
      const unsigned int n_clust,
      const Eigen::RowVectorXd &covariate = Eigen::RowVectorXd(0)) const {
    throw std::runtime_error(
        "Cannot call mass_new_cluster() from non-marginal mixing");
  };

  // GETTERS AND SETTERS
  virtual unsigned int get_num_components() const = 0;
  virtual google::protobuf::Message *get_mutable_prior() = 0;
  virtual void set_num_components(const unsigned int num_) = 0;
  virtual void set_covariates(Eigen::MatrixXd *covar) = 0;
  virtual void set_state_from_proto(
      const google::protobuf::Message &state_) = 0;
  virtual void write_state_to_proto(google::protobuf::Message *out) const = 0;
  //! Returns the Protobuf ID associated to this class
  virtual bayesmix::MixingId get_id() const = 0;
};

#endif  // BAYESMIX_MIXINGS_ABSTRACT_MIXING_H_
