#ifndef BAYESMIX_MIXINGS_MARGINAL_MIXING_H_
#define BAYESMIX_MIXINGS_MARGINAL_MIXING_H_

#include <Eigen/Dense>

#include "base_mixing.h"

//! Abstract base class for a marginal mixture model

//! This class represents a marginal mixture model object to be used in a BNP
//! iterative algorithm. In the context of this library, where a clustering
//! structure is generated on the data, a marginal mixture translates to a
//! certain way of weighing the insertion of data in old clusters vs the
//! creation of new clusters. Therefore any mixture object inheriting from the
//! class must have methods that provide the probabilities for the two
//! aforementioned events. The class will then have its own parameters, and
//! maybe even prior distributions on them.

class MarginalMixing : public BaseMixing {
 public:
  ~MarginalMixing() = default;
  MarginalMixing() = default;
  //!
  bool is_conditional() const override { return false; }

  // PROBABILITIES FUNCTIONS
  //! Mass probability for choosing an already existing cluster

  //! \param card Cardinality of the cluster
  //! \param n    Total number of data points
  //! \return     Probability value
  virtual double mass_existing_cluster(
      const unsigned int n, const bool log, const bool propto,
      std::shared_ptr<AbstractHierarchy> hier,
      const Eigen::RowVectorXd &covariate = Eigen::RowVectorXd(0)) const = 0;

  //! Mass probability for choosing a newly created cluster

  //! \param n_clust Number of clusters
  //! \param n       Total number of data points
  //! \return        Probability value
  virtual double mass_new_cluster(
      const unsigned int n, const bool log, const bool propto,
      const unsigned int n_clust,
      const Eigen::RowVectorXd &covariate = Eigen::RowVectorXd(0)) const = 0;
}

#endif  // BAYESMIX_MIXINGS_MARGINAL_MIXING_H_
