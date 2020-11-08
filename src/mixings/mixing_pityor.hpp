#ifndef BAYESMIX_MIXINGS_MIXING_PITYOR_HPP_
#define BAYESMIX_MIXINGS_MIXING_PITYOR_HPP_

#include <cassert>

#include "mixing_base.hpp"

//! Class that represents the Pitman-Yor process mixture model.

//! This class represents a particular mixture model for iterative BNP
//! algorithms, called the Pitman-Yor process. It has two parameters, strength
//! and discount. It is a generalized version of the Dirichlet process, which
//! has discount = 0 and strength = total mass. In terms of the algorithms, it
//! translates to a mixture that assigns a weight to the creation of a new
//! cluster proportional to their cardinalities, but reduced by the discount
//! factor, while the weight for a newly created cluster is the remaining
//! one counting the total amount as the sample size increased by the strength.

class MixingPitYor : public MixingBase {
 protected:
  //! Strength and discount parameters
  double strength, discount;

 public:
  // DESTRUCTOR AND CONSTRUCTORS
  ~MixingPitYor() = default;
  MixingPitYor() = default;
  MixingPitYor(const double strength_, const double discount_)
      : strength(strength_), discount(discount_) {
    assert(strength > -discount);
    assert(0 <= discount && discount < 1);
  }

  // PROBABILITIES FUNCTIONS
  //! Mass probability for choosing an already existing cluster

  //! \param card Cardinality of the cluster
  //! \param n    Total number of data points
  //! \return     Probability value
  double mass_existing_cluster(const unsigned int card,
                               const unsigned int n) const override {
    return (card == 0) ? 0 : (card - discount) / (n + strength);
  }

  //! Mass probability for choosing a newly created cluster

  //! \param n_clust Number of clusters
  //! \param n       Total number of data points
  //! \return        Probability value
  double mass_new_cluster(const unsigned int n_clust,
                          const unsigned int n) const override {
    return (strength + discount * n_clust) / (n + strength);
  }

  // GETTERS AND SETTERS
  double get_strength() const { return strength; }
  double get_discount() const { return discount; }
  void set_strength(const double strength_) { strength = strength_; }
  void set_discount(const double discount_) { discount = discount_; }

  std::string get_id() const override { return "Pitman-Yor"; }
};

#endif  // BAYESMIX_MIXINGS_MIXING_PITYOR_HPP_
