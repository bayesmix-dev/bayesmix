#ifndef NEAL2_HPP
#define NEAL2_HPP

#include "Algorithm.hpp"


//! Template class for Neal's algorithm 2 for conjugate hierarchies

//! This class implements Neal's Gibbs sampling algorithm 2 that generates a
//! Markov chain on the clustering of the provided data.
//!
//! Using this algorithm implicitly assumes that the provided hierarchy class
//! represents a conjugate model, i.e. one in which posterior distributions have
//! the same form as their corresponding prior distributions. Conjugacy is made
//! use of in the computation of the estimated density's marginal component,
//! since the marginal distribution for the data can be expressed analytically.
//!
//! The basic idea for this algorithm is randomly drawing new allocations for
//! the data points according to weights that depend on the cardinalities of the
//! current clustering and on the mixture model used. This way, sometimes new
//! clusters are created, and thus new unique values for them must be generated
//! from the prior centering distribution. After that, unique values for each
//! cluster are instead updated via the posterior distribution, which again has
//! a closed-form expression thanks to conjugacy.

//! \param Hierarchy Name of the hierarchy template class
//! \param Hypers    Name of the hyperparameters class
//! \param Mixing    Name of the mixing mode class

template<template <class> class Hierarchy, class Hypers, class Mixing>
class Neal2 : public Algorithm<Hierarchy, Hypers, Mixing> {
protected:
    using Algorithm<Hierarchy, Hypers, Mixing>::data;
    using Algorithm<Hierarchy, Hypers, Mixing>::cardinalities;
    using Algorithm<Hierarchy, Hypers, Mixing>::allocations;
    using Algorithm<Hierarchy, Hypers, Mixing>::unique_values;

    // AUXILIARY TOOLS
    //! Computes marginal contribution of a given iteration & cluster
    Eigen::VectorXd density_marginal_component(Hierarchy<Hypers> &temp_hier)
        override;

    // ALGORITHM FUNCTIONS
    void print_startup_message() const override;
    void initialize() override;
    void sample_allocations() override;
    void sample_unique_values() override;
    //! Empty: this algorithm does not use weights
    void sample_weights() override {return;}
    //! Empty: this algorithm does not update hyperparameters
    void update_hypers() override {return;}

public:
    // DESTRUCTOR AND CONSTRUCTORS
    ~Neal2() = default;
    //! \param hypers_  Hyperparameters object for the model
    //! \param mixing_  Mixing object for the model
    //! \param data_    Matrix of row-vectorial data points
    //! \param init     Prescribed n. of clusters for the algorithm initializ.
    Neal2(const Hypers &hypers_, const Mixing &mixing_,
        const Eigen::MatrixXd &data_, const unsigned int init = 0) :
        Algorithm<Hierarchy, Hypers, Mixing>::Algorithm(hypers_, mixing_,
            data_, init) {}

};

#include "Neal2.imp.hpp"

#endif // NEAL2_HPP
