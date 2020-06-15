#ifndef NEAL8_HPP
#define NEAL8_HPP

#include "Neal2.hpp"


//! Template class for Neal's algorithm 8 for conjugate hierarchies

//! This class implements Neal's Gibbs sampling algorithm 8 that generates a
//! Markov chain on the clustering of the provided data.
//!
//! It is a generalization of Neal's algorithm 2 which works for any
//! hierarchical model, even non-conjugate ones, unlike its predecessor. The
//! main difference is the presence of a fixed number of additional unique
//! values, called the auxiliary blocks, which are constantly updated and from
//! which new clusters choose their unique values. They offset the lack of
//! conjugacy of the model by allowing an estimate of the (uncomputable)
//! marginal density via a weighted mean on these new blocks. Other than this
//! and some minor adjustments in the allocation sampling phase to circumvent
//! non-conjugacy, it is the same as Neal's algorithm 2.

//! \param Hierarchy Name of the hierarchy template class
//! \param Hypers    Name of the hyperparameters class
//! \param Mixture   Name of the mixture class

template<template <class> class Hierarchy, class Hypers, class Mixture>
class Neal8 : public Neal2<Hierarchy, Hypers, Mixture>{
protected:
    using Algorithm<Hierarchy, Hypers, Mixture>::data;
    using Algorithm<Hierarchy, Hypers, Mixture>::cardinalities;
    using Algorithm<Hierarchy, Hypers, Mixture>::allocations;
    using Algorithm<Hierarchy, Hypers, Mixture>::unique_values;

    //! Number of auxiliary blocks
    unsigned int n_aux = 3;

    //! Vector of auxiliary blocks
    std::vector<Hierarchy<Hypers>> aux_unique_values;

    // AUXILIARY TOOLS
    //! Computes marginal contribution of a given iteration & cluster
    Eigen::VectorXd density_marginal_component(Hierarchy<Hypers> &temp_hier)
        override;

    // ALGORITHM FUNCTIONS
    void print_startup_message() const override;
    void sample_allocations() override;

public:
    // DESTRUCTOR AND CONSTRUCTORS
    ~Neal8() = default;
    //! \param hypers_  Hyperparameters object for the model
    //! \param mixture_ Mixture object for the model
    //! \param data_    Matrix of row-vectorial data points
    //! \param init     Prescribed n. of clusters for the algorithm initializ.
    Neal8(const Hypers &hypers_, const Mixture &mixture_,
        const Eigen::MatrixXd &data_, const unsigned int init = 0) :
        Neal2<Hierarchy, Hypers, Mixture>::Neal2(hypers_, mixture_, data_,
        init) {
        // Initialize auxiliary blocks
        for(size_t i = 0; i < n_aux; i++){
            aux_unique_values.push_back(this->unique_values[0]);
        }
    }

    // GETTERS AND SETTERS
    unsigned int get_n_aux() const {return n_aux;}
    void set_n_aux(const unsigned int n_aux_) override {
        n_aux = n_aux_;
        // Rebuild the correct amount of auxiliary blocks
        aux_unique_values.clear();
        for(size_t i = 0; i < n_aux; i++){
            aux_unique_values.push_back(this->unique_values[0]);
        }
    }

};

#include "Neal8.imp.hpp"

#endif // NEAL8_HPP
