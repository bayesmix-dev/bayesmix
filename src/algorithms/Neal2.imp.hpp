#ifndef NEAL2_IMP_HPP
#define NEAL2_IMP_HPP

#include "Neal2.hpp"


//! \param temp_hier Temporary hierarchy object
//! \return          Vector of evaluation of component on the provided grid
template<template <class> class Hierarchy, class Hypers, class Mixture>
Eigen::VectorXd Neal2<Hierarchy, Hypers, Mixture>::density_marginal_component(
    Hierarchy<Hypers> &temp_hier){
    // Exploit conjugacy of hierarchy
    return temp_hier.eval_marg(this->density.first);
}


template<template <class> class Hierarchy, class Hypers, class Mixture>
void Neal2<Hierarchy, Hypers, Mixture>::print_startup_message() const {
    std::cout << "Running Neal2 algorithm..." << std::endl;
}


template<template <class> class Hierarchy, class Hypers, class Mixture>
void Neal2<Hierarchy, Hypers, Mixture>::initialize(){
    // Initialize objects
    cardinalities.reserve(data.rows());
    std::default_random_engine generator;
    // Build uniform probability on clusters, given their initial number
    std::uniform_int_distribution<int> distro(0,this->init_num_clusters-1);

    // Allocate one datum per cluster first, and update cardinalities
    for(size_t h = 0; h < this->init_num_clusters; h++){
      allocations.push_back(h);
      cardinalities.push_back(1);
    }

    // Randomly allocate all remaining data, and update cardinalities
    for(size_t j = this->init_num_clusters; j < data.rows(); j++){
        unsigned int clust = distro(generator);
        allocations.push_back(clust);
        cardinalities[clust] += 1;
    }
}


template<template <class> class Hierarchy, class Hypers, class Mixture>
void Neal2<Hierarchy, Hypers, Mixture>::sample_allocations(){
    // Initialize relevant values
    unsigned int n = data.rows();

    // Loop over data points
    for(size_t i = 0; i < n; i++){
        // Current i-th datum as row vector
        Eigen::Matrix<double, 1, Eigen::Dynamic> datum = data.row(i);
        // Initialize current number of clusters
        unsigned int n_clust = unique_values.size();
        // Initialize pseudo-flag
        int singleton = 0;
        if(cardinalities[ allocations[i] ] == 1){
            singleton = 1;
        }

        // Remove datum from cluster
        cardinalities[ allocations[i] ] -= 1;

        // Compute probabilities of clusters
        Eigen::VectorXd probas(n_clust+(1-singleton));
        double tot = 0.0;
        // Loop over clusters
        for(size_t k = 0; k < n_clust; k++){
            // Probability of being assigned to an already existing cluster
            probas(k) = this->mixture.mass_existing_cluster(
                cardinalities[k], n-1) * unique_values[k].like(datum)(0);
            if(singleton == 1 && k == allocations[i]){
                // Probability of being assigned to a newly generated cluster
                probas(k) = this->mixture.mass_new_cluster(n_clust, n-1) *
                    unique_values[0].eval_marg(datum)(0);
            }
            tot += probas(k);
        }
        if(singleton == 0){
            // Further update with marginal component
            probas(n_clust) = this->mixture.mass_new_cluster(n_clust, n-1) *
                unique_values[0].eval_marg(datum)(0);
            tot += probas(n_clust);
        }
        // Normalize
        probas = probas / tot;

        // Draw a NEW value for datum allocation
        unsigned int c_new = stan::math::categorical_rng(probas, this->rng) - 1;

        // Assign datum to its new cluster and update cardinalities:
        // 4 cases are handled separately
        if(singleton == 1){
            if(c_new == allocations[i]){
                // Case 1: datum moves from a singleton to a new cluster
                // Replace former with new cluster by updating unique values
                unique_values[ allocations[i] ].sample_given_data(datum);
                cardinalities[c_new] += 1;
            }

            else { // Case 2: datum moves from a singleton to an old cluster
                unique_values.erase( unique_values.begin() + allocations[i] );
                unsigned int c_old = allocations[i];
                allocations[i] = c_new;
                // Relabel allocations so that they are consecutive numbers
                for(auto &c : allocations){
                    if(c > c_old){
                        c -= 1;
                    }
                }
                cardinalities[c_new] += 1;
                cardinalities.erase(cardinalities.begin() + c_old);
            }
        }

        else { // if singleton == 0
            if(c_new == n_clust){
                // Case 3: datum moves from a non-singleton to a new cluster
                Hierarchy<Hypers> new_unique( unique_values[0].get_hypers() );
                // Generate new unique values with posterior sampling
                new_unique.sample_given_data(datum);
                unique_values.push_back(new_unique);
                allocations[i] = n_clust;
                cardinalities.push_back(1);
            }

            else { // Case 4: datum moves from a non-singleton to an old cluster
                allocations[i] = c_new;
                cardinalities[c_new] += 1;
            }
        }
    }
}


template<template <class> class Hierarchy, class Hypers, class Mixture>
void Neal2<Hierarchy, Hypers, Mixture>::sample_unique_values(){
    // Initialize relevant values
    unsigned int n_clust = unique_values.size();
    unsigned int n = allocations.size();

    // Vector that represents all clusters by the indexes of their data points
    std::vector<std::vector<unsigned int>> clust_idxs(n_clust);
    for(size_t i = 0; i < n; i++){
        clust_idxs[ allocations[i] ].push_back(i);
    }

    // Loop over clusters
    for(size_t i = 0; i < n_clust; i++){
        unsigned int curr_size = clust_idxs[i].size();
        // Build vector that contains the data points in the current cluster
        Eigen::MatrixXd curr_data(curr_size, data.cols());
        for(size_t j = 0; j < curr_size; j++){
            curr_data.row(j) = data.row( clust_idxs[i][j] );
        }
        // Update unique values via the posterior distribution
        unique_values[i].sample_given_data(curr_data);
    }
}


#endif // NEAL2_IMP_HPP
