#ifndef ALGORITHM_HPP
#define ALGORITHM_HPP

#include <math.h>

#include <Eigen/Dense>
#include <fstream>
#include <random>
#include <stan/math/prim/fun.hpp>
#include <stan/math/prim/prob.hpp>
#include <vector>

#include "../../proto/cpp/marginal_state.pb.h"
#include "../collectors/collector_base.hpp"
#include "../hierarchies/HierarchyBase.hpp"
#include "../mixings/BaseMixing.hpp"
#include "../utils/distributions.hpp"
#include "../utils/proto_utils.hpp"

//! Abstract template class for a Gibbs sampling iterative BNP algorithm.

//! This template class implements a generic algorithm that generates a Markov
//! chain on the clustering of the provided data.
//!
//! An algorithm that inherits from this abstract class will have multiple
//! iterations of the same step. Steps are further split into substeps, each of
//! which updates specific values of the state of the Markov chain, which is
//! composed of an allocations vector and a unique values vector (see below).
//! This is known as a Gibbs sampling structure, where a set of values is
//! updated according to a conditional distribution given all other values.
//! The underlying model for the data is assumed to be a so-called hierarchical
//! model, where each datum is independently drawn from a common likelihood
//! function, whose parameters are specific to each unit and are iid generated
//! from a random probability measure, called mixture. Different data points
//! may have the same parameters as each other, and thus a clustering structure
//! on data emerges, with each cluster being identified by its own parameters,
//! called unique values. These will often be generated from the centering
//! distribution, which is the expected value of the mixture, or from its
//! posterior update. The allocation of a datum is instead the label that
//! indicates the cluster it is currently assigned to. The probability
//! distribution for data from each cluster is called a hierarchy and it can
//! have its own hyperparameters, either random themselves or fixed. The model
//! therefore is:
//!   x_i ~ f(x_i|phi_(c_i))  (data likelihood);
//! phi_c ~ G                 (unique values distribution);
//!     G ~ MM                (mixture model);
//!  E[G] = G0                (centering distribution),
//! where c_i is the allocation of the i-th datum.
//!
//! This class is templatized over the types of the elements of this model: the
//! hierarchies of cluster, their hyperparameters, and the mixing mode.

class AlgorithmBase {
 protected:
  // METHOD PARAMETERS
  //! Iterations of the algorithm
  unsigned int maxiter = 1000;
  //! Number of burn-in iterations, which will be discarded
  unsigned int burnin = 100;

  // DATA AND VALUES CONTAINERS
  //! Matrix of row-vectorial data points
  Eigen::MatrixXd data;
  //! Prescribed number of clusters for the algorithm initialization
  unsigned int init_num_clusters;
  //! Cardinalities of clusters
  std::vector<unsigned int> cardinalities;
  //! Allocation for each datum, i.e. label of the cluster it belongs to
  std::vector<unsigned int> allocations;
  //! Hierarchy of the unique values that identify each cluster
  std::vector<std::shared_ptr<HierarchyBase>> unique_values;
  //! Mixing object
  std::shared_ptr<BaseMixing> mixing;

  // AUXILIARY TOOLS
  //! Returns the values of an algo iteration as a Protobuf object
  bayesmix::MarginalState get_state_as_proto(unsigned int iter);
  //! Computes marginal contribution of a given iteration & cluster
  virtual Eigen::VectorXd lpdf_marginal_component(
      std::shared_ptr<HierarchyBase> temp_hier,
      const Eigen::MatrixXd &grid) = 0;

  // ALGORITHM FUNCTIONS
  virtual void print_startup_message() const = 0;
  virtual void initialize() = 0;
  virtual void sample_allocations() = 0;
  virtual void sample_unique_values() = 0;
  virtual void sample_weights() = 0;
  virtual void update_hypers() = 0;
  virtual void print_ending_message() const {
    std::cout << "Done" << std::endl;
  };
  //! Saves the current iteration's state in Protobuf form to a collector
  void save_state(CollectorBase *collector, unsigned int iter) {
    collector->collect(get_state_as_proto(iter));
  }

  //! Single step of algorithm
  void step() {
    sample_allocations();
    sample_unique_values();
    sample_weights();
    update_hypers();
  }

 public:
  //! Runs the algorithm and saves the whole chain to a collector
  void run(CollectorBase *collector) {
    print_startup_message();
    for (auto &un : unique_values) {
      un->check_and_initialize();
    }
    initialize();
    unsigned int iter = 0;
    collector->start();
    while (iter < maxiter) {
      // std::cout << "Iteration n. " << iter << std::endl;
      step();
      if (iter >= burnin) {
        save_state(collector, iter);
      }
      iter++;
    }
    collector->finish();
    print_ending_message();
  }

  // ESTIMATE FUNCTION
  //! Evaluates the logpdf for each single iteration on a given grid of points
  virtual Eigen::MatrixXd eval_lpdf(const Eigen::MatrixXd &grid,
                                    CollectorBase *const collector) = 0;

  // DESTRUCTOR AND CONSTRUCTORS
  virtual ~AlgorithmBase() = default;
  AlgorithmBase() = default;

  // GETTERS AND SETTERS
  unsigned int get_maxiter() const { return maxiter; }
  unsigned int get_burnin() const { return burnin; }
  unsigned int get_init_num_clusters() const { return init_num_clusters; }

  void set_maxiter(const unsigned int maxiter_) { maxiter = maxiter_; }
  void set_burnin(const unsigned int burnin_) { burnin = burnin_; }
  //! Does nothing except for Neal8
  virtual void set_n_aux(const unsigned int n_aux_) { return; }
  void set_mixing(std::shared_ptr<BaseMixing> mixing_) { mixing = mixing_; }
  void set_data_and_initial_clusters(const Eigen::MatrixXd &data_,
                                     std::shared_ptr<HierarchyBase> hier_,
                                     const unsigned int init = 0) {
    if (data.rows() == 0) {
      std::invalid_argument("Error: empty data matrix");
    }
    if (hier_->is_multivariate() == false && data.cols() > 1) {
      std::cout
          << "Warning: multivariate data supplied to univariate hierarchy."
          << " The algorithm will run correctly, but all data rows other"
          << " than the first one will be ignored" << std::endl;
    }
    data = data_;
    init_num_clusters = (init == 0) ? data.rows() : init;
    // "Warning: initial number of clusters will be set equal to the data
    // size (" << data.rows() << ")"  // TODO
    // Initialize hierarchies for starting clusters
    unique_values.clear();
    for (size_t i = 0; i < init_num_clusters; i++) {
      unique_values.push_back(hier_->clone());
    }
  }

  virtual std::string get_id() const = 0;
};

#endif  // ALGORITHM_HPP
