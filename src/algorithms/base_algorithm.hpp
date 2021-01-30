#ifndef BAYESMIX_ALGORITHMS_BASE_ALGORITHM_HPP_
#define BAYESMIX_ALGORITHMS_BASE_ALGORITHM_HPP_

#include <Eigen/Dense>
#include <memory>
#include <vector>

#include "../collectors/base_collector.hpp"
#include "../hierarchies/base_hierarchy.hpp"
#include "../mixings/base_mixing.hpp"
#include "lib/progressbar/progressbar.hpp"
#include "marginal_state.pb.h"

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

class BaseAlgorithm {
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
  //! Allocation for each datum, i.e. label of the cluster it belongs to
  std::vector<unsigned int> allocations;
  //! Hierarchy of the unique values that identify each cluster
  std::vector<std::shared_ptr<BaseHierarchy>> unique_values;
  //!
  Eigen::MatrixXd hier_covariates;
  //! Mixing object
  std::shared_ptr<BaseMixing> mixing;
  //!
  Eigen::MatrixXd mix_covariates;

  // AUXILIARY TOOLS
  //! Returns the values of an algo iteration as a Protobuf object
  bayesmix::MarginalState get_state_as_proto(unsigned int iter);

  // ALGORITHM FUNCTIONS
  virtual void print_startup_message() const = 0;
  virtual void initialize();
  virtual void sample_allocations() = 0;
  virtual void sample_unique_values() = 0;
  void update_hierarchy_hypers();
  virtual void print_ending_message() const {
    std::cout << "Done" << std::endl;
  };

  //! Saves the current iteration's state in Protobuf form to a collector
  void save_state(BaseCollector *collector, unsigned int iter) {
    collector->collect(get_state_as_proto(iter));
  }

  //! Single step of algorithm
  virtual void step() {
    sample_allocations();
    sample_unique_values();
    update_hierarchy_hypers();
    mixing->update_state(unique_values, data.size());
  }

 public:
  //! Runs the algorithm and saves the whole chain to a collector
  void run(BaseCollector *collector) {
    initialize();
    print_startup_message();
    unsigned int iter = 0;
    collector->start_collecting();

    progresscpp::ProgressBar bar(maxiter, 60);

    while (iter < maxiter) {
      step();
      if (iter >= burnin) {
        save_state(collector, iter);
      }
      iter++;
      ++bar;
      bar.display();
    }
    collector->finish_collecting();
    bar.done();
    print_ending_message();
  }

  // ESTIMATE FUNCTION
  //! Evaluates the logpdf for each single iteration on a given grid of points
  virtual Eigen::MatrixXd eval_lpdf(const Eigen::MatrixXd &grid,
                                    BaseCollector *const collector) = 0;
  // TODO will soon become obsolete
  virtual Eigen::MatrixXd eval_lpdf(const Eigen::MatrixXd &grid,
                                    const Eigen::MatrixXd &covariates,
                                    BaseCollector *const collector) = 0;

  // DESTRUCTOR AND CONSTRUCTORS
  virtual ~BaseAlgorithm() = default;
  BaseAlgorithm() = default;

  void add_datum_to_hierarchy(const unsigned int datum_idx,
                              std::shared_ptr<BaseHierarchy> hier);
  void remove_datum_from_hierarchy(const unsigned int datum_idx,
                                   std::shared_ptr<BaseHierarchy> hier);

  // GETTERS AND SETTERS
  unsigned int get_maxiter() const { return maxiter; }
  unsigned int get_burnin() const { return burnin; }
  unsigned int get_init_num_clusters() const { return init_num_clusters; }

  void set_maxiter(const unsigned int maxiter_) { maxiter = maxiter_; }
  void set_burnin(const unsigned int burnin_) { burnin = burnin_; }
  //! Does nothing except for Neal8
  virtual void set_n_aux(const unsigned int n_aux_) {}
  void set_mixing(const std::shared_ptr<BaseMixing> mixing_) {
    mixing = mixing_;
  }
  void set_data(const Eigen::MatrixXd &data_) { data = data_; }
  void set_hier_covariates(const Eigen::MatrixXd &cov) {
    hier_covariates = cov;
  }
  void set_mix_covariates(const Eigen::MatrixXd &cov) { mix_covariates = cov; }
  void set_initial_clusters(const std::shared_ptr<BaseHierarchy> hier_,
                            const unsigned int init = 0) {
    unique_values.clear();
    unique_values.push_back(hier_);
    init_num_clusters = init;
  }
  virtual std::string get_id() const = 0;
};

#endif  // BAYESMIX_ALGORITHMS_BASE_ALGORITHM_HPP_
