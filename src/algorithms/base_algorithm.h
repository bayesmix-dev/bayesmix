#ifndef BAYESMIX_ALGORITHMS_BASE_ALGORITHM_H_
#define BAYESMIX_ALGORITHMS_BASE_ALGORITHM_H_

#include <lib/progressbar/progressbar.h>

#include <Eigen/Dense>
#include <memory>
#include <vector>

#include "algorithm_id.pb.h"
#include "algorithm_params.pb.h"
#include "algorithm_state.pb.h"
#include "src/collectors/base_collector.h"
#include "src/hierarchies/base_hierarchy.h"
#include "src/mixings/abstract_mixing.h"

//! Abstract template class representing a Gibbs sampler for mixture models.
//! Gibbs samplers are a particular class of Markov chain Monte Carlo (MCMC)
//! algorithms that are used to perform posterior inference in Bayesian
//! models.
//! For the specific class of models that can be fit using these algorithm,
//! see `MarginalAlgorithm` and `ConditionalAlgorithm`.
//!
//! This class receives a number of initialization parameters related to the
//! MCMC algorithm itself (such as the number of iterations) in a Protobuf
//! message. It also utilizes several class objects related to the underlying
//! model: a `Mixing` object, and a vector of `Hierarchy` objects.
//! It includes methods for running the MCMC simulation and for estimating the
//! posterior density on a given grid of points. In particular, density
//! estimation is performed by averaging the estimate associated to each
//! single MCMC iteration. The specific method depends on whether the
//! algorithm is marginal or conditional (see derived classes
//! `MarginalAlgorithm` and `ConditionalAlgorithm`).
//! Results for this class' methods are saved either to `Collector` objects, or
//! to text files.

class BaseAlgorithm {
 public:
  BaseAlgorithm() = default;
  virtual ~BaseAlgorithm() = default;

  //! Returns whether the algorithm is conditional or marginal
  virtual bool is_conditional() const = 0;

  //! Returns whether it is restricted to conjugate `Hierarchy` objects
  virtual bool requires_conjugate_hierarchy() const { return false; }

  //! Runs the algorithm and saves the MCMC chain to the given `Collector`
  void run(BaseCollector *collector) {
    initialize();
    if (verbose) {
      print_startup_message();
    }
    unsigned int iter = 0;
    collector->start_collecting();
    progresscpp::ProgressBar *bar = nullptr;
    if (verbose) {
      bar = new progresscpp::ProgressBar(maxiter, 60);
    }
    // Main loop
    while (iter < maxiter) {
      step();
      if (iter >= burnin) {
        save_state(collector, iter);
      }
      iter++;
      if (verbose) {
        ++(*bar);
        bar->display();
      }
    }
    collector->finish_collecting();
    if (verbose) {
      bar->done();
      delete bar;
      print_ending_message();
    }
  }

  // DENSITY ESTIMATE FUNCTIONS
  //! Evaluates the estimated log-pdf on all iterations over a grid of points
  //! @param collector   `Collector` object from which the MCMC is read
  //! @param grid   Grid of row points on which the density is to be evaluated
  //! @param hier_covariate   Optional covariates related to the `Hierarchy`
  //! @param mix_covariate   Optional covariates related to the `Mixing`
  //! @return   The estimation over all iterations (rows) and points (cols)
  virtual Eigen::MatrixXd eval_lpdf(
      BaseCollector *const collector, const Eigen::MatrixXd &grid,
      const Eigen::RowVectorXd &hier_covariate = Eigen::RowVectorXd(0),
      const Eigen::RowVectorXd &mix_covariate = Eigen::RowVectorXd(0));

  unsigned int get_maxiter() const { return maxiter; }

  unsigned int get_burnin() const { return burnin; }

  //! Returns the Protobuf ID associated to this class
  virtual bayesmix::AlgorithmId get_id() const = 0;

  void set_maxiter(const unsigned int maxiter_) { maxiter = maxiter_; }

  void set_burnin(const unsigned int burnin_) { burnin = burnin_; }

  void set_init_num_clusters(const unsigned int init_) {
    init_num_clusters = init_;
  }

  void set_data(const Eigen::MatrixXd &data_) { data = data_; }

  void set_hier_covariates(const Eigen::MatrixXd &cov) {
    hier_covariates = cov;
  }

  void set_mix_covariates(const Eigen::MatrixXd &cov) { mix_covariates = cov; }

  void set_mixing(const std::shared_ptr<AbstractMixing> mixing_) {
    mixing = mixing_;
  }

  void set_hierarchy(const std::shared_ptr<AbstractHierarchy> hier_) {
    unique_values.clear();
    unique_values.push_back(hier_);
  }

  void set_verbose(const bool verbose_) { verbose = verbose_; }

  //! Reads and sets algorithm parameters from an appropriate Protobuf message
  virtual void read_params_from_proto(const bayesmix::AlgorithmParams &params);

  void set_state_proto(std::shared_ptr<google::protobuf::Message> state) {
    curr_state.CopyFrom(
        google::protobuf::internal::down_cast<bayesmix::AlgorithmState &>(
            *state.get()));
  }

  //! Evaluates the estimated log-lpdf on the state contained in `curr_state`
  //! @param grid   Grid of row points on which the density is to be evaluated
  //! @param hier_covariate   Optional covariates related to the `Hierarchy`
  //! @param mix_covariate   Optional covariates related to the `Mixing`
  //! @return   The estimation on the iteration of `curr_state` over all points
  virtual Eigen::VectorXd lpdf_from_state(
      const Eigen::MatrixXd &grid, const Eigen::RowVectorXd &hier_covariate,
      const Eigen::RowVectorXd &mix_covariate) = 0;

  virtual std::shared_ptr<BaseAlgorithm> clone() = 0;

  std::vector<std::shared_ptr<AbstractHierarchy>> get_unique_values() {
    return unique_values;
  }

 protected:
  // ALGORITHM FUNCTIONS
  //! Initializes all members of the class before running the algorithm
  virtual void initialize();

  //! Prints a message at the beginning of `run()`
  virtual void print_startup_message() const = 0;

  //! Performs Gibbs sampling sub-step for all allocation values
  virtual void sample_allocations() = 0;

  //! Performs Gibbs sampling sub-step for all unique values
  virtual void sample_unique_values() = 0;

  //! Updates hyperparameters for all `Hierarchy` objects
  void update_hierarchy_hypers();

  //! Prints a message at the end of `run()`
  virtual void print_ending_message() const {
    std::cout << "Done" << std::endl;
  };

  //! Saves the current iteration's state in Protobuf form to a `Collector`
  void save_state(BaseCollector *collector, unsigned int iter) {
    collector->collect(get_state_as_proto(iter));
  }

  //! Performs a single step of algorithm
  virtual void step() {
    sample_allocations();
    sample_unique_values();
    update_hierarchy_hypers();
    mixing->update_state(unique_values, allocations);
  }

  // AUXILIARY TOOLS
  //! Returns Protobuf object containing current state values and iter number
  bayesmix::AlgorithmState get_state_as_proto(unsigned int iter);

  //! Advances `Collector` reading pointer by one, and returns 1 if successful
  bool update_state_from_collector(BaseCollector *coll);

  /*
    Returns whether the posterior parameters for the hierarchies should be
     updated each time an observation is added or removed from the cluster.
     This can potentially reduce computational effort in algorithms which do
     not need this kind of update. If false, this update is usually performed
     instead during the `sample_unique_values()` substep, and viceversa.
  */
  virtual bool update_hierarchy_params() { return false; }

  // ALGORITHM PARAMETERS
  //! Iterations of the algorithm, including burn-in
  unsigned int maxiter = 1000;

  //! Number of initial burn-in iterations, which will be then discarded
  unsigned int burnin = 100;

  //! Initial number of clusters, only used for initialization
  unsigned int init_num_clusters = 0;

  // DATA AND VALUES CONTAINERS
  //! Matrix of row-vectorial data points
  Eigen::MatrixXd data;

  //! Matrix of covariates (if any) for the `Hierarchy` objects
  Eigen::MatrixXd hier_covariates;

  //! Matrix of covariates (if any) for the `Mixing` object
  Eigen::MatrixXd mix_covariates;

  //! Pointer to the `Mixing` object
  std::shared_ptr<AbstractMixing> mixing;

  //! Protobuf message that holds the currently evaluated `Collector` state
  bayesmix::AlgorithmState curr_state;

  // ALGORITHM STATE
  //! Vector of allocation labels for each datum
  std::vector<unsigned int> allocations;

  //! Vector of pointers to `Hierarchy` objects that identify each cluster
  std::vector<std::shared_ptr<AbstractHierarchy>> unique_values;

  // MISCELLANEOUS
  //! Turns on or off the descriptive output of the class methods
  bool verbose = true;
};

#endif  // BAYESMIX_ALGORITHMS_BASE_ALGORITHM_H_
