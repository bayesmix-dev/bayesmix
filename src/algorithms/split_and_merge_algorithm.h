#ifndef BAYESMIX_ALGORITHMS_SPLIT_AND_MERGE_ALGORITHM_H_
#define BAYESMIX_ALGORITHMS_SPLIT_AND_MERGE_ALGORITHM_H_

#include <Eigen/Dense>
#include <memory>

#include "algorithm_id.pb.h"
#include "marginal_algorithm.h"
#include "src/hierarchies/base_hierarchy.h"

//! Template class for Split and Merge for conjugate hierarchies

//! This class implements Split and Merge algorithm from Jain and Neal (2004)
//! that generates a Markov chain on the clustering of the provided data.
//!
//! This algorithm requires the use of a `ConjugateHierarchy` object.

//! After picking at random two points, the Split and Merge algorithm
//! takes all the points that are in the same clusters as the first two and
//! assigns to each of them, at random, one of two temporary clusters. Then,
//! it updates these assignments by doing a restricted Gibbs sampling (GS),
//! restricted in the sense that it considers only the points cited above
//! and only the two temporary clusters.
//! The result of this operation is used to compute the acceptance
//! probability for the split or merge proposal.
//! These Metropolis-Hastings (MH) steps are alternated with full Gibbs
//! sampling steps, similar to the ones that are computed in Neal3 algorithm.
class SplitAndMergeAlgorithm : public MarginalAlgorithm {
 public:
  // DESTRUCTOR AND CONSTRUCTORS
  SplitAndMergeAlgorithm() = default;
  ~SplitAndMergeAlgorithm() = default;

  bool requires_conjugate_hierarchy() const override { return true; }

  bayesmix::AlgorithmId get_id() const override {
    return bayesmix::AlgorithmId::SplitMerge;
  }

  void read_params_from_proto(
      const bayesmix::AlgorithmParams &params) override;

  std::shared_ptr<BaseAlgorithm> clone() override {
    auto out = std::make_shared<SplitAndMergeAlgorithm>(*this);
    out->set_mixing(mixing->clone());
    out->set_hierarchy(unique_values[0]->deep_clone());
    return out;
  }

 protected:
  void print_startup_message() const override;

  bool update_hierarchy_params() override { return true; }

  void sample_allocations() override;

  void sample_unique_values() override;

  Eigen::VectorXd lpdf_marginal_component(
      std::shared_ptr<AbstractHierarchy> hier, const Eigen::MatrixXd &grid,
      const Eigen::RowVectorXd &covariate) const override;

  void compute_restricted_gs_data_idx(const unsigned int first_random_idx,
                                      const unsigned int second_random_idx);

  void compute_restricted_gs_unique_values(
      const unsigned int first_random_idx,
      const unsigned int second_random_idx);

  /* Auxiliary function that computes the log of the ratio of the likelihoods
   * and of the priors between the proposal and the original clustering.
   * The two ratios are returned via the reference parameters.
   * The parameter "split" represents if the function is called in the "split"
   * case or in the "merge" case.
   */
  std::pair<double, double> compute_log_ratios(
      const unsigned int first_random_idx,
      const unsigned int second_random_idx, bool split);

  void split_or_merge(const unsigned int first_random_idx,
                      const unsigned int second_random_idx);

  /* Updates unique_values and allocations when a split or a merge proposal
   * is accepted.
   */
  void proposal_update_allocations(const unsigned int first_random_idx,
                                   const unsigned int second_random_idx,
                                   const bool split);

  void full_gs();

  /* If return_log_res_prod is true, the function returns the log of the
   * probability of the transition that restricted_gs_unique_values has done
   * in the function.
   * If return_log_res_prod is false, ignore the return value.
   *
   * If step_to_original_clust is true, the function moves all the points
   * in restricted_gs_data_idx to their original clustering configuration.
   */
  double restricted_gs(const double first_random_idx,
                       bool return_log_res_prod = false,
                       bool step_to_original_clust = false);

  /* Vector that contains the indexes of the data points that are considered
   * in the restricted Gibbs sampling.
   */
  std::vector<unsigned int> restricted_gs_data_idx;

  /* Vector of two elements that represent the two temporary clusters used
   * in the restricted Gibbs sampling.
   * Attention! Each cluster contains also the point that is drawn at the
   * beginning of the MH iteration and assigned to that cluster, i.e. point
   * first_random_idx for the first cluster and point second_random_idx for
   * the second cluster.
   */
  std::vector<std::shared_ptr<AbstractHierarchy>> restricted_gs_unique_values;

  /* Vector that associates each element of restricted_GS_data_idx to one of
   * the two temporary clusters.
   */
  std::vector<bool> allocations_restricted_gs;

  // Number of restricted GS scans for each MH step.
  unsigned int n_restr_gs_updates = 5;

  // Number of MH updates for each iteration of Split and Merge algorithm.
  unsigned int n_mh_updates = 1;

  // Number of full GS scans for each iteration of Split and Merge algorithm.
  unsigned int n_full_gs_updates = 1;
};

#endif  // BAYESMIX_ALGORITHMS_SPLIT_AND_MERGE_ALGORITHM_H_
