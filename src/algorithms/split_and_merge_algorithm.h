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

 protected:
  void initialize() override;

  void print_startup_message() const override;

  /* We need to update the parameters when we add or remove a datum from a
   * cluster to esure that conditional_pred_lpdf returns a correct value.
   */
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
  void compute_log_ratio_like_and_prior(const unsigned int first_random_idx,
                                        const unsigned int second_random_idx,
                                        bool split,
                                        double &log_ratio_prior_prob,
                                        double &log_ratio_likelihoods);

  void split_or_merge(const unsigned int first_random_idx,
                      const unsigned int second_random_idx);

  void split(const unsigned int first_random_idx,
             const unsigned int second_random_idx);

  void merge(const unsigned int first_random_idx,
             const unsigned int second_random_idx);

  bool accepted_proposal(const double acRa) const;

  /* Updates unique_values and allocations when a split or a merge proposal
   * is accepted.
   */
  void proposal_update_allocations(const unsigned int first_random_idx,
                                   const unsigned int second_random_idx,
                                   const bool split);

  void full_gibbs_sampling();

  /* After picking at random two points, the Split and Merge algorithm
   * takes all the points that are in the same clusters as the first two and
   * assigns to each of them, at random, one of two temporary clusters. Then,
   * it updates these assignments by doing a restricted Gibbs sampling,
   * restricted in the sense that it considers only the points cited above
   * and only the two temporary clusters.
   * The result of this operation is used to compute the acceptance
   * probability for the split or merge proposal.
   * For more information, check the article on Split and Merge by
   * Jain and Neal (2004).
   *
   * This function computes one iteration of the restricted Gibbs sampling.
   *
   * If return_log_res_prod is true, the function returns the log of the
   * probability of the transition that restricted_gs_unique_values has done
   * in the function.
   * If return_log_res_prod is false, ignore the return value.
   */
  double restricted_gibbs_sampling(bool return_log_res_prod = false);

  /* Vector that contains the indexes of the data points that are considered
   * in the restricted Gibbs sampling.
   *
   * For more information about restricted Gibbs Sampling, check out the
   * comment on function restricted_gibbs_sampling().
   */
  std::vector<unsigned int> restricted_gs_data_idx;

  /* Vector of two elements that represent the two temporary clusters used
   * in the restricted Gibbs sampling.
   * Attention! Each cluster contains also the point that is drawn at the
   * beginning of the MH iteration and assigned to that cluster, i.e. point
   * first_random_idx for the first cluster and point second_random_idx for
   * the second cluster.
   *
   * For more information about restricted Gibbs Sampling, check out the
   * comment on function restricted_gibbs_sampling().
   */
  std::vector<std::shared_ptr<AbstractHierarchy>> restricted_gs_unique_values;

  /* Vector that associates each element of restricted_GS_data_idx to one of
   * the two temporary clusters.
   *
   * For more information about restricted Gibbs Sampling, check out the
   * comment on function restricted_gibbs_sampling().
   */
  std::vector<bool> allocations_restricted_gs;

  /* Number of restricted GS scans for each MH step.
   *
   * For more information about restricted Gibbs Sampling, check out the
   * comment on function restricted_gibbs_sampling().
   */
  unsigned int n_restr_gs_updates = 5;

  // Number of MH updates for each iteration of Split and Merge algorithm.
  unsigned int n_mh_updates = 1;

  // Number of full GS scans for each iteration of Split and Merge algorithm.
  unsigned int n_full_gs_updates = 1;
};

#endif  // BAYESMIX_ALGORITHMS_SPLIT_AND_MERGE_ALGORITHM_H_
