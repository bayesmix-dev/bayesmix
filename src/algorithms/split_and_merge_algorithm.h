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
  SplitAndMergeAlgorithm() = default;

  SplitAndMergeAlgorithm(unsigned int T_, unsigned int K_, unsigned int M_):
  	T(T_), K(K_), M(M_) {}

  ~SplitAndMergeAlgorithm() = default;

  bool requires_conjugate_hierarchy() const override { return true; }

  bayesmix::AlgorithmId get_id() const override {
    return bayesmix::AlgorithmId::SplitMerge;
  }

 protected:
  std::vector<unsigned int> S;
  unsigned int T=5;
  unsigned int K=1;
  unsigned int M=1;
  unsigned int LabI;

  void print_startup_message() const override;

  void sample_allocations() override;

  void sample_unique_values() override;

  Eigen::VectorXd lpdf_marginal_component(
      std::shared_ptr<AbstractHierarchy> hier, const Eigen::MatrixXd &grid,
      const Eigen::RowVectorXd &covariate) const override;

  void compute_S(const unsigned int i, const unsigned int j);

  std::vector<unsigned int> compute_C_launch(const unsigned int i,
  	const unsigned int j);

  // The function is void because we update the allocation labels directly 
  // inside the function.
  void split_or_merge(std::vector<unsigned int>& cl, const unsigned int i,
  	const unsigned int j);

  // This function was __MH in Python
  bool accepted_proposal(const double acRa) const;

  void full_GS();

  void restricted_GS(std::vector<unsigned int>& cl, const unsigned int i, 
  	const unsigned int j, double &res_prod) const;

  // TODO: Proposal swap
};


#endif  // BAYESMIX_ALGORITHMS_SPLIT_AND_MERGE_ALGORITHM_H_
