#ifndef BAYESMIX_MIXINGS_TRUNCATED_SB_MIXING_H_
#define BAYESMIX_MIXINGS_TRUNCATED_SB_MIXING_H_

#include <google/protobuf/message.h>

#include <memory>
#include <stan/math/rev.hpp>
#include <vector>

#include "base_mixing.h"
#include "mixing_id.pb.h"
#include "mixing_prior.pb.h"
#include "src/hierarchies/abstract_hierarchy.h"

namespace TruncSB {
struct State {
  Eigen::VectorXd sticks, logweights;
};
};  // namespace TruncSB

/**
 * Class that represents a truncated stick-breaking process, as shown in
 * Ishwaran and James (2001).
 *
 * A truncated stick-breaking process is a prior for weights
 * \f$ (w_1,...,w_H) \f$ in the H-1 dimensional unit simplex, and is defined as
 * follows:
 *
 * \f[
 *    w_1 &= v_1 \\
 *    w_j &= v_j (1 - v_1) ... (1 - v_{j-1}), \quad \text{for } j=1, ... H-1 \\
 *    w_H &= 1 - (w_1 + w_2 + ... + w_{H-1})
 * \f]
 *
 * The \f$ v_j \f$'s are called sticks and we assume them to be independently
 * distributed as \f$ v_j \sim \text{Beta}(a_j, b_j) \f$.
 *
 * When \f$ a_j = 1 \f$ and \f$ b_j = M \f$, the stick-breaking process is a
 * truncation of the stick-breaking representation of the DP.
 * When \f$ a_j = 1-d \f$ and \f$ b_j = M+id \f$, it is the trunctation of a PY
 * process. Its state is composed of the weights \f$ w_j \f$ in log-scale and
 * the sticks \f$ v_j \f$. For more information about the class, please refer
 * instead to base classes, `AbstractMixing` and `BaseMixing`.
 */

class TruncatedSBMixing : public BaseMixing<TruncatedSBMixing, TruncSB::State,
                                            bayesmix::TruncSBPrior> {
 public:
  TruncatedSBMixing() = default;
  ~TruncatedSBMixing() = default;

  //! Performs conditional update of state, given allocations and unique values
  //! @param unique_values  A vector of (pointers to) Hierarchy objects
  //! @param allocations    A vector of allocations label
  void update_state(
      const std::vector<std::shared_ptr<AbstractHierarchy>> &unique_values,
      const std::vector<unsigned int> &allocations) override;

  //! Read and set state values from a given Protobuf message
  void set_state_from_proto(const google::protobuf::Message &state_) override;

  //! Writes current state to a Protobuf message and return a shared_ptr
  //! New hierarchies have to first modify the field 'oneof val' in the
  //! MixingState message by adding the appropriate type
  std::shared_ptr<bayesmix::MixingState> get_state_proto() const override;

  //! Returns the Protobuf ID associated to this class
  bayesmix::MixingId get_id() const override {
    return bayesmix::MixingId::TruncSB;
  }

  //! Returns whether the mixing is conditional or marginal
  bool is_conditional() const override { return true; }

  Eigen::VectorXd get_sticks() const { return state.sticks; }

  //! Returns the prior shape parameters of the Beta-distributed sticks
  Eigen::MatrixXd get_prior_shape_parameters() const;

  //! Adds `num_sticks` sticks to the state by keep breaking
  //! returns the sum of the new weights
  double keep_breaking(int num_sticks);

  void set_sticks(Eigen::VectorXd sticks);

  bool is_infinite_mixture();

 protected:
  //! Returns mixing weights (for conditional mixings only)
  //! @param log        Whether to return logarithm-scale values or not
  //! @param propto     Whether to include normalizing constants or not
  //! @return           The vector of mixing weights
  Eigen::VectorXd mixing_weights(const bool log,
                                 const bool propto) const override;

  //! Initializes state parameters to appropriate values
  void initialize_state() override;

  //! Returns weights in log-scale computing them from sticks
  Eigen::VectorXd logweights_from_sticks() const;

  std::pair<double, double> get_beta_params(int ind);
};

#endif  // BAYESMIX_MIXINGS_TRUNCATED_SB_MIXING_H_
