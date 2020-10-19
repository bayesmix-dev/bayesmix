#ifndef CONDITIONALALGORITHM_HPP
#define CONDITIONALALGORITHM_HPP

#include "Algorithm.hpp"

template <template <class> class Hierarchy, class Hypers, class Mixing>
class ConditionalAlgorithm : public Algorithm<Hierarchy, Hypers, Mixing> {
public:
  virtual void eval_density(const Eigen::MatrixXd &grid,
                            BaseCollector *const collector) override;
};

#include "ConditionalAlgorithm.imp.hpp"

#endif  // CONDITIONALALGORITHM_HPP
