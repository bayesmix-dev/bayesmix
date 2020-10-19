#ifndef MARGINALALGORITHM_HPP
#define MARGINALALGORITHM_HPP

#include "Algorithm.hpp"

template <template <class> class Hierarchy, class Hypers, class Mixing>
class MarginalAlgorithm : public Algorithm<Hierarchy, Hypers, Mixing> {
 public:
  MarginalAlgorithm(const Hypers &hypers_, const Mixing &mixing_,
                    const Eigen::MatrixXd &data_, const unsigned int init = 0)
      : Algorithm<Hierarchy, Hypers, Mixing>::Algorithm(hypers_, mixing_,
                                                        data_, init) {}
  virtual void eval_density(const Eigen::MatrixXd &grid,
                            BaseCollector *const collector) override;
};

#include "MarginalAlgorithm.imp.hpp"

#endif  // MARGINALALGORITHM_HPP
