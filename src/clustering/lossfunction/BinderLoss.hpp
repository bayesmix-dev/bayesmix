#ifndef BINDERLOSSHEADER
#define BINDERLOSSHEADER

#include "LossFunction.hpp"

class BinderLoss : public LossFunction {
 private:
  // penalties hyperparameters
  double l1;
  double l2;

 public:
  ~BinderLoss();
  BinderLoss() : BinderLoss(1, 1){};
  BinderLoss(double l1_, double l2_);
  double Loss();
};
#endif