#ifndef HYPERSBASE_HPP
#define HYPERSBASE_HPP

#include <cassert>

class HypersBase {
 protected:
  //! Raises error if the hypers values are not valid w.r.t. their own domain
  virtual void check_hypers_validity() = 0;

 public:
  ~HypersBase() = default;
  HypersBase() = default;

  virtual void print_id() const = 0;
};

#endif  // HYPERSBASE_HPP
