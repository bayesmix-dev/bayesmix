#ifndef BAYESMIX_UTILS_RNG_H_
#define BAYESMIX_UTILS_RNG_H_

#include <random>

//! Simple Random Number Generation class wrapper.
//! This class wraps the C++ standard RNG object and allows the use of any RNG
//! seed. It is implemented as a singleton, so that every object used in the
//! library has access to the same exact RNG engine.
//! This is needed to ensure that the rng stream is well defined and that every
//! random number generation causes an update in the rng state.
//! The main drawback is that this design does not allow for efficient
//! parallelization, as calls to the Rng::Instance() from different threads
//! could cause data races. A preferred solution would be to define the Rng to
//! be thread-local if omp-parallelism over several cores is desired, see:
//! https://stackoverflow.com/q/64937761

namespace bayesmix {
class Rng {
 public:
  //! Returns (and creates if nonexistent) the singleton of this class
  static Rng &Instance() {
    static Rng s;
    return s;
  }

  //! Returns a reference to the underlying RNG object
  std::mt19937 &get() { return mt; }

  //! Sets the RNG seed
  void seed(const int seed_val) { mt.seed(seed_val); }

 private:
  Rng(const int seed_val = 20201103) { mt.seed(seed_val); }
  ~Rng() {}
  Rng(Rng const &) = delete;
  Rng &operator=(Rng const &) = delete;

  //! C++ standard library RNG object
  std::mt19937 mt;
};
}  // namespace bayesmix

#endif  // BAYESMIX_UTILS_RNG_H_
