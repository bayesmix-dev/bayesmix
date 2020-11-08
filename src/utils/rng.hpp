#ifndef BAYESMIX_UTILS_RNG_HPP_
#define BAYESMIX_UTILS_RNG_HPP_

#include <random>

namespace bayesmix {
class Rng {
 public:
  static Rng &Instance() {
    static Rng s;
    return s;
  }

  std::mt19937_64 &get() { return mt; }
  void seed(int seed_val) { mt.seed(seed_val); }

 private:
  Rng(int seed_val = 20201103) { mt.seed(seed_val); }
  ~Rng() {}

  Rng(Rng const &) = delete;
  Rng &operator=(Rng const &) = delete;

  std::mt19937_64 mt;
};
}  // namespace bayesmix

#endif  // BAYESMIX_UTILS_RNG_HPP_
