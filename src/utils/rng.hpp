#ifndef RNG_HPP
#define RNG_HPP

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
    Rng(int seed_val = 123414) { mt.seed(seed_val); }
    ~Rng() {}
  
    Rng(Rng const &) = delete;
    Rng &operator=(Rng const &) = delete;
  
    std::mt19937_64 mt;
  };
}

#endif  // RNG_HPP
