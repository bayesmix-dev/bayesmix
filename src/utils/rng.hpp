#ifndef RNG_HPP
#define RNG_HPP

#include <random>

class Rng
{
public:
    static Rng &Instance()
    {
        static Rng s;
        return s;
    }

    std::mt19937_64 &get() {return mt;}

private:
    Rng(int seed_val = 123414) { mt.seed(seed_val); }
    ~Rng() {}

    Rng(Rng const &) = delete;
    Rng &operator=(Rng const &) = delete;

    std::mt19937_64 mt;
};

#endif