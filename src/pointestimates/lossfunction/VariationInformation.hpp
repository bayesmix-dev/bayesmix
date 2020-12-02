#ifndef VARIATIONINFORMATIONHEADER
#define VARIATIONINFORMATIONHEADER

#include <cmath>
#include <iostream>

#include "LossFunction.hpp"

class VariationInformation : public LossFunction
{
private:
    bool normalise;

public:
    VariationInformation(bool normalise_);
    double Entropy(std::vector<int> cluster);
    double JointEntropy();      // This method calculates the value on the members of LossFunction directly
    double MutualInformation(); // This method calculates the value on the members of LossFunction directly
    double Loss();
};
#endif