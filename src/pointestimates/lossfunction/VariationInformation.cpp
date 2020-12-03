#include "VariationInformation.hpp"

using namespace std;

VariationInformation::VariationInformation(bool normalise_)
{
    normalise = normalise_;
}

double VariationInformation::Entropy(Eigen::VectorXi cluster)
{
    double H = 0.0;
    int K = GetNumberOfGroups(cluster);
    int nbr = (int)cluster.size();

    for (int i = 0; i < K; i++)
    {
        int n = ClassCounter(cluster, i + 1);
        double p = (double)n / nbr;

        // x*log(x) = 0, if x = 0
        if (fabs(p) >= 1.0e-9) // ie p != 0
        {
            H += p * log2(p);
        }
    }

    return -H;
}

double VariationInformation::JointEntropy()
{
    double H = 0.0;
    for (int g = 0; g < K1; g++)
    {
        double tmp = 0;
        for (int h = 0; h < K2; h++)
        {
            double p = (double)ClassCounterExtended(cluster1, cluster2, 1 + g, 1 + h) / N;
            // x*log(x) = 0, if x = 0
            if (fabs(p) >= 1.0e-9) // ie p != 0
            {
                tmp += p * log2(p);
            }
        }
        H += tmp;
    }

    return -H;
}

double VariationInformation::MutualInformation()
{

    return Entropy(cluster1) + Entropy(cluster2) - JointEntropy();
}

double VariationInformation::Loss()
{
    if (!normalise)
    {
        return 2 * JointEntropy() - Entropy(cluster1) - Entropy(cluster2);
    }
    else
    {
        return 1 - MutualInformation() / JointEntropy();
    }
}