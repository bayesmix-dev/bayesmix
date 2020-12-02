#include "BinderLoss.hpp"

using namespace std;

int indicator(bool exp);

BinderLoss::BinderLoss(double l1_, double l2_)
{
    l1 = l1_;
    l2 = l2_;
}

double BinderLoss::Loss()
{
    double var = 0.0;
    size_t size = cluster1.size();

    for (size_t i = 0; i < size; i++)
    {
        for (size_t j = i + 1; j < size; j++)
        {
            var += l1 * indicator(cluster1[i] != cluster1[j]) * indicator(cluster2[i] == cluster2[j]) +
                   l2 * indicator(cluster1[i] == cluster1[j]) * indicator(cluster2[i] != cluster2[j]);
        }
    }

    return var;
}

int indicator(bool exp)
{
    if (exp)
    {
        return 1;
    }
    else
    {
        return 0;
    }
}