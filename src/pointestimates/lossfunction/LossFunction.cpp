#include "LossFunction.hpp"

void LossFunction::SetCluster(std::vector<int> cluster1_,
                              std::vector<int> cluster2_)
{
    auto tmp = cluster1_.size();
    if (tmp != cluster2_.size())
    {
        throw std::domain_error("Clusters of different sizes!");
    }

    N = (int)tmp;
    cluster1 = cluster1_;
    K1 = GetNumberOfGroups(cluster1_);
    cluster2 = cluster2_;
    K2 = GetNumberOfGroups(cluster2_);
}

int LossFunction::GetNumberOfGroups(std::vector<int> cluster)
{
    std::set<int> groups;

    for (auto &elem : cluster)
    {
        groups.insert(elem);
    }

    return (int)groups.size();
}

int LossFunction::ClassCounter(std::vector<int> cluster, int index)
{
    int count = 0;
    for (const auto &val : cluster)
    {
        if (val == index)
        {
            count += 1;
        }
    }

    return count;
}

int LossFunction::ClassCounterExtended(std::vector<int> cluster1,
                                       std::vector<int> cluster2, int g, int h)
{
    int count = 0;

    for (std::size_t i = 0; i < cluster1.size(); i++)
    {
        if (cluster1[i] == g && cluster2[i] == h)
        {
            count += 1;
        }
    }

    return count;
}