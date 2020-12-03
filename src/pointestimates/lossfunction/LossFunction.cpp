#include "LossFunction.hpp"
#include <iostream>

void LossFunction::SetCluster(Eigen::VectorXi cluster1_,
                              Eigen::VectorXi cluster2_)
{
    auto n_rows = cluster1_.rows();

    if (n_rows != cluster2_.rows())
    {
        throw std::domain_error("Clusters of different sizes!");
    }

    N = (int) n_rows;

    cluster1 = cluster1_;
    K1 = GetNumberOfGroups(cluster1_);
    cluster2 = cluster2_;
    K2 = GetNumberOfGroups(cluster2_);
}

int LossFunction::GetNumberOfGroups(Eigen::VectorXi cluster)
{
    std::set<int> groups;

    for (int i = 0; i < cluster.rows(); i++)
    {
        groups.insert(cluster(i));
    }

    std::cout << "GetNumberOfGroups :" << (int)groups.size() << std::endl;
    return (int)groups.size();
}

int LossFunction::ClassCounter(Eigen::VectorXi cluster, int index)
{
    int count = 0;
    for (int i = 0; i < N; i++)
    {
        if (cluster(i) == index)
        {
            count += 1;
        }
    }

    return count;
}

int LossFunction::ClassCounterExtended(Eigen::VectorXi cluster1, Eigen::VectorXi cluster2, int g, int h)
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