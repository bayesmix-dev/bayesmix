#include "LossFunction.hpp"
#include <iostream>

LossFunction::LossFunction() : cluster1(0), cluster2(0) {
  std::cout << "Loss Function Constructor" << std::endl;
}

LossFunction::~LossFunction() {
  std::cout << "Loss Function virtual destructor" << std::endl;
}


void LossFunction::SetCluster(Eigen::VectorXi cluster1_,
                              Eigen::VectorXi cluster2_)
{
    auto n_rows = cluster1_.rows();

    if (n_rows != cluster2_.rows())
    {
        throw std::domain_error("Clusters of different sizes!");
    }

    N = (int) n_rows;

    cluster1 = &cluster1_;
    K1 = GetNumberOfGroups(cluster1_);
    cluster2 = &cluster2_;
    K2 = GetNumberOfGroups(cluster2_);
}

void LossFunction::SetFirstCluster(Eigen::VectorXi cluster1_)
{
    auto n_rows = cluster1_.rows();

    /* Since we don't set clusters at the same time, there are many cases
     * where c1 and c2 are from differents sizes.
    if (n_rows != cluster2.rows())
    {
      throw std::domain_error("Clusters of different sizes!");
    }
    */

    N = (int) n_rows;
    cluster1 = &cluster1_;
    K1 = GetNumberOfGroups(cluster1_);
}

void LossFunction::SetSecondCluster(Eigen::VectorXi cluster2_) {
  auto n_rows = cluster2_.rows();

  /*
  if (n_rows != cluster1.rows())
  {
    throw std::domain_error("Clusters of different sizes!");
  }
  */

  N = (int) n_rows;

  cluster2 = &cluster2_;
  K2 = GetNumberOfGroups(cluster2_);
}

int LossFunction::GetNumberOfGroups(Eigen::VectorXi cluster)
{
    std::set<int> groups;

    for (int i = 0; i < cluster.rows(); i++)
    {
        groups.insert(cluster(i));
    }

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

    for (int i = 0; i < cluster1.size(); i++)
    {
        if (cluster1[i] == g && cluster2[i] == h)
        {
            count += 1;
        }
    }

    return count;
}