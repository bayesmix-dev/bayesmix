#include "LossFunction.hpp"
#include <iostream>

LossFunction::LossFunction() : cluster1(0), cluster2(0) {
  std::cout << "Loss Function Constructor" << std::endl;
  cluster1 = new Eigen::VectorXi();
  cluster2 = new Eigen::VectorXi();
}

LossFunction::~LossFunction() {
  std::cout << "Loss Function virtual destructor" << std::endl;
  delete cluster1;
  delete cluster2;
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

    *cluster1 = cluster1_;
    K1 = GetNumberOfGroups(cluster1_);
    *cluster2 = cluster2_;
    K2 = GetNumberOfGroups(cluster2_);
}

void LossFunction::SetFirstCluster(Eigen::VectorXi cluster1_)
{
    N = cluster1_.rows();
    *cluster1 = cluster1_;
    K1 = GetNumberOfGroups(cluster1_);
}

void LossFunction::SetSecondCluster(Eigen::VectorXi cluster2_) {
    N = cluster2_.rows();
    *cluster2 = cluster2_;
    K2 = GetNumberOfGroups(cluster2_);
}

Eigen::VectorXi * LossFunction::GetCluster(int i) const {
    switch(i) {
      case 1 : return cluster1;
      case 2: return cluster2;
      default : throw std::domain_error("Wrong cluster index.");
    }
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

ostream &operator<<(ostream &out, LossFunction const * loss_function) {
    out << "["    << typeid(*loss_function).name();
    out << "; cluster1 = " << loss_function->GetCluster(1)->transpose();
    out << "; cluster2 = " << loss_function->GetCluster(2)->transpose();
    out << "]";
    return out;
}
