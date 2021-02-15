#include "VariationInformation.hpp"
#include <map>

using namespace std;

VariationInformation::VariationInformation(bool normalise_)
{
  cout << "VariationInformation Constructor" << endl;
  normalise = normalise_;
}


void rename_label(Eigen::VectorXi &cluster) {
  map<int, int> m;
  int current_index(1);
  for (int i = 0; i < cluster.size(); i ++) {
    if (m[cluster(i)] == 0) {
      m[cluster(i)] = current_index;
      cluster(i) = current_index;
      current_index++;
    } else {
      cluster(i) = m[cluster(i)];
    }
  }
}

double VariationInformation::Entropy(Eigen::VectorXi &cluster)
{
  double H = 0.0;
  int K = GetNumberOfGroups(cluster);
  int nbr = cluster.size();
  rename_label(cluster);

  for (int i = 0; i < nbr; i++)
  {
    int n = ClassCounter(cluster, i + 1);
    double p = (double)n / nbr;

    // x*log(x) = 0, if x = 0
    if (fabs(p) != 0) // ie p != 0
    {
      H += p * log2(p);
    }
  }
  //cout << H << endl;
  return -H;
}

double VariationInformation::JointEntropy()
{
  double H = 0.0;
  rename_label(*cluster1);
  rename_label(*cluster2);
  for (int g = 0; g < K1; g++)
  {
    double tmp = 0;
    for (int h = 0; h < K2; h++)
    {
      double p = (double)ClassCounterExtended(*cluster1, *cluster2, 1 + g, 1 + h) / N;
      // x*log(x) = 0, if x = 0
      if (fabs(p) != 0) // ie p != 0
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

  return Entropy(*cluster1) + Entropy(*cluster2) - JointEntropy();
}

double VariationInformation::Loss()
{
  if (!normalise)
  {
  //cout << Entropy(*cluster1) << endl;
  //cout << Entropy(*cluster2) << endl;
    return Entropy(*cluster1) + Entropy(*cluster2) - 2*MutualInformation();
  }
  else
  {
    return 1 - MutualInformation() / JointEntropy();
  }
}
