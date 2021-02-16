#include "VariationInformation.hpp"
#include <set>
using namespace std;

VariationInformation::VariationInformation(bool normalise_) {
  cout << "VariationInformation Constructor" << endl;
  normalise = normalise_;
}

set<int, greater<int>> getClasses(Eigen::VectorXi &partition) {
  set<int, greater<int>> s;
  int size = partition.size();

  for (int i = 0; i < size; i++) {
    s.insert(partition(i));
  }

  return s;
}

double VariationInformation::Entropy(Eigen::VectorXi &cluster) {
  double H = 0.0;
  // int K = GetNumberOfGroups(cluster);
  int nbr = cluster.size();
  set<int, greater<int>> classes = getClasses(cluster);

  for (auto i : classes) { // Possible optimization --> TODO: test it
    int n = ClassCounter(cluster, i + 1);

    double p = (double)n / nbr;

    // x*log(x) = 0, if x = 0
    if (fabs(p) >= 1.0e-9)  // ie p != 0
    {
      H += p * log2(p);
    }
  }
//  for (int i = 0; i < nbr; i++) {
//    int n = ClassCounter(cluster, i + 1);
//
//    double p = (double)n / nbr;
//
//    // x*log(x) = 0, if x = 0
//    if (fabs(p) >= 1.0e-9)  // ie p != 0
//    {
//      H += p * log2(p);
//    }
//  }

  return -H;
}



double VariationInformation::JointEntropy() {
  double H = 0.0;
  for (int g = 0; g < N; g++) {
    double tmp = 0;
    for (int h = 0; h < N; h++) {
      double p =
          (double)ClassCounterExtended(*cluster1, *cluster2, 1 + g, 1 + h) / N;
      // x*log(x) = 0, if x = 0
      if (fabs(p) >= 1.0e-9)  // ie p != 0
      {
        tmp += p * log2(p);
      }
    }
    H += tmp;
  }

  return -H;
}

double VariationInformation::MutualInformation() {
  return Entropy(*cluster1) + Entropy(*cluster2) - JointEntropy();
}

double VariationInformation::Loss() {
  if (!normalise) {
    return 2 * JointEntropy() - Entropy(*cluster1) - Entropy(*cluster2);
  } else {
    return 1 - MutualInformation() / JointEntropy();
  }
}
