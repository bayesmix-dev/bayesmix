#ifndef LOSSFUNCTIONHEADERDEF
#define LOSSFUNCTIONHEADERDEF

#include <vector>
#include <set>
#include <stdexcept>

// !This class implements a Loss Function for two partitions (clusters).
// !This is the base class. The common information for Loss functions is implemented here.

class LossFunction
{
protected:
    std::vector<int> cluster1;
    int K1; // nº of groups in cluster1
    std::vector<int> cluster2;
    int K2; // nº of groups in cluster2
    int N;  // nº of points

public:
    void SetCluster(std::vector<int> cluster1_,
                    std::vector<int> cluster2_);           // Populate the members
    int GetNumberOfGroups(std::vector<int> cluster);       // returns the nº of groups in a cluster (ie K)
    int ClassCounter(std::vector<int> cluster, int index); // returns how many times the group "index" appears inside "cluster" (n(a,g) in the article)
    int ClassCounterExtended(std::vector<int> cluster1,
                             std::vector<int> cluster2, int g, int h); // mutual count of how many times the group "g" and "h" appear inside "cluster1" and "cluster2" simultaneously (n_{g,h} ^ (a,z) in the article)
    virtual double Loss() = 0;                                         //* Loss Function to be implemented in the extended classes.
};
#endif