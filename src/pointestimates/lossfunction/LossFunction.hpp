#ifndef LOSSFUNCTIONHEADERDEF
#define LOSSFUNCTIONHEADERDEF

#include "../../../lib/math/lib/eigen_3.3.7/Eigen/Dense"
#include <set>
#include <stdexcept>

// !This class implements a Loss Function for two partitions (clusters).
// !This is the base class. The common information for Loss functions is implemented here.

class LossFunction
{
protected:
    Eigen::VectorXi * cluster1;
    int K1; // nº of groups in cluster1
    Eigen::VectorXi * cluster2;
    int K2; // nº of groups in cluster2
    int N;  // nº of points

public:
    LossFunction();
    virtual ~LossFunction() = 0;
    void SetCluster(Eigen::VectorXi cluster1_,
                    Eigen::VectorXi cluster2_);           // Populate the members
    void SetFirstCluster(Eigen::VectorXi cluster1_);
    void SetSecondCluster(Eigen::VectorXi cluster2_);

    int GetNumberOfGroups(Eigen::VectorXi cluster);       // returns the nº of groups in a cluster (ie K)
    int ClassCounter(Eigen::VectorXi cluster, int index); // returns how many times the group "index" appears inside "cluster" (n(a,g) in the article)
    int ClassCounterExtended(Eigen::VectorXi cluster1,
                             Eigen::VectorXi cluster2, int g, int h); // mutual count of how many times the group "g" and "h" appear inside "cluster1" and "cluster2" simultaneously (n_{g,h} ^ (a,z) in the article)
    virtual double Loss() = 0;                                         //* Loss Function to be implemented in the extended classes.
};
#endif