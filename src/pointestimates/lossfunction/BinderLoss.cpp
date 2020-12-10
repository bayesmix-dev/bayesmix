#include "BinderLoss.hpp"
#include <iostream>

using namespace std;

int indicator(bool exp);

BinderLoss::BinderLoss(double l1_, double l2_) : LossFunction()
{
    cout << "Binder Loss constructor" << endl;
    l1 = l1_;
    l2 = l2_;
}

BinderLoss::~BinderLoss() {
    cout << "BinderLoss Destructor" << endl  ;
}

double BinderLoss::Loss()
{
    double var = 0.0;
    int size = cluster1->size();

    for (int i = 0; i < size; i++)
    {
        for (int j = i + 1; j < size; j++)
        {
          var += l1 * indicator((*cluster1)(i) != (*cluster1)(j)) * indicator((*cluster2)(i) == (*cluster2)(j)) +
                   l2 * indicator((*cluster1)(i) == (*cluster1)(j)) * indicator((*cluster2)(i) != (*cluster2)(j));
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