#include <iostream>
#include <vector>

#include "BinderLoss.hpp"
#include "VariationInformation.hpp"

using namespace std;

// To run this main.cpp file, run the following CLI in a terminal:
// (i) g++ -Wall -O -c LossFunction.cpp, (ii) g++ -Wall -O -c VariationInformation.cpp, (iii) g++ -Wall -O -c BinderLoss.cpp
// (iv) g++ -lm -o loss loss.cpp BinderLoss.o VariationInformation.o LossFunction.o
// (v) ./main

int main(int argc, char const *argv[])
{
    // Initialize the clusters (N = 5)
    vector<int> c1{1, 1, 1, 2, 3}; // K = 3
    vector<int> c2{1, 1, 2, 2, 2}; // K = 2

    // Call the desired loss functions
    BinderLoss binder_loss(1.0, 1.0);        // l1 = l2 = 1 (penalties)
    VariationInformation vi_loss(false);     // normalise = false
    VariationInformation vi_loss_norm(true); // normalise = true

    // Populate the members for each loss above
    binder_loss.SetCluster(c1, c2);
    vi_loss.SetCluster(c1, c2);
    vi_loss_norm.SetCluster(c1, c2);

    // Calculate the loss function
    double bl_value = binder_loss.Loss();
    double vi_value = vi_loss.Loss();
    double vi_norm_value = vi_loss_norm.Loss();

    // Print the calculated values
    cout << "Binder Loss (this should be 5): " << bl_value << '\n';
    cout << "---------------------------------" << '\n';
    cout << "(Unormalised) VI Loss: " << vi_value << '\n';
    cout << "---------------------------------" << '\n';
    cout << "(Normalised) VI Loss: " << vi_norm_value << '\n';

    return 0;
}
