#include <iostream>

#include "../../../lib/math/lib/eigen_3.3.9/Eigen/Dense"
#include "BinderLoss.hpp"
#include "VariationInformation.hpp"

using namespace std;

// To run this main.cpp file, run the following CLI in a terminal:
// make loss

int main() {
  // Initialize the clusters (N = 5)
  Eigen::VectorXi c1(5);  // K = 3
  c1 << 1, 1, 1, 2, 3;

  Eigen::VectorXi c2(5);  // K = 2
  c2 << 1, 1, 2, 2, 2;

  // Call the desired loss functions
  cout << "[CONSTRUCTORS]" << endl;
  BinderLoss binder_loss(1.0, 1.0);  // l1 = l2 = 1 (penalties)
  BinderLoss binder_loss_default;
  VariationInformation vi_loss(false);      // normalise = false
  VariationInformation vi_loss_norm(true);  // normalise = true
  cout << endl << endl;

  // Populate the members for each loss above
  binder_loss.SetCluster(c1, c2);
  binder_loss_default.SetCluster(c1, c2);
  vi_loss.SetCluster(c1, c2);
  vi_loss_norm.SetCluster(c1, c2);

  // Calculate the loss function
  double bl_value = binder_loss.Loss();
  double bld_value = binder_loss_default.Loss();
  double vi_value = vi_loss.Loss();
  double vi_norm_value = vi_loss_norm.Loss();

  // Print the calculated values
  cout << "Binder Loss (this should be 5): " << bl_value << '\n';
  cout << "---------------------------------" << '\n';
  cout << "Binder Loss with default parameters (this should be 5): "
       << bld_value << '\n';
  cout << "---------------------------------" << '\n';
  cout << "(Unormalised) VI Loss: " << vi_value << '\n';
  cout << "---------------------------------" << '\n';
  cout << "(Normalised) VI Loss: " << vi_norm_value << '\n';

  cout << endl << "[DESTRUCTORS]" << endl;

  return 0;
}
