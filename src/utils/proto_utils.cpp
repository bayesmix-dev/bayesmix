#include "proto_utils.hpp"

//! \param un_val Unique value in Protobuf-object form
//! \return       Matrix version of un_val
Eigen::MatrixXd bayesmix::proto_param_to_matrix(const Param &un_val) {
  Eigen::MatrixXd par_matrix = Eigen::MatrixXd::Zero(
      un_val.par_cols(0).elems_size(), un_val.par_cols_size());

  // Loop over unique values to copy them one at a time
  for (size_t i = 0; i < un_val.par_cols_size(); i++) {
    for (size_t j = 0; j < un_val.par_cols(i).elems_size(); j++) {
      par_matrix(j, i) = un_val.par_cols(i).elems(j);
    }
  }
  return par_matrix;
}
