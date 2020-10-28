#include "Algorithm.hpp"

//! \param iter Number of the current iteration
//! \return     Protobuf-object version of the current state
State Algorithm::get_state_as_proto(unsigned int iter) {
  // Transcribe allocations vector
  MarginalState iter_out;
  *iter_out.mutable_cluster_allocs() = {allocations.begin(),
                                        allocations.end()};

  // Transcribe unique values vector
  for (size_t i = 0; i < unique_values.size(); i++) {
    MarginalState::ClusterVal* clusval = iter_out.add_cluster_vals();
    unique_values[i]->get_state_as_proto(clusval);
    
  //   UniqueValues uniquevalues_temp;



  //   for (size_t j = 0; j < unique_values[i]->get_state().size(); j++) {
  //     Eigen::MatrixXd par_temp = unique_values[i]->get_state()[j];
  //     Param par_temp_proto;
  //     for (size_t k = 0; k < par_temp.cols(); k++) {
  //       Par_Col col_temp;
  //       for (size_t h = 0; h < par_temp.rows(); h++) {
  //         col_temp.add_elems(par_temp(h, k));
  //       }
  //       par_temp_proto.add_par_cols();
  //       *par_temp_proto.mutable_par_cols(j) = col_temp;
  //     }
  //     uniquevalues_temp.add_params();
  //     *uniquevalues_temp.mutable_params(j) = par_temp_proto;
  //   }
  //   iter_out.add_uniquevalues();
  //   *iter_out.mutable_uniquevalues(i) = uniquevalues_temp;
  // }
  }
  return iter_out;
}
