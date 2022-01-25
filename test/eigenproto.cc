#include "src/utils/eigenproto.h"

#include <gtest/gtest.h>


TEST(eigenproto, vector_conversion) {
  Eigen::VectorXd v = Eigen::VectorXd::Random(3);
  MyVectorType mv(v);
  bayesmix::Vector vproto;
  bayesmix::to_proto(mv, &vproto);

  bayesmix::Vector mvproto = mv;

  EXPECT_EQ(vproto.DebugString(), mvproto.DebugString());  
}