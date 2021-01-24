#include <gtest/gtest.h>

#include <Eigen/Dense>
#include <vector>

#include "../src/collectors/file_collector.hpp"
#include "../src/collectors/memory_collector.hpp"
#include "../src/utils/proto_utils.hpp"
#include "matrix.pb.h"

TEST(collectors, memory) {
  MemoryCollector coll;
  coll.start_collecting();

  std::vector<Eigen::VectorXd> chain(5);
  for (int i = 0; i < 5; i++) {
    chain[i] = Eigen::VectorXd::Ones(3) * i;
    bayesmix::Vector curr;
    to_proto(chain[i], &curr);
    coll.collect(curr);
  }
  coll.finish_collecting();

  int iter = 0;
  bool keep = true;
  while (keep) {
    bayesmix::Vector curr;
    keep = coll.get_next_state(&curr);
    if (!keep) {
      iter--;
      break;
    }
    ASSERT_EQ(curr.size(), 3);
    ASSERT_EQ(curr.data(0), iter);
    iter++;
  }

  ASSERT_EQ(chain[iter](0), chain[4][0]);
}

TEST(collectors, file_writing) {
  FileCollector coll("test.recordio");
  coll.start_collecting();
  std::vector<Eigen::VectorXd> chain(5);
  for (int i = 0; i < 5; i++) {
    chain[i] = Eigen::VectorXd::Ones(3) * i;
    bayesmix::Vector curr;
    to_proto(chain[i], &curr);
    coll.collect(curr);
  }
  coll.finish_collecting();
}

TEST(collectors, file_reading) {
  FileCollector coll("test.recordio");
  coll.start_collecting();

  std::vector<Eigen::VectorXd> chain(5);
  for (int i = 0; i < 5; i++) {
    chain[i] = Eigen::VectorXd::Ones(3) * i;
    bayesmix::Vector curr;
    to_proto(chain[i], &curr);
    coll.collect(curr);
  }
  coll.finish_collecting();

  FileCollector coll2("test.recordio");
  int iter = 0;
  bool keep = true;
  while (keep) {
    bayesmix::Vector curr;
    keep = coll2.get_next_state(&curr);
    if (!keep) {
      iter--;
      break;
    }
    ASSERT_EQ(curr.size(), 3);
    ASSERT_EQ(curr.data(0), iter);
    iter++;
  }
  ASSERT_EQ(chain[iter](0), chain[4][0]);
}
