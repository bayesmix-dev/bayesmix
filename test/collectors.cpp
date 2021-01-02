#include <gtest/gtest.h>

#include <Eigen/Dense>
#include <vector>

#include "matrix.pb.h"
#include "../src/collectors/file_collector.hpp"
#include "../src/collectors/memory_collector.hpp"
#include "../src/utils/proto_utils.hpp"

TEST(collectors, memory) {
  MemoryCollector coll;

  std::vector<Eigen::VectorXd> chain(5);
  for (int i = 0; i < 5; i++) {
    chain[i] = Eigen::VectorXd::Ones(3) * i;
    bayesmix::Vector curr;
    to_proto(chain[i], &curr);
    coll.collect(curr);
  }

  int iter = 0;
  bool keep = true;
  while (keep) {
    bayesmix::Vector curr;
    keep = coll.get_next_state(&curr);
    if (!keep) {
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
  coll.start();
  std::vector<Eigen::VectorXd> chain(5);
  for (int i = 0; i < 5; i++) {
    chain[i] = Eigen::VectorXd::Ones(3) * i;
    bayesmix::Vector curr;
    to_proto(chain[i], &curr);
    coll.collect(curr);
  }
  coll.finish();
}

TEST(collectors, file_reading) {
  FileCollector coll("test.recordio");
  coll.start();

  std::vector<Eigen::VectorXd> chain(5);
  for (int i = 0; i < 5; i++) {
    chain[i] = Eigen::VectorXd::Ones(3) * i;
    bayesmix::Vector curr;
    to_proto(chain[i], &curr);
    coll.collect(curr);
  }
  coll.finish();

  FileCollector coll2("test.recordio");
  coll2.start();
  int iter = 0;
  bool keep = true;
  while (keep) {
    bayesmix::Vector curr;
    keep = coll2.get_next_state(&curr);
    if (!keep) {
      break;
    }
    ASSERT_EQ(curr.size(), 3);
    ASSERT_EQ(curr.data(0), iter);
    iter++;
  }

  ASSERT_EQ(chain[iter](0), chain[4][0]);
  coll2.finish();
}
