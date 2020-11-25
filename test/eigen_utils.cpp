#include <gtest/gtest.h>

#include <Eigen/Dense>
#include <vector>
#include <src/utils/eigen_utils.hpp>

TEST(vstack, 1) {
    std::vector<Eigen::MatrixXd> mats(3);
    mats[0] = Eigen::MatrixXd::Ones(3, 2);
    mats[1] = Eigen::MatrixXd::Zero(1, 2);
    mats[2] = Eigen::MatrixXd(1, 2);
    mats[2] << 1, 2;

    EXPECT_NO_THROW(bayesmix::vstack(mats));
    Eigen::MatrixXd out = bayesmix::vstack(mats);

    ASSERT_EQ(out(0, 0), mats[0](0, 0));
    ASSERT_EQ(out(3, 0), mats[1](0, 0));
    ASSERT_EQ(out(4, 0), mats[2](0, 0));
    ASSERT_EQ(out(4, 1), mats[2](0, 1));

    mats[2] = Eigen::MatrixXd::Zero(1, 1);
    EXPECT_THROW(bayesmix::vstack(mats), std::invalid_argument);
}
