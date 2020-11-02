cmake_minimum_required(VERSION 3.13)
project(test_bayesmix)
enable_testing()

find_package(PkgConfig REQUIRED)

find_package(GTest REQUIRED)

SET(CMAKE_CXX_FLAGS -pthread)

include_directories(${GTEST_INCLUDE_DIR})

add_executable(test_bayesmix get_state_as_proto_test.cpp proto_utils.cpp)


target_link_libraries(test_bayesmix ${GTEST_LIBRARY} ${GTEST_MAIN_LIBRARY} bayesmix)
add_test(NAME runtest COMMAND test_bayesmix)
