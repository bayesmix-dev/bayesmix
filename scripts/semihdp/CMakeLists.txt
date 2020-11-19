cmake_minimum_required(VERSION 3.13)
project(semihdp)
enable_testing()

find_package(PkgConfig REQUIRED)
find_package(GTest REQUIRED)

SET(CMAKE_CXX_FLAGS -pthread)

include_directories(${GTEST_INCLUDE_DIR})

add_executable(semihdp run_simulation1.cpp)

target_link_libraries(semihdp bayesmix)
