cmake_minimum_required(VERSION 3.13.0)
project(test_bayesmix)
enable_testing()

find_package(PkgConfig REQUIRED)
find_package(GTest REQUIRED)

set(CMAKE_CXX_FLAGS -pthread)

include_directories(${GTEST_INCLUDE_DIR})

add_executable(test_bayesmix $<TARGET_OBJECTS:bayesmix>
  # write_proto.cc
  # proto_utils.cc
  # rng.cc
  # hierarchies.cc
  # lpdf.cc
  # priors.cc
  # eigen_utils.cc
  # distributions.cc
  # semi_hdp.cc
  # collectors.cc
  # runtime.cc
  crtp.cc
  crtp_aux.h
)

target_include_directories(test_bayesmix PUBLIC ${INCLUDE_PATHS})
target_link_libraries(test_bayesmix PUBLIC
  ${LINK_LIBRARIES} ${GTEST_LIBRARY} ${GTEST_MAIN_LIBRARY}
)
target_compile_options(test_bayesmix PUBLIC ${COMPILE_OPTIONS})

add_test(NAME runtest COMMAND test_bayesmix)
