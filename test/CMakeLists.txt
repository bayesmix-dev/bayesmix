cmake_minimum_required(VERSION 3.13.0)
project(test_bayesmix)
enable_testing()

find_package(PkgConfig REQUIRED)
set(CMAKE_CXX_FLAGS -pthread)

include(FetchContent)
FetchContent_Declare(
  googletest
  GIT_REPOSITORY https://github.com/google/googletest.git
  GIT_TAG        9c332145b71c36a5bad9688312c79184f98601ff # release-1.13
)
# For Windows: Prevent overriding the parent project's compiler/linker settings
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

add_executable(test_bayesmix $<TARGET_OBJECTS:bayesmix>
  write_proto.cc
  proto_utils.cc
  likelihoods.cc
  prior_models.cc
  hierarchies.cc
  lpdf.cc
  eigen_utils.cc
  distributions.cc
  semi_hdp.cc
  collectors.cc
  runtime.cc
  rng.cc
  logit_sb.cc
  gradient.cc
  slice_sampler.cc
)

target_include_directories(test_bayesmix PUBLIC ${INCLUDE_PATHS})
target_link_libraries(test_bayesmix PUBLIC
  ${LINK_LIBRARIES} gtest_main gtest
)
target_compile_options(test_bayesmix PUBLIC ${COMPILE_OPTIONS})

add_test(NAME runtest COMMAND test_bayesmix)
