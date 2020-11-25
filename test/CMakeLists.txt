cmake_minimum_required(VERSION 3.13.0)
project(test_bayesmix)
enable_testing()

find_package(PkgConfig REQUIRED)
find_package(GTest REQUIRED)

set(CMAKE_CXX_FLAGS -pthread)

include_directories(${GTEST_INCLUDE_DIR})

<<<<<<< HEAD
add_executable(test_bayesmix write_proto.cpp proto_utils.cpp rng.cpp
               hierarchies.cpp lpdf.cpp priors.cpp distributions.cpp)

target_link_libraries(test_bayesmix ${GTEST_LIBRARY} ${GTEST_MAIN_LIBRARY}
                      bayesmix)
=======
add_executable(test_bayesmix $<TARGET_OBJECTS:bayesmix>
  write_proto.cpp
  proto_utils.cpp
  rng.cpp
  hierarchies.cpp
  lpdf.cpp
  priors.cpp
  eigen_utils.cpp
)
target_include_directories(test_bayesmix PUBLIC ${INCLUDE_PATHS})
target_link_libraries(test_bayesmix PUBLIC
  ${LINK_LIBRARIES} ${GTEST_LIBRARY} ${GTEST_MAIN_LIBRARY}
)
target_compile_options(test_bayesmix PUBLIC ${COMPILE_OPTIONS})
>>>>>>> 49ab2ad012b4d870ec8a7ef472d2674af5e21228

add_test(NAME runtest COMMAND test_bayesmix)
