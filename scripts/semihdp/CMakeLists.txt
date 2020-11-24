cmake_minimum_required(VERSION 3.13)
project(semihdp)
enable_testing()

find_package(PkgConfig REQUIRED)
find_package(GTest REQUIRED)

SET(CMAKE_CXX_FLAGS -pthread)

include_directories(${GTEST_INCLUDE_DIR})

add_executable(simu1 run_simulation1.cpp)
add_executable(simu2 run_simulation2.cpp)
add_executable(simu100 run_simulation_many.cpp)
add_executable(chile run_chile.cpp)

target_link_libraries(simu1 bayesmix)
target_link_libraries(simu2 bayesmix)
target_link_libraries(simu100 bayesmix)
target_link_libraries(chile bayesmix)



