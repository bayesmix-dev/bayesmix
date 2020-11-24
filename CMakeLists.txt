cmake_minimum_required(VERSION 3.13.0)
project(bayesmix)


set (CMAKE_CXX_STANDARD 14)
set(CMAKE_BUILD_TYPE RelWithDebInfo)
# set(CMAKE_BUILD_TYPE Release)


set(CMAKE_CXX_FLAGS_RELEASE "-O3 -march=native -msse2 -funroll-loops -ftree-vectorize -fopenmp")


find_package(PkgConfig REQUIRED)
find_package(OpenMP REQUIRED)



set(CMAKE_BUILD_WITH_INSTALL_RPATH ON)
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)
SET(CMAKE_EXE_LINKER_FLAGS "-fopenmp -Wl,-rpath,/home/mario/dev/stan/math/lib/tbb")

if (OPENMP_FOUND)
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set (CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()

SET(BASEPATH "${CMAKE_SOURCE_DIR}")
INCLUDE_DIRECTORIES("${BASEPATH}")


add_library(bayesmix)
add_subdirectory(src)
add_subdirectory(proto)
add_subdirectory(test)
add_subdirectory(scripts)


target_include_directories(bayesmix PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/
    ${CMAKE_CURRENT_SOURCE_DIR}/lib/math
    ${CMAKE_CURRENT_SOURCE_DIR}/lib/math/lib/boost_1.72.0
    ${CMAKE_CURRENT_SOURCE_DIR}/lib/math/lib/eigen_3.3.7
    ${CMAKE_CURRENT_SOURCE_DIR}/lib/math/lib/tbb_2019_U8/include
    ${CMAKE_CURRENT_SOURCE_DIR}/lib/protobuf/src
)
target_link_directories(bayesmix PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/lib/protobuf/src/.libs
)
target_link_libraries(bayesmix tbb pthread protobuf)
target_compile_options(bayesmix PUBLIC -D_REENTRANT -fPIC)

add_executable(run run.cpp)
target_link_libraries(run bayesmix)
target_include_directories(run PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

target_compile_options(run PUBLIC -D_REENTRANT -fPIC -fopenmp)
