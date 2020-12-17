cmake_minimum_required(VERSION 3.13.0)
project(bayesmix)


set (CMAKE_CXX_STANDARD 14)
set(CMAKE_BUILD_TYPE RelWithDebInfo)
# set(CMAKE_BUILD_TYPE Release)


set(CMAKE_CXX_FLAGS_RELEASE "-O3 -march=native -msse2 -funroll-loops -ftree-vectorize -fopenmp")


find_package(PkgConfig REQUIRED)
find_package(OpenMP REQUIRED)



set(CMAKE_CXX_STANDARD 14)
set(CMAKE_BUILD_TYPE Debug)

set(CMAKE_BUILD_WITH_INSTALL_RPATH ON)
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)
set(BASEPATH "${CMAKE_CURRENT_LIST_DIR}")
set(INCLUDE_PATHS
  ${BASEPATH}
  ${CMAKE_CURRENT_LIST_DIR}/lib/math
  ${CMAKE_CURRENT_LIST_DIR}/lib/math/lib/boost_1.72.0
  ${CMAKE_CURRENT_LIST_DIR}/lib/math/lib/eigen_3.3.9
  ${CMAKE_CURRENT_LIST_DIR}/lib/math/lib/tbb_2019_U8/include
)

message("************************* INCLUDE PATHS ${INCLUDE_PATHS}")

message("CMAKE_CURRENT_LIST_DIR is ${CMAKE_CURRENT_LIST_DIR}")  
message("CMAKE_SOURCE_DIR is ${CMAKE_SOURCE_DIR}")  


set(LINK_LIBRARIES tbb pthread protobuf)
set(COMPILE_OPTIONS -D_REENTRANT -fPIC)

# Build library object
add_library(bayesmix OBJECT)
add_subdirectory(src)    # source files
add_subdirectory(proto)  # source files
target_include_directories(bayesmix PUBLIC ${INCLUDE_PATHS})
target_link_libraries(bayesmix PUBLIC ${LINK_LIBRARIES})
target_compile_options(bayesmix PUBLIC ${COMPILE_OPTIONS})

# Build static library
add_library(bayesmixlib $<TARGET_OBJECTS:bayesmix>)

# Build run executable
add_executable(run $<TARGET_OBJECTS:bayesmix> run.cpp)
target_include_directories(run PUBLIC ${INCLUDE_PATHS})
target_link_libraries(run PUBLIC ${LINK_LIBRARIES})
target_compile_options(run PUBLIC ${COMPILE_OPTIONS})

# Build test executable
# add_subdirectory(test)
# add_subdirectory(scripts)

