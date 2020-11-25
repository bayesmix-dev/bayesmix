cmake_minimum_required(VERSION 3.13.0)
project(bayesmix)

find_package(PkgConfig REQUIRED)

set (CMAKE_CXX_STANDARD 14)
set(CMAKE_BUILD_TYPE Debug)

set(CMAKE_BUILD_WITH_INSTALL_RPATH ON)
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)
set(BASEPATH "${CMAKE_SOURCE_DIR}")
set(INCLUDE_PATHS
  ${BASEPATH}
  ${CMAKE_CURRENT_SOURCE_DIR}/lib/math
  ${CMAKE_CURRENT_SOURCE_DIR}/lib/math/lib/boost_1.72.0
  ${CMAKE_CURRENT_SOURCE_DIR}/lib/math/lib/eigen_3.3.7
  ${CMAKE_CURRENT_SOURCE_DIR}/lib/math/lib/tbb_2019_U8/include
  ${CMAKE_CURRENT_SOURCE_DIR}/lib/protobuf/src
)
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
add_subdirectory(test)
