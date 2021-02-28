cmake_minimum_required(VERSION 3.13.0)

find_program(CCACHE_PROGRAM ccache)
if (CCACHE_PROGRAM)
	set_property(GLOBAL PROPERTY RULE_LAUNCH_COMPILE "${CCACHE_PROGRAM}")
endif()

project(bayesmix)

set (CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -march=native -msse2 -funroll-loops
  -ftree-vectorize -fopenmp")

include(FindPythonInterp)
add_custom_target(
 build_tbb ALL
 COMMAND ${PYTHON_EXECUTABLE} ${CMAKE_CURRENT_LIST_DIR}/build_tbb.py
)

find_package(PkgConfig REQUIRED)
find_package(OpenMP REQUIRED)
find_package(Protobuf REQUIRED)

option(DISABLE_TESTS
      "If tests should be compiled or no" OFF)

set(CMAKE_BUILD_WITH_INSTALL_RPATH ON)
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)
set(BASEPATH "${CMAKE_CURRENT_LIST_DIR}")

set(INCLUDE_PATHS
  ${BASEPATH}
  ${BASEPATH}/proto
  ${CMAKE_CURRENT_LIST_DIR}/lib/math
  ${CMAKE_CURRENT_LIST_DIR}/lib/math/lib/boost_1.72.0
  ${CMAKE_CURRENT_LIST_DIR}/lib/math/lib/eigen_3.3.9
  ${CMAKE_CURRENT_LIST_DIR}/lib/math/lib/tbb/include
  ${CMAKE_CURRENT_BINARY_DIR}
)

set(LINK_LIBRARIES ${CMAKE_CURRENT_LIST_DIR}/lib/math/lib/tbb/libtbb.so pthread
  protobuf)
set(COMPILE_OPTIONS -D_REENTRANT -fPIC)

file(GLOB ProtoFiles "${BASEPATH}/proto/*.proto")
PROTOBUF_GENERATE_CPP(PROTO_SRCS PROTO_HDRS ${ProtoFiles})
SET_SOURCE_FILES_PROPERTIES(${PROTO_SRCS} ${PROTO_HDRS} PROPERTIES GENERATED
  TRUE)

get_directory_property(HAS_PARENT PARENT_DIRECTORY)
if(HAS_PARENT)
  set(BAYESMIX_INCLUDE_PATHS ${INCLUDE_PATHS} PARENT_SCOPE)
  set(PROTO_HEADERS ${PROTO_HDRS} PARENT_SCOPE)
  set(PROTO_SOURCES ${PROTO_SRCS} PARENT_SCOPE)
endif()


# Build library object
add_library(bayesmix OBJECT)
add_dependencies(bayesmix build_tbb)
target_sources(bayesmix PUBLIC ${PROTO_SRCS} ${PROTO_HDRS})
add_subdirectory(src)    # source files
target_include_directories(bayesmix PUBLIC ${INCLUDE_PATHS})
target_link_libraries(bayesmix PUBLIC ${LINK_LIBRARIES})
target_compile_options(bayesmix PUBLIC ${COMPILE_OPTIONS})

# Build static library
add_library(bayesmixlib $<TARGET_OBJECTS:bayesmix>)
add_dependencies(bayesmixlib build_tbb)


# Build run executable
add_executable(run $<TARGET_OBJECTS:bayesmix> run.cc)
target_include_directories(run PUBLIC ${INCLUDE_PATHS})
target_link_libraries(run PUBLIC ${LINK_LIBRARIES})
target_compile_options(run PUBLIC ${COMPILE_OPTIONS})

add_executable(run_pe $<TARGET_OBJECTS:bayesmix> run_pe.cpp)
target_include_directories(run_pe PUBLIC ${INCLUDE_PATHS})
target_link_libraries(run_pe PUBLIC ${LINK_LIBRARIES})
target_compile_options(run_pe PUBLIC ${COMPILE_OPTIONS})

add_executable(run_cb $<TARGET_OBJECTS:bayesmix> run_cb.cpp)
target_include_directories(run_cb PUBLIC ${INCLUDE_PATHS})
target_link_libraries(run_cb PUBLIC ${LINK_LIBRARIES})
target_compile_options(run_cb PUBLIC ${COMPILE_OPTIONS})

if (NOT DISABLE_TESTS)
  # Build test executable
  add_subdirectory(test)
endif()
