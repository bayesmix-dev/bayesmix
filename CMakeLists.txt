cmake_minimum_required(VERSION 3.16.0)
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_LIST_DIR}")
message("CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH}")

# Check if ccache is present to speed up compilation time
find_program(CCACHE_PROGRAM ccache)
if (CCACHE_PROGRAM)
	set_property(GLOBAL PROPERTY RULE_LAUNCH_COMPILE "${CCACHE_PROGRAM}")
endif()

# On MacOS find clang, otherwise find gcc
if(${CMAKE_HOST_SYSTEM_NAME} MATCHES "Darwin")
  find_program(CC_PROGRAM clang REQUIRED)
  find_program(CXX_PROGRAM clang++ REQUIRED)
else()
  find_program(CC_PROGRAM gcc REQUIRED)
  find_program(CXX_PROGRAM g++ REQUIRED)
endif()

# Set C and CXX compilers
if(CC_PROGRAM AND CXX_PROGRAM)
  set(CMAKE_C_COMPILER ${CC_PROGRAM})
  set(CMAKE_CXX_COMPILER ${CXX_PROGRAM})
endif()

# Project name
project(bayesmix)

# Require PkgConfig
find_package(PkgConfig REQUIRED)

# Require OpenMP
find_package(OpenMP REQUIRED)

# Set cmake variables
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_BUILD_TYPE Release)
set(CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake" ${CMAKE_MODULE_PATH})
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -funroll-loops ${OpenMP_CXX_FLAGS} -ftree-vectorize -Wno-deprecated")
set(CMAKE_CXX_FLAGS_DEBUG "-Og")
set(CMAKE_FIND_PACKAGE_PREFER_CONFIG TRUE)
set(CMAKE_FIND_PACKAGE_TARGETS_GLOBAL TRUE)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Include FetchContent
include(FetchContent)

# Set up FetchContent options
set(FETCHCONTENT_BASE_DIR ${CMAKE_CURRENT_LIST_DIR}/lib/_deps)
set(FETCHCONTENT_QUIET OFF)
set(FETCHCONTENT_UPDATES_DISCONNECTED ON)
set(BUILD_SHARED_LIBS OFF)
set(BUILD_TESTING OFF)

# Set BASEPATH variable
set(BASEPATH "${CMAKE_CURRENT_LIST_DIR}")

# Make available bayesmix-dev/math (and TBB)
include(cmake/math.cmake)

# Check math has been populated
if(math_POPULATED)
  message(STATUS "math source dir: ${math_SOURCE_DIR}")
  message(STATUS "math binary dir: ${math_BINARY_DIR}")
else()
  message(FATAL_ERROR "math library required but not found!")
endif()

# Make available protocolbuffers/protobuf (v3.16.0)
include(cmake/protobuf.cmake)

# Check if protobuf is found
if(Protobuf_FOUND)
  message(STATUS "Protobuf version : ${Protobuf_VERSION}")
  message(STATUS "Protobuf include path : ${Protobuf_INCLUDE_DIRS}")
  message(STATUS "Protobuf libraries : ${Protobuf_LIBRARIES}")
  message(STATUS "Protobuf compiler libraries : ${Protobuf_PROTOC_LIBRARIES}")
  message(STATUS "Protobuf lite libraries : ${Protobuf_LITE_LIBRARIES}")
  message(STATUS "Protobuf protoc : ${Protobuf_PROTOC_EXECUTABLE}")
else()
  message(FATAL_ERROR "Protobuf library required but not found!")
endif()

# Define project options
option(DISABLE_TESTS
      "If tests should be compiled or no" OFF)
option(ENABLE_BENCHMARKS
      "If benchmarks should be compiled or no" OFF)
option(ENABLE_DOCS
      "If docs should be generated or no" OFF)
option(BUILD_RUN "" ON)

# Include pathstests
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)
set(INCLUDE_PATHS
  ${BASEPATH}
  ${math_SOURCE_DIR}
  ${math_SOURCE_DIR}/lib/boost_1.78.0
  ${math_SOURCE_DIR}/lib/eigen_3.4.0
  ${math_SOURCE_DIR}/lib/sundials_6.1.1/include
  ${math_SOURCE_DIR}/lib/tbb_2020.3/include
  ${CMAKE_CURRENT_BINARY_DIR}
  ${Protobuf_INCLUDE_DIRS}
)

# Link paths
set(LINK_LIBRARIES
  pthread
  protobuf::libprotobuf
  tbb
  OpenMP::OpenMP_CXX
)

# Compiler options
set(COMPILE_OPTIONS -D_REENTRANT -fPIC)

# Compile proto files
file(GLOB ProtoFiles "${BASEPATH}/src/proto/*.proto")
set(PROTO_DIR proto)
foreach(PROTO_FILE IN LISTS ProtoFiles)
  message(STATUS "protoc proto(cc): ${PROTO_FILE}")
  get_filename_component(PROTO_DIR ${PROTO_FILE} DIRECTORY)
  get_filename_component(PROTO_NAME ${PROTO_FILE} NAME_WE)
  set(PROTO_HDR ${CMAKE_CURRENT_BINARY_DIR}/${PROTO_NAME}.pb.h)
  set(PROTO_SRC ${CMAKE_CURRENT_BINARY_DIR}/${PROTO_NAME}.pb.cc)
  message(STATUS "protoc hdr: ${PROTO_HDR}")
  message(STATUS "protoc src: ${PROTO_SRC}")
  add_custom_command(
    OUTPUT ${PROTO_SRC} ${PROTO_HDR}
    COMMAND protobuf::protoc
    "--proto_path=${BASEPATH}/src/proto" ${PROTO_DIRS}
    "--cpp_out=${PROJECT_BINARY_DIR}"
    "--python_out=${BASEPATH}/python/bayesmixpy/proto"
    ${PROTO_FILE}
    DEPENDS ${PROTO_FILE} protobuf::protoc
    COMMENT "Generate C++ protocol buffer for ${PROTO_FILE}"
    VERBATIM)
  list(APPEND PROTO_HDRS ${PROTO_HDR})
  list(APPEND PROTO_SRCS ${PROTO_SRC})
endforeach()
SET_SOURCE_FILES_PROPERTIES(${PROTO_SRCS} ${PROTO_HDRS} PROPERTIES GENERATED TRUE)

# Export variables to parent scope if bayesmix is used as dependency library
get_directory_property(HAS_PARENT PARENT_DIRECTORY)
if(HAS_PARENT)
  set(BAYESMIX_TBB_ROOT ${TBB_ROOT} PARENT_SCOPE)
  set(BAYESMIX_INCLUDE_PATHS ${INCLUDE_PATHS} PARENT_SCOPE)
  set(BAYESMIX_LINK_LIBRARIES ${LINK_LIBRARIES} PARENT_SCOPE)
  set(BAYESMIX_COMPILE_OPTIONS ${COMPILE_OPTIONS} PARENT_SCOPE)
  set(BAYESMIX_PROTO_PATH "${BASEPATH}/src/proto" PARENT_SCOPE)
  set(PROTO_HEADERS ${PROTO_HDRS} PARENT_SCOPE)
  set(PROTO_SOURCES ${PROTO_SRCS} PARENT_SCOPE)
  set(ProtoFiles ${ProtoFiles} PARENT_SCOPE)
endif()

# Build library object
add_library(bayesmix OBJECT)
target_sources(bayesmix PUBLIC ${PROTO_SRCS} ${PROTO_HDRS})
add_subdirectory(src)
target_include_directories(bayesmix PUBLIC ${INCLUDE_PATHS})
target_link_libraries(bayesmix PUBLIC ${LINK_LIBRARIES})
target_compile_options(bayesmix PUBLIC ${COMPILE_OPTIONS})

# Build static library
add_library(bayesmixlib $<TARGET_OBJECTS:bayesmix>)

# Build run executable
if (BUILD_RUN)
  add_executable(run_mcmc $<TARGET_OBJECTS:bayesmix> executables/run_mcmc.cc)
  target_include_directories(run_mcmc PUBLIC ${INCLUDE_PATHS})
  target_link_libraries(run_mcmc PUBLIC ${LINK_LIBRARIES})
  target_compile_options(run_mcmc PUBLIC ${COMPILE_OPTIONS})
endif()

# Add tests
if (NOT DISABLE_TESTS)
  add_subdirectory(test)
endif()

# Add benchmarks
if (ENABLE_BENCHMARKS)
  add_subdirectory(benchmarks)
endif()

# Add docs
if (ENABLE_DOCS)
  add_subdirectory(docs)
endif()

if (NOT DISABLE_PLOTS)
  # Make available matplotplusplus
  include(cmake/matplotplusplus.cmake)
  # Check matplotplusplus has been populated
  if(matplotplusplus_POPULATED)
    message(STATUS "matplotplusplus source dir: ${matplotplusplus_SOURCE_DIR}")
    message(STATUS "matplotplusplus binary dir: ${matplotplusplus_BINARY_DIR}")
  else()
    message(FATAL_ERROR "matplotplusplus library required but not found!")
  endif()
  # Build plot_mcmc executable
  add_executable(plot_mcmc
    $<TARGET_OBJECTS:bayesmix>
    executables/plot_mcmc.cc
    src/plots/plot_utils.h
    src/plots/plot_utils.cc
  )
  target_include_directories(plot_mcmc PUBLIC ${INCLUDE_PATHS} ${matplotplusplus_SOURCE_DIR}/source)
  target_link_libraries(plot_mcmc PUBLIC ${LINK_LIBRARIES} matplot)
  target_compile_options(plot_mcmc PUBLIC ${COMPILE_OPTIONS})
endif()

# Add examples
if (NOT DISABLE_EXAMPLES)
  add_subdirectory(examples)
endif()
