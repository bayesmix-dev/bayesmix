cmake_minimum_required(VERSION 3.14.0)
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_LIST_DIR}")
message("CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH}")

find_program(CCACHE_PROGRAM ccache)
if (CCACHE_PROGRAM)
	set_property(GLOBAL PROPERTY RULE_LAUNCH_COMPILE "${CCACHE_PROGRAM}")
endif()

project(bayesmix)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_BUILD_TYPE Release)

set(CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake" ${CMAKE_MODULE_PATH})
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -funroll-loops -fopenmp -ftree-vectorize -Wno-deprecated")
set(CMAKE_CXX_FLAGS_DEBUG "-Og")
set(CMAKE_FIND_PACKAGE_PREFER_CONFIG TRUE)
set(CMAKE_FIND_PACKAGE_TARGETS_GLOBAL TRUE)

# Require PkgConfig
find_package(PkgConfig REQUIRED)

# Require OpenMP
find_package(OpenMP REQUIRED)

# Include FetchContent
include(FetchContent)

# Set up FetchContent options
set(FETCHCONTENT_BASE_DIR ${CMAKE_CURRENT_LIST_DIR}/lib/_deps)
set(FETCHCONTENT_QUIET OFF)
set(FETCHCONTENT_UPDATES_DISCONNECTED ON)
set(BUILD_SHARED_LIBS OFF)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(BUILD_TESTING OFF)

# Make bayesmix-dev/math available
FetchContent_Declare(math
  GIT_REPOSITORY "https://github.com/bayesmix-dev/math.git"
  GIT_TAG "develop"
)
FetchContent_MakeAvailable(math)

# Clone bayesmix-dev/math into lib/math
# include(cmake/GitUtils.cmake)
# git_clone(
# 	PROJECT_NAME math
# 	GIT_URL https://github.com/bayesmix-dev/math.git
# 	GIT_BRANCH develop
# 	DIRECTORY ${CMAKE_CURRENT_LIST_DIR}/lib
# )

# TBB CMake integration
message(STATUS "Using math TBB")
# Define TBB_ROOT Folder
# set(TBB_ROOT ${CMAKE_CURRENT_LIST_DIR}/lib/math/lib/tbb)
# file(COPY ${CMAKE_CURRENT_LIST_DIR}/lib/math/lib/tbb_2020.3/ DESTINATION ${TBB_ROOT})
set(TBB_ROOT ${math_SOURCE_DIR}/lib/tbb)
file(COPY ${math_SOURCE_DIR}/lib/tbb_2020.3/ DESTINATION ${TBB_ROOT})
# Build TBB Library with CMake Integration
include(${TBB_ROOT}/cmake/TBBBuild.cmake)
list(APPEND MAKE_ARGS "tbb_build_dir=${TBB_ROOT}")
list(APPEND MAKE_ARGS "tbb_build_prefix=tbb")
tbb_build(TBB_ROOT ${TBB_ROOT} CONFIG_DIR TBB_DIR MAKE_ARGS ${MAKE_ARGS})
# Require TBB library (for compile / link options)
find_package(TBB)

# Check if Protobuf is present in system
find_package(Protobuf)
if (NOT Protobuf_FOUND AND NOT PROTOBUF_FOUND AND NOT TARGET protobuf::libprotobuf)
  if (${CMAKE_VERSION} VERSION_LESS "3.20.0")
    message(FATAL_ERROR
            "Your cmake version is too old: either install a newer version (>=3.20)"
            " or install google's protocol buffer (protobuf) library by hand.")
  endif()

  message(CHECK_START "Fetching Protobuf")
  list(APPEND CMAKE_MESSAGE_INDENT "  ")

  option(protobuf_BUILD_TESTS "" OFF)
  set(protobuf_BUILD_EXPORT OFF)
  set(protobuf_MSVC_STATIC_RUNTIME OFF)
  FetchContent_Declare(
      protobuf
      GIT_REPOSITORY "https://github.com/protocolbuffers/protobuf.git"
      GIT_TAG "v3.14.0"
      GIT_SUBMODULES ""
      SOURCE_SUBDIR cmake)
  FetchContent_MakeAvailable(protobuf)

  list(POP_BACK CMAKE_MESSAGE_INDENT)
  message(CHECK_PASS "fetched")
  message("Protobuf_FOUND ${Protobuf_FOUND}")
  message("   --> PROTOBUF LIB: ${PROTOBUF_LIBRARIES}")
  message("   --> PROTOBUF INCLUDE: ${PROTOBUF_INCLUDE_DIRS}")
  message("   --> PROTOBUF VERSION: ${Protobuf_VERSION}")
  message("   --> PROTOBUF Found: ${Protobuf_FOUND}")

endif()

option(DISABLE_TESTS
      "If tests should be compiled or no" OFF)
option(ENABLE_BENCHMARKS
      "If benchmarks should be compiled or no" OFF)
option(ENABLE_DOCS
      "If docs should be generated or no" OFF)
option(BUILD_RUN "" ON)

# set(CMAKE_BUILD_WITH_INSTALL_RPATH ON)
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)
set(BASEPATH "${CMAKE_CURRENT_LIST_DIR}")
set(INCLUDE_PATHS
  ${BASEPATH}
  ${math_SOURCE_DIR}
  ${math_SOURCE_DIR}/lib/boost_1.78.0
  ${math_SOURCE_DIR}/lib/eigen_3.4.0
  ${math_SOURCE_DIR}/lib/sundials_6.1.1/include
  ${TBB_ROOT}/include
  ${CMAKE_CURRENT_BINARY_DIR}
  ${protobuf_SOURCE_DIR}/src
)

set(LINK_LIBRARIES
	pthread
	protobuf::libprotobuf
	TBB::tbb
  OpenMP::OpenMP_CXX
)

set(COMPILE_OPTIONS -D_REENTRANT -fPIC)

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
    COMMAND protobuf::protoc "--proto_path=${BASEPATH}/src/proto"
    ${PROTO_DIRS} "--cpp_out=${PROJECT_BINARY_DIR}" ${PROTO_FILE}
    DEPENDS ${PROTO_FILE} protobuf::protoc
    COMMENT "Generate C++ protocol buffer for ${PROTO_FILE}"
    VERBATIM)
  list(APPEND PROTO_HDRS ${PROTO_HDR})
  list(APPEND PROTO_SRCS ${PROTO_SRC})
endforeach()

SET_SOURCE_FILES_PROPERTIES(${PROTO_SRCS} ${PROTO_HDRS} PROPERTIES GENERATED TRUE)

get_directory_property(HAS_PARENT PARENT_DIRECTORY)
if(HAS_PARENT)
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

if (BUILD_RUN)
  # Build run executable
  add_executable(run_mcmc $<TARGET_OBJECTS:bayesmix> executables/run_mcmc.cc)
  target_include_directories(run_mcmc PUBLIC ${INCLUDE_PATHS})
  target_link_libraries(run_mcmc PUBLIC ${LINK_LIBRARIES})
  target_compile_options(run_mcmc PUBLIC ${COMPILE_OPTIONS})
endif()

if (NOT DISABLE_TESTS)
  add_subdirectory(test)
endif()

if (ENABLE_BENCHMARKS)
  add_subdirectory(benchmarks)
endif()

if (ENABLE_DOCS)
  add_subdirectory(docs)
endif()

if (NOT DISABLE_PLOTS)
if (${CMAKE_VERSION} VERSION_LESS "3.20.0")
  message(FATAL_ERROR
          "Your cmake version is too old: version >= 3.20.0 is required for plotting.")
  endif()

  # include(FetchContent)

  # set(FETCHCONTENT_QUIET OFF)
  # set(FETCHCONTENT_UPDATES_DISCONNECTED ON)
  # set(BUILD_SHARED_LIBS OFF)
  # set(CMAKE_POSITION_INDEPENDENT_CODE ON)
  # set(BUILD_TESTING OFF)

  # Define patch command to inject
  set(matplotplusplus_patch git apply ${CMAKE_CURRENT_LIST_DIR}/resources/patches/matplotplusplus.patch)

  # Make matplotplusplus available
  message(CHECK_START "Fetching Matplotplusplus")
  list(APPEND CMAKE_MESSAGE_INDENT "  ")
  FetchContent_Declare(matplotplusplus
    GIT_REPOSITORY "https://github.com/alandefreitas/matplotplusplus"
    GIT_TAG "origin/master" # or whatever tag you want
    PATCH_COMMAND ${matplotplusplus_patch} 
  )
  FetchContent_MakeAvailable(matplotplusplus)

  # FetchContent_GetProperties(matplotplusplus)
  # if(NOT matplotplusplus_POPULATED)
  #   FetchContent_Populate(matplotplusplus)
  #   add_subdirectory(${matplotplusplus_SOURCE_DIR} ${matplotplusplus_BINARY_DIR}
  #                    EXCLUDE_FROM_ALL)
  # endif()
  # # find_package(Matplot++ REQUIRED)
  # message("matplot " ${matplotplusplus_SOURCE_DIR})

  add_executable(plot_mcmc $<TARGET_OBJECTS:bayesmix> executables/plot_mcmc.cc
    src/plots/plot_utils.h
    src/plots/plot_utils.cc)
    target_include_directories(plot_mcmc PUBLIC
      ${INCLUDE_PATHS} ${matplotplusplus_SOURCE_DIR}/source)
  target_link_libraries(plot_mcmc PUBLIC ${LINK_LIBRARIES} matplot)
  target_compile_options(plot_mcmc PUBLIC ${COMPILE_OPTIONS})
endif()

if (NOT DISABLE_EXAMPLES)
  add_subdirectory(examples)
endif()
