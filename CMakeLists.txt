cmake_minimum_required(VERSION 3.14.0)
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_LIST_DIR}")
message("CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH}")

find_program(CCACHE_PROGRAM ccache)
if (CCACHE_PROGRAM)
	set_property(GLOBAL PROPERTY RULE_LAUNCH_COMPILE "${CCACHE_PROGRAM}")
endif()

project(bayesmix)


set (CMAKE_CXX_STANDARD 14)
set (CMAKE_BUILD_TYPE Release)
set(CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake" ${CMAKE_MODULE_PATH})
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -funroll-loops -ftree-vectorize")
set(CMAKE_FIND_PACKAGE_PREFER_CONFIG TRUE)

include(FindPythonInterp)
add_custom_target(
 build_tbb ALL
 COMMAND ${PYTHON_EXECUTABLE} ${CMAKE_CURRENT_LIST_DIR}/build_tbb.py
)

find_package(PkgConfig REQUIRED)
find_package(Protobuf)

if (NOT Protobuf_FOUND AND NOT PROTOBUF_FOUND AND NOT TARGET protobuf::libprotobuf)
  include(FetchContent)
  set(FETCHCONTENT_QUIET OFF)
  set(FETCHCONTENT_UPDATES_DISCONNECTED ON)
  set(BUILD_SHARED_LIBS OFF)
  set(CMAKE_POSITION_INDEPENDENT_CODE ON)
  set(BUILD_TESTING OFF)

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
option(DISABLE_BENCHMARKS
      "If benchmarks should be compiled or no" OFF)
option(DISABLE_DOCS
      "If docs should be generated or no" OFF)
option(BUILD_RUN "" ON)

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
  ${protobuf_SOURCE_DIR}/src
)


set(LINK_LIBRARIES 
  pthread 
  protobuf::libprotobuf)
set(COMPILE_OPTIONS -D_REENTRANT -fPIC)

file(GLOB ProtoFiles "${BASEPATH}/proto/*.proto")
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
    COMMAND protobuf::protoc "--proto_path=${BASEPATH}/proto"
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
  set(PROTO_HEADERS ${PROTO_HDRS} PARENT_SCOPE)
  set(PROTO_SOURCES ${PROTO_SRCS} PARENT_SCOPE)
  set(ProtoFiles ${ProtoFiles} PARENT_SCOPE)
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


if (BUILD_RUN)
  # Build run executable
  add_executable(run $<TARGET_OBJECTS:bayesmix> run.cc)
  target_include_directories(run PUBLIC ${INCLUDE_PATHS})
  target_link_libraries(run PUBLIC ${LINK_LIBRARIES})
  target_compile_options(run PUBLIC ${COMPILE_OPTIONS})
endif()

if (NOT DISABLE_TESTS)
  add_subdirectory(test)
endif()

if (NOT DISABLE_BENCHMARKS)
  add_subdirectory(benchmarks)
endif()

if (NOT DISABLE_DOCS)
  add_subdirectory(docs)
endif()

add_subdirectory(examples)
