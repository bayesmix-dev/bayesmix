include(GNUInstallDirs)

# Set protobuf options
set(Protobuf_USE_STATIC_LIBS ON)
set(Protobuf_MSVC_STATIC_RUNTIME OFF)
set(protobuf_BUILD_TESTS OFF)
set(protobuf_BUILD_PROTOC_BINARIES ON)

# Fetch protocolbuffers_protobuf
message(STATUS "")
message(STATUS "Fetching protocolbuffers/protobuf")
FetchContent_Declare(protobuf
  URL https://github.com/protocolbuffers/protobuf/archive/refs/tags/v3.16.0.tar.gz
  DOWNLOAD_EXTRACT_TIMESTAMP TRUE
)
FetchContent_MakeAvailable(protobuf)

# Set variables
set(Protobuf_ROOT ${protobuf_SOURCE_DIR}/cmake)
set(Protobuf_DIR ${Protobuf_ROOT}/${CMAKE_INSTALL_LIBDIR}/cmake/protobuf)

# Configure protobuf
message(STATUS "Setting up protobuf ...")
message("CMAKE_COMMAND: ${CMAKE_COMMAND}")
execute_process(
  COMMAND ${CMAKE_COMMAND} -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE} -Dprotobuf_BUILD_TESTS=OFF -Dprotobuf_BUILD_PROTOC_BINARIES=ON -DCMAKE_POSITION_INDEPENDENT_CODE=ON -G "${CMAKE_GENERATOR}" .
  RESULT_VARIABLE result
  WORKING_DIRECTORY ${Protobuf_ROOT}
)
if(result)
  message(FATAL_ERROR "Failed to download protobuf (${result})!")
endif()

# Build protobuf
message(STATUS "Building protobuf ...")
execute_process(
  COMMAND ${CMAKE_COMMAND} --build .
  RESULT_VARIABLE result
  WORKING_DIRECTORY ${Protobuf_ROOT}
)
if(result)
  message(FATAL_ERROR "Failed to build protobuf (${result})!")
endif()

# Find package in installed folder
find_package(Protobuf REQUIRED HINTS ${Protobuf_DIR})

# Include protobuf related informations
include(${Protobuf_DIR}/protobuf-config.cmake)
include(${Protobuf_DIR}/protobuf-module.cmake)
include(${Protobuf_DIR}/protobuf-options.cmake)
include(${Protobuf_DIR}/protobuf-targets.cmake)
