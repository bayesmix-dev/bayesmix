cmake_minimum_required(VERSION 3.14.0)

include(FetchContent)

set(FETCHCONTENT_QUIET OFF)
set(FETCHCONTENT_UPDATES_DISCONNECTED ON)
set(BUILD_SHARED_LIBS OFF)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(BUILD_TESTING OFF)

message(CHECK_START "Fetching Matplotplusplus")
list(APPEND CMAKE_MESSAGE_INDENT "  ")

FetchContent_Declare(matplotplusplus
  GIT_REPOSITORY https://github.com/alandefreitas/matplotplusplus
  GIT_TAG origin/master # or whatever tag you want
)
# FetchContent_MakeAvailable(protobuf)

FetchContent_GetProperties(matplotplusplus)
if(NOT matplotplusplus_POPULATED)
  FetchContent_Populate(matplotplusplus)
  add_subdirectory(${matplotplusplus_SOURCE_DIR} ${matplotplusplus_BINARY_DIR} EXCLUDE_FROM_ALL)
endif()

add_executable(plot_mcmc plot_mcmc.cc $<TARGET_OBJECTS:bayesmix>)

target_include_directories(plot_mcmc PUBLIC ${INCLUDE_PATHS})
target_link_libraries(plot_mcmc PUBLIC ${LINK_LIBRARIES} matplot)
target_compile_options(plot_mcmc PUBLIC ${COMPILE_OPTIONS})
