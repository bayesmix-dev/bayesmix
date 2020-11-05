cmake_minimum_required(VERSION 3.13.0)
project(bayesmix)

find_package(PkgConfig REQUIRED)

set(CMAKE_BUILD_WITH_INSTALL_RPATH ON)
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)
SET(CMAKE_EXE_LINKER_FLAGS "-Wl,-rpath,/home/mario/dev/stan/math/lib/tbb")

SET(BASEPATH "${CMAKE_SOURCE_DIR}")
INCLUDE_DIRECTORIES("${BASEPATH}")

# add_executable(bayesmix run.cpp)
# add_executable(bayesmix maintest_uni.cpp)

add_library(bayesmix)

add_subdirectory(src)
add_subdirectory(proto)
#add_subdirectory(test)

message("source dir: " "${CMAKE_CURRENT_SOURCE_DIR}")

target_include_directories(bayesmix PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/lib/math
    ${CMAKE_CURRENT_SOURCE_DIR}/lib/math/stan/math
    ${CMAKE_CURRENT_SOURCE_DIR}/lib/math/lib/boost_1.72.0
    ${CMAKE_CURRENT_SOURCE_DIR}/lib/math/lib/eigen_3.3.7
    ${CMAKE_CURRENT_SOURCE_DIR}/lib/math/lib/tbb_2019_U8/include
    ${CMAKE_CURRENT_SOURCE_DIR}/lib/protobuf/src
)

target_link_directories(bayesmix PUBLIC
    /home/mario/dev/stan/math/lib/tbb
    ${CMAKE_CURRENT_SOURCE_DIR}/lib/protobuf/src/.libs
)

target_link_libraries(bayesmix
    tbb
    pthread
    protobuf
)
target_compile_options(bayesmix PUBLIC -D_REENTRANT -fPIC)


add_executable(run run.cpp)
target_link_libraries(run bayesmix)
target_include_directories(run PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

target_compile_options(run PUBLIC -D_REENTRANT -fPIC)
