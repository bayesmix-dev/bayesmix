cmake_minimum_required(VERSION 3.13.0)
project(bayesmix)

find_package(PkgConfig REQUIRED)
pkg_check_modules(PROTO REQUIRED protobuf)

set(CMAKE_BUILD_WITH_INSTALL_RPATH ON)
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)
# SET(CMAKE_EXE_LINKER_FLAGS "-Wl,-rpath,/home/mario/dev/stan/math/lib/tbb")

SET(BASEPATH "${CMAKE_SOURCE_DIR}")
INCLUDE_DIRECTORIES("${BASEPATH}")

add_executable(bayesmix maintest_uni.cpp)
add_subdirectory(src)

target_include_directories(bayesmix PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/lib/stan/math
    ${CMAKE_CURRENT_SOURCE_DIR}/lib/stan/math/lib/boost_1.72.0
    ${CMAKE_CURRENT_SOURCE_DIR}/lib/stan/math/lib/eigen_3.3.7
    ${CMAKE_CURRENT_SOURCE_DIR}/lib/stan/math/lib/tbb_2019_U8/include)

target_link_directories(bayesmix PRIVATE
    /home/mario/dev/stan/math/lib/tbb
    ${CMAKE_CURRENT_SOURCE_DIR}/lib/protobuf-3.12.3/src/.libs)

target_link_libraries(bayesmix 
    tbb 
    pthread
	protobuf)

target_compile_options(bayesmix PUBLIC -D_REENTRANT -fPIC)
