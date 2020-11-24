cmake_minimum_required(VERSION 3.13.0)
project(bayesmix)

find_package(PkgConfig REQUIRED)

set(CMAKE_BUILD_WITH_INSTALL_RPATH ON)
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)
set(BASEPATH "${CMAKE_SOURCE_DIR}")
set(INCLUDEPATHS
    ${CMAKE_CURRENT_SOURCE_DIR}/lib/math
    ${CMAKE_CURRENT_SOURCE_DIR}/lib/math/lib/boost_1.72.0
    ${CMAKE_CURRENT_SOURCE_DIR}/lib/math/lib/eigen_3.3.7
    ${CMAKE_CURRENT_SOURCE_DIR}/lib/math/lib/tbb_2019_U8/include
    ${CMAKE_CURRENT_SOURCE_DIR}/lib/protobuf/src
)

add_library(bayesmix OBJECT)
# Add sources for bayesmix
add_subdirectory(src)
add_subdirectory(proto)
add_subdirectory(test)

target_include_directories(bayesmix PUBLIC ${INCLUDEPATHS})
target_link_libraries(bayesmix tbb pthread protobuf)
target_compile_options(bayesmix PUBLIC -D_REENTRANT -fPIC)

add_executable(run $<TARGET_OBJECTS:bayesmix> run.cpp)
target_include_directories(run PUBLIC ${INCLUDEPATHS})
target_link_libraries(run tbb pthread protobuf)
target_compile_options(run PUBLIC -D_REENTRANT -fPIC)
