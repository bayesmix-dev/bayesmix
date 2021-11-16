cmake_minimum_required(VERSION 3.13.0)
project(examples_bayesmix)

add_executable(run_gamma $<TARGET_OBJECTS:bayesmix>  
    gamma_hierarchy/run_gamma_gamma.cc
    gamma_hierarchy/gamma_gamma_hier.h
)

target_include_directories(run_gamma PUBLIC ${INCLUDE_PATHS})
target_link_libraries(run_gamma PUBLIC
  ${LINK_LIBRARIES})
target_compile_options(run_gamma PUBLIC ${COMPILE_OPTIONS})
