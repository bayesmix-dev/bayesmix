cmake_minimum_required(VERSION 3.13.0)
project(examples_bayesmix)

add_executable(run_ex1 $<TARGET_OBJECTS:bayesmix>  
    run_gamma_gamma.cc
    gamma_gamma_hier.h
)

target_include_directories(run_ex1 PUBLIC ${INCLUDE_PATHS})
target_link_libraries(run_ex1 PUBLIC
  ${LINK_LIBRARIES})
target_compile_options(run_ex1 PUBLIC ${COMPILE_OPTIONS})
