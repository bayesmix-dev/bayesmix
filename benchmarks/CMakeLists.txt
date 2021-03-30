cmake_minimum_required(VERSION 3.13.0)
project(benchmark_bayesmix)
enable_testing()


find_package(benchmark REQUIRED)

add_executable(benchmark_bayesmix $<TARGET_OBJECTS:bayesmix>
  nnw_marg_lpdf.cc
  mcmc_runs.cc
  eval_lpdf.cc
  lpd_grid.cc
  utils.h
  utils.cc
  main.cpp
)

target_include_directories(benchmark_bayesmix PUBLIC ${INCLUDE_PATHS})
target_link_libraries(benchmark_bayesmix PUBLIC
  ${LINK_LIBRARIES}  benchmark::benchmark)
target_compile_options(benchmark_bayesmix PUBLIC ${COMPILE_OPTIONS})
