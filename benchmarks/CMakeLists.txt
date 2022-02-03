cmake_minimum_required(VERSION 3.13.0)
project(benchmark_bayesmix)
enable_testing()

add_subdirectory(${CMAKE_CURRENT_LIST_DIR}/../lib/math/lib/benchmark_1.5.1 build)

add_executable(benchmark_bayesmix $<TARGET_OBJECTS:bayesmix>
  # nnw_marg_lpdf.cc
  # mcmc_runs.cc
  eval_lpdf.cc
  # lpd_grid.cc
  main.cpp
)

target_include_directories(benchmark_bayesmix PUBLIC ${INCLUDE_PATHS})
target_link_libraries(benchmark_bayesmix PUBLIC
  ${LINK_LIBRARIES}  benchmark::benchmark benchmark::benchmark_main)
target_compile_options(benchmark_bayesmix PUBLIC ${COMPILE_OPTIONS})
