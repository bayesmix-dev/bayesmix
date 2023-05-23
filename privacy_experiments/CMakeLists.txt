cmake_minimum_required(VERSION 3.13.0)
project(privacy_experiments)


# Build run executable
add_executable(laplace1d $<TARGET_OBJECTS:bayesmix>
    unidimensional_laplace/main.cc
    utils.h
    utils.cc
)

add_executable(wavelet $<TARGET_OBJECTS:bayesmix>
    wavelets/main.cc
    utils.h
    utils.cc
)

target_include_directories(laplace1d PUBLIC ${INCLUDE_PATHS})
target_link_libraries(laplace1d PUBLIC ${LINK_LIBRARIES})
target_compile_options(laplace1d PUBLIC ${COMPILE_OPTIONS})

target_include_directories(wavelet PUBLIC ${INCLUDE_PATHS})
target_link_libraries(wavelet PUBLIC ${LINK_LIBRARIES})
target_compile_options(wavelet PUBLIC ${COMPILE_OPTIONS})
