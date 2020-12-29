add_subdirectory(clustering)
add_subdirectory(lossfunction)

add_executable(point_estimate $<TARGET_OBJECTS:bayesmix> point_estimate_test.cpp)
target_include_directories(point_estimate PUBLIC ${INCLUDE_PATHS})
target_link_libraries(point_estimate PUBLIC ${LINK_LIBRARIES})
target_compile_options(point_estimate PUBLIC ${COMPILE_OPTIONS})

