add_subdirectory(lossfunction)
add_subdirectory(uncertainty)

target_sources(bayesmix
        PUBLIC
        ClusterEstimator.cpp
        ClusterEstimator.hpp
        )

#target_include_directories(point_estimate PUBLIC ${INCLUDE_PATHS})
#target_link_libraries(point_estimate PUBLIC ${LINK_LIBRARIES})
#target_compile_options(point_estimate PUBLIC ${COMPILE_OPTIONS})

