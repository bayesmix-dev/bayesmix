target_sources(bayesmix
  PUBLIC
    includes.hpp
    utils.hpp
)

#add_subdirectory(algorithms)
add_subdirectory(collectors)
add_subdirectory(hierarchies)
add_subdirectory(mixings)
add_subdirectory(runtime)
add_subdirectory(utils)
