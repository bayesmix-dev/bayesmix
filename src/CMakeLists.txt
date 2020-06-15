target_sources(bayesmix
  PUBLIC
    includes.hpp
    utils.hpp
)


add_subdirectory(algorithms)
add_subdirectory(collectors)
add_subdirectory(hierarchies)
add_subdirectory(hyperparameters)
add_subdirectory(mixtures)
add_subdirectory(runtime)



