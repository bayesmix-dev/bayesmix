target_sources(bayesmix PUBLIC includes.h)

    add_subdirectory(algorithms) add_subdirectory(collectors)
        add_subdirectory(hierarchies) add_subdirectory(mixings)
            add_subdirectory(runtime) add_subdirectory(utils)
