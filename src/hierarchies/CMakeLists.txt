target_sources(bayesmix
  PUBLIC
    abstract_hierarchy.h
    base_hierarchy.h
    nnig_hierarchy.h
    nnxig_hierarchy.h
    nnw_hierarchy.h
    lin_reg_uni_hierarchy.h
    fa_hierarchy.h
    lapnig_hierarchy.h
)

add_subdirectory(likelihoods)
add_subdirectory(priors)
add_subdirectory(updaters)
