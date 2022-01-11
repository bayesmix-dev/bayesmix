target_sources(bayesmix
  PUBLIC
    abstract_hierarchy.h
    base_hierarchy.h
    conjugate_hierarchy.h
    lin_reg_uni_hierarchy.h
    lin_reg_uni_hierarchy.cc
    nnig_hierarchy.h
    nnig_hierarchy.cc
    nnw_hierarchy.h
    nnw_hierarchy.cc
    nnxig_hierarchy.h
    nnxig_hierarchy.cc
)

add_subdirectory(likelihoods)
