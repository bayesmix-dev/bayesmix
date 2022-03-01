target_sources(bayesmix PUBLIC
    abstract_likelihood.h
    base_likelihood.h
    uni_norm_likelihood.h
    uni_norm_likelihood.cc
    multi_norm_likelihood.h
    multi_norm_likelihood.cc
    laplace_likelihood.h
    laplace_likelihood.cc
)

add_subdirectory(states)
