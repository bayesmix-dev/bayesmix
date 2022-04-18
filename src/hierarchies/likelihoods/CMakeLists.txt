target_sources(bayesmix PUBLIC
    likelihood_internal.h
    abstract_likelihood.h
    base_likelihood.h
    uni_norm_likelihood.h
    uni_norm_likelihood.cc
    multi_norm_likelihood.h
    multi_norm_likelihood.cc
    # uni_lin_reg_likelihood.h
    # uni_lin_reg_likelihood.cc
    # laplace_likelihood.h
    # laplace_likelihood.cc
    # fa_likelihood.h
    # fa_likelihood.cc
)

add_subdirectory(states)
