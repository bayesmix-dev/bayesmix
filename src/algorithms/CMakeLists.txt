target_sources(
    bayesmix PUBLIC base_algorithm.h base_algorithm.cc blocked_gibbs_algorithm
        .h blocked_gibbs_algorithm.cc conditional_algorithm
        .h conditional_algorithm.cc marginal_algorithm.h marginal_algorithm
        .cc neal2_algorithm.h neal2_algorithm.cc neal3_algorithm
        .h neal3_algorithm.cc neal8_algorithm.h neal8_algorithm
        .cc semihdp_sampler.h semihdp_sampler.cc)
