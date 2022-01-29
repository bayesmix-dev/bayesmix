#!/usr/bin/env bash

# ./build/run_mcmc_mfa <n_threads> <config file name>
#
# set n_threads to -1 let openMP decide the number of threads
# you can specify more than one file name in order to work in parallel with different data
#
# the config file should be a txt file like this:
#
# --algo-params-file examples/mfa_hierarchy/in/algo.asciipb 
# --hier-type MFA --hier-args examples/mfa_hierarchy/in/mfa_auto.asciipb 
# --mix-type DP --mix-args examples/mfa_hierarchy/in/dp_gamma.asciipb 
# --coll-name examples/mfa_hierarchy/out/chains_auto.recordio 
# --data-file examples/mfa_hierarchy/in/data.csv 
# --grid-file examples/mfa_hierarchy/in/data.csv 
# --dens-file examples/mfa_hierarchy/out/density_file_auto.csv 
# --n-cl-file examples/mfa_hierarchy/out/numclust_auto.csv 
# --clus-file examples/mfa_hierarchy/out/clustering_auto.csv 
# --best-clus-file resources/tutorial/out/best_clustering.csv
# --<argument> <value to assign to that specific argument>

./build/run_mcmc_mfa -1 examples/mfa_hierarchy/config.txt
