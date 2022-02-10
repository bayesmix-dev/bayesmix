#!/usr/bin/env bash

  build/run_mcmc \
  --algo-params-file examples/fa_hierarchy/in/algo.asciipb \
  --hier-type FA --hier-args examples/fa_hierarchy/in/fa_auto.asciipb \
  --mix-type DP --mix-args examples/fa_hierarchy/in/dp_gamma.asciipb \
  --coll-name examples/fa_hierarchy/out/chains.recordio \
  --data-file examples/fa_hierarchy/in/data.csv \
  --grid-file examples/fa_hierarchy/in/data.csv \
  --dens-file examples/fa_hierarchy/out/density_file.csv \
  --n-cl-file examples/fa_hierarchy/out/numclust.csv \
  --clus-file examples/fa_hierarchy/out/clustering.csv \
  --best-clus-file examples/fa_hierarchy/out/best_clustering.csv
