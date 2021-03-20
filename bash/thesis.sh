#!/usr/bin/env bash

# Define paths
ASCIIPB='resources/asciipb/thesis'
CSVIN='resources/csv/in/thesis'
CSVOUT='resources/csv/out/thesis'
RECORDIO='resources/recordio'

data=$1
# Case 1: galaxy dataset
if [ "$data" == 'galaxy' ]; then
  declare -a algos=('Neal2' 'Neal3' 'Neal8')
  declare -a mixs=('DP' 'PY')
  hier='NNIG'
  covsname=''
# Case 2: faithful dataset
elif [ "$data" == 'faithful' ]; then
  declare -a algos=('Neal2' 'Neal3' 'Neal8')
  declare -a mixs=('DP' 'PY')
  hier='NNW'
  covsname=''
# Case 3: dde dataset
elif [ "$data" == 'dde' ]; then
  declare -a algos=('BlockedGibbs')
  declare -a mixs=('LogSB' 'TruncSB')
  hier='NNIG'
  covsname='dde_covs'
else
  echo 'Use arg galaxy, faithful, or dde'
  exit
fi

# Run all combinations
for algo in "${algos[@]}"; do
  for mix in "${mixs[@]}"; do
    #echo \
    build/run \
      $ASCIIPB/$algo.asciipb \
      $hier $ASCIIPB/hier_$data.asciipb \
      $mix  $ASCIIPB/$mix.asciipb \
      $RECORDIO/${data}_${algo}_${mix}.recordio \
      $CSVIN/$data.csv \
      $CSVIN/${data}_grid.csv \
      $CSVOUT/${data}_dens_${algo}_$mix.csv \
      $CSVOUT/${data}_nclu_${algo}_$mix.csv \
      $CSVOUT/${data}_clus_${algo}_$mix.csv \
      "" \
      "" \
      $CSVIN/${covsname}.csv \
      $CSVIN/${covsname}_grid.csv
    echo 
  done
done
