#!/bin/bash
#PBS -N create_input
#PBS -l ncpus=1
#PBS -l walltime=12:00:00
#PBS -l mem=20gb
#PBS -l storage=gdata/hh5+gdata/rt52
#PBS -o aus2200.out
#PBS -e aus2200.err
#PBS -A w40

module use /g/data/hh5/public/modules
module load conda/analysis3

/g/data/hh5/public/apps/miniconda3/envs/analysis3-22.10/bin/python3.9 /home/565/fl2086/dales_openBC_setup/scripts/create_input.py /home/565/fl2086/dales_openBC_setup/scripts/input_coarse_anika.json
