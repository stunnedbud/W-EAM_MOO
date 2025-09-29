#!/bin/bash

#SBATCH --job-name=smacglobal
#SBATCH --mem-per-cpu=16G

python SMAC_optimize_global_noprior.py
