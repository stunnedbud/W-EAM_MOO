#!/bin/bash

#SBATCH --job-name=smacpermem
#SBATCH --mem-per-cpu=16G

python SMAC_optimize_per_memory_BBFacade.py
