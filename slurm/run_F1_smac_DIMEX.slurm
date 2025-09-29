#!/bin/bash
#SBATCH --job-name=F1SMACdmx
#SBATCH --mem-per-cpu=8G
python eam_run_smac_minimizingF1.py 0.5 0.5
