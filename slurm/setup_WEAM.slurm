#!/bin/bash
#SBATCH --job-name=WEAM_50E
#SBATCH --mem-per-cpu=8G
python eam.py -n 0 
python eam.py -f 0
python eam.py -a 0
python eam.py -c 0
python eam.py -e 0
