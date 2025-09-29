#!/bin/bash
#SBATCH --job-name=WEAM_extend_corpus
#SBATCH --mem-per-cpu=8G
python eam.py -n 0 --tolerance=15 --iota=0.409042472 --kappa=1.950540673
python eam.py -f 0 --tolerance=15 --iota=0.409042472 --kappa=1.950540673
python eam.py -a 0 --tolerance=15 --iota=0.409042472 --kappa=1.950540673
python eam.py -c 0 --tolerance=15 --iota=0.409042472 --kappa=1.950540673
python eam.py -e 0 --tolerance=15 --iota=0.409042472 --kappa=1.950540673
python eam.py -n 1 --tolerance=15 --iota=0.409042472 --kappa=1.950540673
python eam.py -f 1 --tolerance=15 --iota=0.409042472 --kappa=1.950540673
python eam.py -a 1 --tolerance=15 --iota=0.409042472 --kappa=1.950540673
python eam.py -c 1 --tolerance=15 --iota=0.409042472 --kappa=1.950540673
python eam.py -e 1 --tolerance=15 --iota=0.409042472 --kappa=1.950540673
python eam.py -n 2 --tolerance=15 --iota=0.409042472 --kappa=1.950540673
python eam.py -f 2 --tolerance=15 --iota=0.409042472 --kappa=1.950540673
python eam.py -a 2 --tolerance=15 --iota=0.409042472 --kappa=1.950540673
python eam.py -c 2 --tolerance=15 --iota=0.409042472 --kappa=1.950540673
python eam.py -e 2 --tolerance=15 --iota=0.409042472 --kappa=1.950540673
python eam.py -n 3 --tolerance=15 --iota=0.409042472 --kappa=1.950540673
python eam.py -f 3 --tolerance=15 --iota=0.409042472 --kappa=1.950540673
python eam.py -a 3 --tolerance=15 --iota=0.409042472 --kappa=1.950540673
python eam.py -c 3 --tolerance=15 --iota=0.409042472 --kappa=1.950540673
python eam.py -e 3 --tolerance=15 --iota=0.409042472 --kappa=1.950540673
python eam.py -n 4 --tolerance=15 --iota=0.409042472 --kappa=1.950540673
python eam.py -f 4 --tolerance=15 --iota=0.409042472 --kappa=1.950540673
python eam.py -a 4 --tolerance=15 --iota=0.409042472 --kappa=1.950540673
python eam.py -c 4 --tolerance=15 --iota=0.409042472 --kappa=1.950540673
python eam.py -e 4 --tolerance=15 --iota=0.409042472 --kappa=1.950540673
python eam.py -n 5 --tolerance=15 --iota=0.409042472 --kappa=1.950540673
python eam.py -f 5 --tolerance=15 --iota=0.409042472 --kappa=1.950540673
python eam.py -a 5 --tolerance=15 --iota=0.409042472 --kappa=1.950540673
python eam.py -c 5 --tolerance=15 --iota=0.409042472 --kappa=1.950540673
python eam.py -e 5 --tolerance=15 --iota=0.409042472 --kappa=1.950540673
