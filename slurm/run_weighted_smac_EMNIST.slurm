#!/bin/bash
#SBATCH --job-name=wsmacEMNIST
#SBATCH --mem-per-cpu=16G
#SBATCH --ntasks=21

srun --exclusive --ntasks=1 python weightedSMAC.py 0 1 &
srun --exclusive --ntasks=1 python weightedSMAC.py 0.05 0.95 &
srun --exclusive --ntasks=1 python weightedSMAC.py 0.1 0.9 &
srun --exclusive --ntasks=1 python weightedSMAC.py 0.15 0.85 &
srun --exclusive --ntasks=1 python weightedSMAC.py 0.2 0.8 &
srun --exclusive --ntasks=1 python weightedSMAC.py 0.25 0.75 &
srun --exclusive --ntasks=1 python weightedSMAC.py 0.3 0.7 &
srun --exclusive --ntasks=1 python weightedSMAC.py 0.35 0.65 &
srun --exclusive --ntasks=1 python weightedSMAC.py 0.4 0.6 &
srun --exclusive --ntasks=1 python weightedSMAC.py 0.45 0.55 &
srun --exclusive --ntasks=1 python weightedSMAC.py 0.5 0.5 &
srun --exclusive --ntasks=1 python weightedSMAC.py 0.55 0.45 &
srun --exclusive --ntasks=1 python weightedSMAC.py 0.6 0.4 &
srun --exclusive --ntasks=1 python weightedSMAC.py 0.65 0.35 &
srun --exclusive --ntasks=1 python weightedSMAC.py 0.7 0.3 &
srun --exclusive --ntasks=1 python weightedSMAC.py 0.75 0.25 &
srun --exclusive --ntasks=1 python weightedSMAC.py 0.8 0.2 &
srun --exclusive --ntasks=1 python weightedSMAC.py 0.85 0.15 &
srun --exclusive --ntasks=1 python weightedSMAC.py 0.9 0.1 &
srun --exclusive --ntasks=1 python weightedSMAC.py 0.95 0.05 &
srun --exclusive --ntasks=1 python weightedSMAC.py 1 0 &
wait
