#!/bin/bash
python DIMEX/eam.py -n 0 --tolerance=0 --sigma=0.1 --iota=0.3 --kappa=1.5
python DIMEX/eam.py -n 1 --tolerance=0 --sigma=0.1 --iota=0.3 --kappa=1.5
python DIMEX/eam.py -n 2 --tolerance=0 --sigma=0.1 --iota=0.3 --kappa=1.5
python DIMEX/eam.py -n 3 --tolerance=0 --sigma=0.1 --iota=0.3 --kappa=1.5
python DIMEX/eam.py -n 4 --tolerance=0 --sigma=0.1 --iota=0.3 --kappa=1.5
python DIMEX/eam.py -n 5 --tolerance=0 --sigma=0.1 --iota=0.3 --kappa=1.5
python DIMEX/eam.py -f

