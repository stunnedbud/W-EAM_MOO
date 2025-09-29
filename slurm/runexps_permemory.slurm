#!/bin/bash
########### Script de Ejecución (en Slurm) #########
#SBATCH --cpus-per-task=22
#SBATCH --job-name=smacpermem
#SBATCH --mem-per-cpu=8G

# source ~/.bash_profile
learned=4
tolerance=0
sigma=0.10
iota=0.0
kappa=0.0
extended="-x"
runpath="runs-SMACPerMemory-d$learned-t$tolerance-i$iota-k$kappa-s$sigma"
ok=1
echo "Storing results in $runpath"
for i in 0 1 2 3 4 5 ; do
    j=$i
    if [  $ok != 1 ]; then
        exit 1
    else
	ok=0
    fi
    echo "=================== Starting stage ${i}..."
    python eam.py -n $j --learned=$learned --tolerance=$tolerance $extended --sigma=$sigma --iota=$iota --kappa=$kappa --runpath=$runpath && \
    python eam.py -f $j --learned=$learned --tolerance=$tolerance $extended --sigma=$sigma --iota=$iota --kappa=$kappa --runpath=$runpath && \
    python eam.py -a $j --learned=$learned --tolerance=$tolerance $extended --sigma=$sigma --iota=$iota --kappa=$kappa --runpath=$runpath && \
    python eam.py -e $j --learned=$learned --tolerance=$tolerance $extended --sigma=$sigma --iota=$iota --kappa=$kappa --runpath=$runpath && \
    python eam.py -i $j --learned=$learned --tolerance=$tolerance $extended --sigma=$sigma --iota=$iota --kappa=$kappa --runpath=$runpath && \
    python eam.py -r $j --learned=$learned --tolerance=$tolerance $extended --sigma=$sigma --iota=$iota --kappa=$kappa --runpath=$runpath &&
    ok=1
    echo "=================== Stage $i finished."
done
