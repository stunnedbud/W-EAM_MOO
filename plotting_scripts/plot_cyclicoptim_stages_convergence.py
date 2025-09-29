import os
import sys
import csv
import argparse
import matplotlib as mpl
import matplotlib.pyplot as plt
import random

# Articulo original de EAM EMNIST
original_best_solution = [0.872, 0.817]
Xlimit = 100000

# Arreglos de datos
wsmac_dict = {}
wsmac_monotones = {}
wsmac_bestsolutions = []
smsemoa_header = []
wsmac_header = []

# Directorio de salida
indir = "cyclicoptim_csvs"
outdir = "PLOTS_CyclicOptim"
if not os.path.exists(outdir):
    os.makedirs(outdir)


def F1score(pre, rec):
    if (pre + rec) > 0:
        return (2.0 * pre * rec) / (pre + rec)
    else:
        return 0.0


originalF1 = F1score(original_best_solution[0], original_best_solution[1])

# Funcion auxiliar para comparar la dominancia de Pareto entre dos vectores
# Devuelve 1 si p < q ie. p domina a q
#         -1 si q < p ie. q domina a p
#          0 eoc.
def compareParetoDominance(p, q):
    pdq = True  # p domina a q
    qdp = True  # q domina a p
    for i in range(len(p)):  # p y q deben ser del mismo tamaño
        if p[i] < q[i]:
            qdp = False
        if q[i] < p[i]:
            pdq = False
    if pdq and not qdp:
        return 1
    if qdp and not pdq:
        return -1
    return 0


# Funcion auxiliar para filtrar solo soluciones no dominadas de SMS EMOA
def nds(P):
    FP = []
    F1 = []
    for p in P:
        FP.append(
            (float(p[0]), float(p[1]))
        )  # solo ordenaremos usando las primeras dos columnas de datos
    # print(FP)
    p_id = 0
    for p in P:
        sp = []  # soluciones dominadas por p
        n_p = 0  # conteo de soluciones que dominan a p
        q_id = 0
        for q in P:
            if p == q:
                q_id += 1
                continue
            f_p = FP[p_id]
            f_q = FP[q_id]
            pd = compareParetoDominance(f_p, f_q)
            if pd == 1:  # p < q
                sp.append(q_id)
            elif pd == -1:  # q < p
                n_p += 1
            q_id += 1

        if n_p == 0:  # prank = 1, lo representaremos como np == -1
            n_p = -1
            F1.append(p)
        # NP[p_id] = n_p
        # SP[p_id] = sp
        p_id += 1
        # print(p_id)

    # print(F1)
    # Devolvemos 1er frente
    return F1


# Leer datos de SMAC ciclico ( LAST STAGE )
header = True
smac_cyclic_data = {}  # []
smac_cyclic_X = {}
smac_cyclic_Y = {}

for stgnum in range(6):
    fname = indir + "/all_configs_stage_00" + str(stgnum) + ".csv"
    print(fname)
    csvdata = []
    with open(fname, mode="r") as file:
        csvFile = csv.reader(file)
        for row in csvFile:
            if header:
                header = False
                continue
            csvdata.append(row)
    smac_cyclic_data[stgnum] = csvdata

    best_f1 = 0.0
    start = True
    i = 0
    X = []
    Y = []
    for p in csvdata:
        i += 1
        if i > Xlimit:
            break
        f1 = float(p[1])  # avg f1 de todas las memorias
        if start or f1 > best_f1:
            best_f1 = f1
            X.append(i)
            Y.append(f1)
            start = False
    smac_cyclic_X[stgnum] = X
    smac_cyclic_Y[stgnum] = Y
    print(smac_cyclic_X)


plt.title("Cyclical SMAC: Convergence comparison")
plt.xlabel("Iteration")
plt.ylabel("F1 (Average of memories)")
plt.plot(smac_cyclic_X[0], smac_cyclic_Y[0], c="cyan", marker="", label="Stage 0")
plt.plot(smac_cyclic_X[1], smac_cyclic_Y[1], c="blue", marker="", label="Stage 1")
plt.plot(smac_cyclic_X[2], smac_cyclic_Y[2], c="red", marker="", label="Stage 2")
plt.plot(smac_cyclic_X[3], smac_cyclic_Y[3], c="tab:orange", marker="", label="Stage 3")
plt.plot(smac_cyclic_X[4], smac_cyclic_Y[4], c="tab:purple", marker="", label="Stage 4")
plt.plot(smac_cyclic_X[5], smac_cyclic_Y[5], c="tab:pink", marker="", label="Stage 5")
# specifying horizontal line type
plt.axhline(
    y=originalF1, color="yellow", linestyle="-", label="Scenario V (Replicated)"
)
leg = plt.legend(loc="lower right")
plt.savefig(outdir + "/cyclicoptim_stgs_convergence_comparison.svg", format="svg")
plt.close()

print("Convergence comparison done")
