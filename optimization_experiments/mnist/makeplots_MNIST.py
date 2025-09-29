import os
import sys
import csv
import argparse
import matplotlib as mpl
import matplotlib.pyplot as plt
import random

# Articulo original de EAM EMNIST
original_best_solution = [0.984, 0.956]
Xlimit = 100000

# Arreglos de datos
smsemoa_header = []

# Directorio de salida
outdir = "PlotsMNIST"
if not os.path.exists(outdir):
    os.makedirs(outdir)

# Directorios de entrada
smsemoa_csv = "SMS-EMOA_tenfold_results/MNIST_configurations.csv"
smac_csv = "SMAC_tenfold_results/configs_statistics.csv"


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


print("Starting reading SMSEMOA")
# Leer datos de smsemoa
header = True
smsemoa_header = []
smsemoa_data = []
with open(smsemoa_csv, mode="r") as file:
    csvFile = csv.reader(file)
    for row in csvFile:
        if header:
            header = False
            smsemoa_header = row
            continue
        smsemoa_data.append(row)
print("Finished")
# Calcular grafica de convergencia monotona de SMS-EMOA
best_f1 = 0.0
start = True
i = 0
smsemoa_X = []
smsemoa_Y = []
print("Starting convergence")
for p in smsemoa_data:
    i += 1
    if i > Xlimit:
        break
    f1 = float(p[3])  # avg f1 de todas las memorias
    if start or f1 > best_f1:
        best_f1 = f1
        smsemoa_X.append(i)
        smsemoa_Y.append(f1)
        start = False


# Leer datos de SMAC
header = True
smac_noprior_data = []
with open(smac_csv, mode="r") as file:
    csvFile = csv.reader(file)
    for row in csvFile:
        if header:
            header = False
            # smsemoa_header = row
            continue
        smac_noprior_data.append(row)
# Calcular grafica de convergencia monotona de SMAC
best_f1 = 0.0
start = True
i = 0
smac_noprior_X = []
smac_noprior_Y = []
for p in smac_noprior_data:
    i += 1
    if i > Xlimit:
        break
    f1 = (-1 * float(p[0])) / 10  # avg f1 de todas las memorias
    if start or f1 > best_f1:
        best_f1 = f1
        smac_noprior_X.append(i)
        smac_noprior_Y.append(f1)
        start = False


plt.title("W-EAM MNIST: Comparación de convergencia")
plt.xlabel("Iteración")
plt.ylabel("Score")
plt.plot(smsemoa_X, smsemoa_Y, c="blue", marker=".", label="SMS-EMOA")
plt.plot(smac_noprior_X, smac_noprior_Y, c="red", marker=".", label="SMAC")
# specifying horizontal line type
plt.axhline(y=originalF1, color="yellow", linestyle="-", label="EAM original")
leg = plt.legend(loc="lower right")
plt.savefig("convergence_comparison_MNIST.svg", format="svg")
plt.close()


# Filtrar 1er frente de pareto
nds_smsemoa = nds(smsemoa_data)

# Guardamos csvs de mejores soluciones
outfile1 = outdir + "/SMSEMOA_nondominatedsolutions.csv"
with open(outfile1, "w+") as csvfile:
    csvwriter = csv.writer(csvfile)
    csvwriter.writerow(smsemoa_header)  # writing the header
    csvwriter.writerows(nds_smsemoa)  # writing the data rows
print("SMS-EMOA done")

plt.title("SMS-EMOA: Soluciones no dominadas")
plt.xlabel("Precision (Avg)")
plt.ylabel("Recall (Avg)")
plt.scatter(smsemoa_X, smsemoa_Y, c="blue", marker=".")
plt.scatter(
    [original_best_solution[0]], [original_best_solution[1]], c="green", marker="x"
)
plt.savefig(outdir + "/nds_sms-emoa.svg", format="svg")
plt.close()
