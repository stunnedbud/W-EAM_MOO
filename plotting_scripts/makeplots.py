import os
import sys
import csv
import argparse
import matplotlib as mpl
import matplotlib.pyplot as plt
import random


# Directorio de salida
outdir = "Plots"
if not os.path.exists(outdir):
    os.makedirs(outdir)

# Directorios de entrada
datadir_smsemoa = "SMS-EMOA_results"
datadir_weightedsmac = "Weighted_SMAC_results"
smsemoa_csv = datadir_smsemoa + "/EMNIST_configurations_new2023.csv"

# Arreglos de datos
smsemoa_data = []
wsmac_dict = {}
wsmac_monotones = {}
wsmac_bestsolutions = []
smsemoa_header = []
wsmac_header = []
original_best_solution = [0.8868, 0.8361]


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
    print(FP)
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
        print(p_id)

    print(F1)
    # Devolvemos 1er frente
    return F1


# Leer datos de smsemoa
header = True
smsemoa_header = []
with open(smsemoa_csv, mode="r") as file:
    csvFile = csv.reader(file)
    for row in csvFile:
        if header:
            header = False
            smsemoa_header = row
            continue
        smsemoa_data.append(row)
# Filtrar 1er frente de pareto
nds_smsemoa = nds(smsemoa_data)

# Guardamos csvs de mejores soluciones
outfile1 = outdir + "/SMSEMOA_nondominatedsolutions.csv"
with open(outfile1, "w+") as csvfile:
    csvwriter = csv.writer(csvfile)
    csvwriter.writerow(smsemoa_header)  # writing the header
    csvwriter.writerows(nds_smsemoa)  # writing the data rows
print("SMS-EMOA done")

# Leer datos de SMAC pesado
for filename in os.listdir(datadir_weightedsmac):
    fpath = os.path.join(datadir_weightedsmac, filename)
    # checking if it is a file
    if os.path.isfile(fpath):
        # Parseamos valores de W1, W2 del nombre de archivo csv
        tmp_wstr = filename[7:-4]
        tmp_warr = tmp_wstr.split("_")
        W1 = tmp_warr[0].split("-")[1]
        W2 = tmp_warr[1].split("-")[1]

        print(tmp_wstr)

        wsmac_data = []
        wsmac_monotone = []
        best_score = 0.0
        best_solution = []
        header = True
        rownum = -2
        # Leemos datos
        with open(fpath, mode="r") as file:
            csvFile = csv.reader(file)
            for row in csvFile:
                rownum += 1
                if header:
                    header = False
                    if wsmac_header == []:
                        wsmac_header = row
                        wsmac_header = ["W1", "W2"] + wsmac_header
                    continue
                wsmac_data.append(row)
                score = float(row[0])
                if score < best_score:
                    best_solution = row
                    wsmac_monotone.append([rownum, score])
                    best_score = score
                    # print("Found better solution {} on iteration num {}".format(score, rownum))

        # Guardamos en diccionarios
        wsmac_monotones[tmp_wstr] = wsmac_monotone
        wsmac_dict[tmp_wstr] = wsmac_data
        wsmac_bestsolutions.append([W1, W2] + best_solution)


# zip(*li)
smac_X = []
smac_Y = []
for row in wsmac_bestsolutions:
    smac_X.append(float(row[5]))
    smac_Y.append(float(row[7]))

smsemoa_X = []
smsemoa_Y = []
for row in nds_smsemoa:
    smsemoa_X.append(float(row[9]))
    smsemoa_Y.append(float(row[11]))

# Generamos figuras
plt.title("SMS-EMOA: Soluciones no dominadas")
plt.xlabel("Precision (Avg)")
plt.ylabel("Recall (Avg)")
plt.scatter(smsemoa_X, smsemoa_Y, c="blue", marker="x")
plt.scatter(
    [original_best_solution[0]], [original_best_solution[1]], c="green", marker="o"
)
plt.savefig(outdir + "/nds_sms-emoa.png", dpi=1000)
plt.close()

plt.title("WSMAC: Mejores soluciones")
plt.xlabel("Precision (Avg)")
plt.ylabel("Recall (Avg)")
plt.scatter(smac_X, smac_Y, c="red", marker="x")
plt.scatter(
    [original_best_solution[0]], [original_best_solution[1]], c="green", marker="o"
)
plt.savefig(outdir + "/wsmac_best-solutions.png", dpi=1000)
plt.close()


plt.title("Comparación de mejores soluciones encontradas")
plt.xlabel("Precision (Avg)")
plt.ylabel("Recall (Avg)")
plt.scatter(smsemoa_X, smsemoa_Y, c="blue", marker="x")
plt.scatter(smac_X, smac_Y, c="red", marker="x")
plt.scatter(
    [original_best_solution[0]], [original_best_solution[1]], c="green", marker="o"
)
plt.savefig(outdir + "/comparison_pre_rec.png", dpi=1000)
plt.close()

# Figuras de convergencia monotonas
for wstr in wsmac_monotones:
    x, y = zip(*wsmac_monotones[wstr])
    plt.title(wstr)
    plt.xlabel("Iteración")
    plt.ylabel("Score")
    plt.plot(x, y, c="blue", marker="o")
    plt.savefig(outdir + "/convergence_" + wstr + ".png")
    plt.close()


outfile2 = outdir + "/Weighted_SMAC_bestsolutions.csv"
with open(outfile2, "w+") as csvfile:
    csvwriter = csv.writer(csvfile)
    csvwriter.writerow(wsmac_header)  # writing the header
    csvwriter.writerows(wsmac_bestsolutions)  # writing the data rows
