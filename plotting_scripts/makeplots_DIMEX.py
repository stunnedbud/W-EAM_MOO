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
outdir = "OUT_PLOTS"
if not os.path.exists(outdir):
    os.makedirs(outdir)

# Directorios de entrada
# smsemoa_csv = datadir_smsemoa+"/DIMEX_configurations.csv"
# smac_apriori_csv = "DIMEX_Weighted_SMAC_results/DIMEX_minimizing_F1.csv"

datadir_weightedsmac = "DIMEX_Weighted_SMAC_results"
smsemoa_apriori_csv = "SMSEMOA_configurations_APRIORI.csv"
smsemoa_aposteriori_csv = "SMSEMOA_configurations_APOSTERIORI.csv"
smac_apriori_csv = "SMAC_configurations_APRIORI.csv"
smac_aposteriori_csv = "SMAC_configurations_APOSTERIORI.csv"
smac_cyclic_csv = "all_configs_stage_000.csv"  # "WEAM_cyclic_optim_configurations/all_configs_stage_005.csv"


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


# Leer datos de smsemoa A PRIORI
header = True
smsemoa_header = []
smsemoa_apriori_data = []
with open(smsemoa_apriori_csv, mode="r") as file:
    csvFile = csv.reader(file)
    for row in csvFile:
        if header:
            header = False
            smsemoa_header = row
            continue
        smsemoa_apriori_data.append(row)
# Calcular grafica de convergencia monotona de SMS-EMOA
best_f1 = 0.0
start = True
i = 0
smsemoa_apriori_X = []
smsemoa_apriori_Y = []
for p in smsemoa_apriori_data:
    i += 1
    if i > Xlimit:
        break
    f1 = float(p[2])  # avg f1 de todas las memorias
    if start or f1 > best_f1:
        best_f1 = f1
        smsemoa_apriori_X.append(i)
        smsemoa_apriori_Y.append(f1)
        start = False


# Leer datos de smsemoa A POSTERIORI
header = True
smsemoa_header = []
smsemoa_aposteriori_data = []
with open(smsemoa_aposteriori_csv, mode="r") as file:
    csvFile = csv.reader(file)
    for row in csvFile:
        if header:
            header = False
            smsemoa_header = row
            continue
        smsemoa_aposteriori_data.append(row)
# Calcular grafica de convergencia monotona de SMS-EMOA
best_f1 = 0.0
start = True
i = 0
smsemoa_aposteriori_X = []
smsemoa_aposteriori_Y = []
for p in smsemoa_aposteriori_data:
    i += 1
    if i > Xlimit:
        break
    f1 = float(p[2])  # avg f1 de todas las memorias
    if start or f1 > best_f1:
        best_f1 = f1
        smsemoa_aposteriori_X.append(i)
        smsemoa_aposteriori_Y.append(f1)
        start = False


# Leer datos de SMAC a priori
header = True
smac_apriori_data = []
with open(smac_apriori_csv, mode="r") as file:
    csvFile = csv.reader(file)
    for row in csvFile:
        if header:
            header = False
            # smsemoa_header = row
            continue
        smac_apriori_data.append(row)
# Calcular grafica de convergencia monotona de SMS-EMOA
best_f1 = 0.0
start = True
i = 0
smac_apriori_X = []
smac_apriori_Y = []
for p in smac_apriori_data:
    i += 1
    if i > Xlimit:
        break
    f1 = float(p[1])  # avg f1 de todas las memorias
    if start or f1 > best_f1:
        best_f1 = f1
        smac_apriori_X.append(i)
        smac_apriori_Y.append(f1)
        start = False


# Leer datos de SMAC a posteriori
header = True
smac_aposteriori_data = []
with open(smac_aposteriori_csv, mode="r") as file:
    csvFile = csv.reader(file)
    for row in csvFile:
        if header:
            header = False
            # smsemoa_header = row
            continue
        smac_aposteriori_data.append(row)
# Calcular grafica de convergencia monotona de SMS-EMOA
best_f1 = 0.0
start = True
i = 0
smac_aposteriori_X = []
smac_aposteriori_Y = []
for p in smac_aposteriori_data:
    i += 1
    if i > Xlimit:
        break
    f1 = float(p[1])  # avg f1 de todas las memorias
    if start or f1 > best_f1:
        best_f1 = f1
        smac_aposteriori_X.append(i)
        smac_aposteriori_Y.append(f1)
        start = False


# Leer datos de SMAC ciclico ( LAST STAGE )
header = True
smac_cyclic_data = []
with open(smac_cyclic_csv, mode="r") as file:
    csvFile = csv.reader(file)
    for row in csvFile:
        if header:
            header = False
            # smsemoa_header = row
            continue
        smac_cyclic_data.append(row)
# Calcular grafica de convergencia monotona de SMS-EMOA
best_f1 = 0.0
start = True
i = 0
smac_cyclic_X = []
smac_cyclic_Y = []
for p in smac_cyclic_data:
    i += 1
    if i > Xlimit:
        break
    f1 = float(p[1])  # avg f1 de todas las memorias
    if start or f1 > best_f1:
        best_f1 = f1
        smac_cyclic_X.append(i)
        smac_cyclic_Y.append(f1)
        start = False


plt.title("W-EAM DIMEX: Comparación de convergencia")
plt.xlabel("Iteración")
plt.ylabel("F1 (Promedio de las mems)")
plt.plot(
    smsemoa_apriori_X, smsemoa_apriori_Y, c="cyan", marker="", label="SMS-EMOA a priori"
)
plt.plot(
    smsemoa_aposteriori_X,
    smsemoa_aposteriori_Y,
    c="blue",
    marker="",
    label="SMS-EMOA a posteriori",
)
plt.plot(smac_apriori_X, smac_apriori_Y, c="red", marker="", label="SMAC a priori")
plt.plot(
    smac_aposteriori_X,
    smac_aposteriori_Y,
    c="tab:orange",
    marker="",
    label="SMAC a posteriori",
)
# plt.plot(smac_cyclic_X, smac_cyclic_Y, c="green", marker="", label="SMAC cíclico")
# specifying horizontal line type
plt.axhline(
    y=originalF1, color="yellow", linestyle="-", label="Escenario V (Replicado)"
)
leg = plt.legend(loc="lower right")
plt.savefig(outdir + "/convergence_comparison_dimex.csv", format="svg")
plt.close()

print("Convergence comparison done")


### COMENTAR DE AQUI PARA ABAJO PARA SOLO GRAFICAR LA COMPARACION DE CONVERGENCIAS


# Filtrar 1er frente de pareto
nds_smsemoa = nds(smsemoa_aposteriori_data)

# Guardamos csvs de mejores soluciones
outfile1 = outdir + "/SMSEMOA_apos_nondominatedsolutions.csv"
with open(outfile1, "w+") as csvfile:
    csvwriter = csv.writer(csvfile)
    csvwriter.writerow(smsemoa_header)  # writing the header
    csvwriter.writerows(nds_smsemoa)  # writing the data rows
print("SMS-EMOA NDS done")

# Leer datos de SMAC pesado
for filename in os.listdir(datadir_weightedsmac):
    fpath = os.path.join(datadir_weightedsmac, filename)
    # checking if it is a file
    if os.path.isfile(fpath):
        # Parseamos valores de W1, W2 del nombre de archivo csv
        tmp_wstr = filename[7:-4]
        if not tmp_wstr.__contains__("-"):  # es el archivo de configs_optimizando_F1
            continue
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
nds_smsemoa.sort(
    key=lambda r: r[4]
)  # (lambda x, y: x[9]) # ordenamos segun el valor del eje x para poder obtener una muestra distribuida equitativamente
xlimit = 16
x = 1
for row in nds_smsemoa:
    x += 1
    if x < xlimit:
        continue
    smsemoa_X.append(float(row[4]))
    smsemoa_Y.append(float(row[6]))
    x = 0

# Generamos figuras
plt.title("SMS-EMOA: Soluciones no dominadas")
plt.xlabel("Precision (Avg)")
plt.ylabel("Recall (Avg)")
plt.scatter(smsemoa_X, smsemoa_Y, c="blue", marker="x")
plt.scatter(
    [original_best_solution[0]], [original_best_solution[1]], c="green", marker="o"
)
plt.savefig(outdir + "/nds_sms-emoa.svg", format="svg")
plt.close()

plt.title("W-SMAC: Mejores soluciones")
plt.xlabel("Precision (Avg)")
plt.ylabel("Recall (Avg)")
plt.scatter(smac_X, smac_Y, c="red", marker="x")
plt.scatter(
    [original_best_solution[0]], [original_best_solution[1]], c="green", marker="o"
)
plt.savefig(outdir + "/wsmac_best-solutions.svg", format="svg")
plt.close()

plt.title("Comparación de mejores soluciones encontradas")
plt.xlabel("Precision (Avg)")
plt.ylabel("Recall (Avg)")
plt.scatter(smsemoa_X, smsemoa_Y, c="blue", marker="x", label="SMS-EMOA")
plt.scatter(smac_X, smac_Y, c="red", marker="x", label="W-SMAC")
plt.scatter(
    [original_best_solution[0]],
    [original_best_solution[1]],
    c="yellow",
    marker="o",
    label="Artículo original",
)
leg = plt.legend(loc="lower left")
plt.savefig(outdir + "/comparison_pre_rec.svg", format="svg")
plt.close()

# Figuras de convergencia monotonas
for wstr in wsmac_monotones:
    x, y = zip(*wsmac_monotones[wstr])
    plt.title(wstr)
    plt.xlabel("Iteración")
    plt.ylabel("Score")
    plt.plot(x, y, c="blue", marker="o")
    plt.savefig(outdir + "/convergence_" + wstr + ".svg", format="svg")
    plt.close()

outfile2 = outdir + "/Weighted_SMAC_bestsolutions.csv"
with open(outfile2, "w+") as csvfile:
    csvwriter = csv.writer(csvfile)
    csvwriter.writerow(wsmac_header)  # writing the header
    csvwriter.writerows(wsmac_bestsolutions)  # writing the data rows
