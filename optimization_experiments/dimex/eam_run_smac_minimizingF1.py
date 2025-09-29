# Copyright [2020] Luis Alberto Pineda Cortés, Gibrán Fuentes Pineda,
# Rafael Morales Gamboa.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Entropic Associative Memory Experiments

Usage:
  eam -h | --help
  eam (-n | -f | -a | -c | -e | -i | -r) <stage> [--learned=<learned_data>] [-x]
  [--tolerance=<tolerance>] [--sigma=<sigma>] [--iota=<iota>] [--kappa=<kappa>] [--runpath=<runpath>] [ -l (en | es) ]

Options:
  -h        Show this screen.
  -n        Trains the encoder + classifier Neural networks.
  -f        Generates Features for all data using the encoder.
  -a        Trains the encoder + decoder (Autoencoder) neural networks.
  -c        Generates graphs Characterizing classes of features (by label).
  -e        Run the experiment 1 (Evaluation).
  -i        Increase the amount of data (learning).
  -r        Run the experiment 2 (Recognition).
  --learned=<learned_data>      Selects which learneD Data is used for evaluation, recognition or learning [default: 0].
  -x        Use the eXtended data set as testing data for memory.
  --tolerance=<tolerance>       Allow Tolerance (unmatched features) in memory [default: 0].
  --sigma=<sigma>   Scale of standard deviation of the distribution of influence of the cue [default: 0.25]
  --iota=<iota>     Scale of the expectation (mean) required as minimum for the cue to be accepted by a memory [default: 0.0]
  --kappa=<kappa>   Scale of the expectation (mean) rquiered as minimum for the cue to be accepted by the system of memories [default: 0.0]
  --runpath=<runpath>           Sets the path to the directory where everything will be saved [default: runs]
  -l        Chooses Language for graphs.

The parameter <stage> indicates the stage of learning from which data is used.
Default is the last one.
"""
from docopt import docopt
import copy
import csv
from datetime import datetime
import sys

sys.setrecursionlimit(10000)
import gc
import gettext
from itertools import islice
import numpy as np
from numpy.core.einsumfunc import einsum_path
from joblib import Parallel, delayed
import matplotlib as mpl
import matplotlib.pyplot as plt
import random
import json
from numpy.core.defchararray import array
import seaborn
from associative import AssociativeMemory, AssociativeMemorySystem
import ciempiess
import constants
import dimex
import recnet

# SMAC3
from ConfigSpace import (
    Configuration,
    ConfigurationSpace,
    UniformIntegerHyperparameter,
    UniformFloatHyperparameter,
)
from smac.scenario.scenario import Scenario
from smac.facade.smac_bb_facade import SMAC4BB as BBFacade
from smac.facade.smac_hpo_facade import SMAC4HPO as HPOFacade
from smac.initial_design.default_configuration_design import DefaultConfiguration

# Other
import os
import argparse

import time
import csv


fname_txt = "/home/arivera/smac_runtimes/DIMEX_SMAC_minimizing_F1_ExecutionTimeLogs.txt"
fname_csv = (
    "/home/arivera/smac_runtimes/DIMEX_SMAC_minimizing_F1_ExecutionTimeResultChart.csv"
)


def log(msg, fname):
    print(msg)
    with open(fname, "a+") as f:
        f.write(msg + "\n")


# Translation
gettext.install("ame", localedir=None, codeset=None, names=None)


hl = 2.0  # cota superior de parametros iota, kappa
maxtol = 64  # cota superior para parametro de tolerancia
maxmemsize = 300  # cota superior del tamaño de memoria
experiment = 1

# Solucion inicial para empezar la busqueda
prior_memsize = 23
prior_tolerance = 17
prior_kappa = 0.00151258507990818
prior_iota = 0.601543557593993

n_memories = int(constants.n_labels)


# Parseo de argumentos de entrada (precisionweight, recallweight)
parser = argparse.ArgumentParser(
    description="Este programa recibe dos argumentos posicionales obligatorios, ejemplo: weightedSMAC.py <peso de precision> <peso de recall>"
)
parser.add_argument(
    "precision_weight", type=float, help="W1: A required float positional argument"
)
parser.add_argument(
    "recall_weight", type=float, help="W2: A required float positional argument"
)

args = parser.parse_args()
W1 = args.precision_weight
W2 = args.recall_weight

print("Argumentos de entrada")
print("Precision Weight (W1): " + str(W1))
print("Recall Weight (W2): " + str(W2))

# Directorio de salida
smac_outdir = "DIMEX_Weighted_SMAC_results"
# Historico de configuraciones intentadas
statsfilename = smac_outdir + "/DIMEX_minimizing_F1.csv"
headstr = "Score=-1*F1_SUM, F1 (Promedio), F1 (desviación estándar), Precisión (promedio), Precisión (desviación estándar), Recall (promedio), Recall (desviación estándar), Entropía (promedio), Entropía (desviación estándar), "
for j in range(n_memories):
    headstr += "EAM{0} tolerancia, EAM{0} iota, EAM{0} kappa, EAM{0} Tamaño, EAM{0} F1, EAM{0} Precision, EAM{0} Recall, EAM{0} Entropia, ".format(
        j
    )

if not os.path.exists(smac_outdir):
    os.makedirs(smac_outdir)
if not os.path.exists(os.path.join(os.getcwd(), statsfilename)):
    with open(os.path.join(os.getcwd(), statsfilename), "w+") as outf:
        outf.write(headstr[:-2] + "\n")


# SMAC necesita 4 componentes:
# Espacio de configuracion
# Funcion objetivo
# Escenario
# Fachada

# Generamos una configuración para cada memoria, con el conocimiento a priori que contamos
config = []
for j in range(n_memories):
    config.append(
        UniformIntegerHyperparameter(
            str(j) + "_tolerance", lower=0, upper=maxtol, default_value=prior_tolerance
        )
    )  # (1, 10), default=4))
    config.append(
        UniformFloatHyperparameter(
            str(j) + "_iota", lower=0.0, upper=hl, default_value=prior_iota
        )
    )
    config.append(
        UniformFloatHyperparameter(
            str(j) + "_kappa", lower=0.0, upper=hl, default_value=prior_kappa
        )
    )
    # config.append(UniformIntegerHyperparameter(str(j)+"_memory_size", lower=0, upper=maxmemsize, default_value=prior_memsize))
cs = ConfigurationSpace()
cs.add_hyperparameters(config)

print(config)


def plot_pre_graph(
    pre_mean,
    rec_mean,
    acc_mean,
    ent_mean,
    pre_std,
    rec_std,
    acc_std,
    ent_std,
    es,
    tag="",
    xlabels=constants.memory_sizes,
    xtitle=None,
    ytitle=None,
):

    plt.clf()
    plt.figure(figsize=(6.4, 4.8))

    full_length = 100.0
    step = 0.1
    main_step = full_length / len(xlabels)
    x = np.arange(0, full_length, main_step)

    # One main step less because levels go on sticks, not
    # on intervals.
    xmax = full_length - main_step + step

    # Gives space to fully show markers in the top.
    ymax = full_length + 2

    # Replace undefined precision with 1.0.
    pre_mean = np.nan_to_num(pre_mean, copy=False, nan=100.0)

    plt.errorbar(x, pre_mean, fmt="r-o", yerr=pre_std, label=_("Precision"))
    plt.errorbar(x, rec_mean, fmt="b--s", yerr=rec_std, label=_("Recall"))
    if not ((acc_mean is None) or (acc_std is None)):
        plt.errorbar(x, acc_mean, fmt="y:d", yerr=acc_std, label=_("Accuracy"))

    plt.xlim(0, xmax)
    plt.ylim(0, ymax)
    plt.xticks(x, xlabels)

    if xtitle is None:
        xtitle = _("Range Quantization Levels")
    if ytitle is None:
        ytitle = _("Percentage")

    plt.xlabel(xtitle)
    plt.ylabel(ytitle)
    plt.legend(loc=4)
    plt.grid(True)

    entropy_labels = [str(e) for e in np.around(ent_mean, decimals=1)]

    cmap = mpl.colors.LinearSegmentedColormap.from_list("mycolors", ["cyan", "purple"])
    Z = [[0, 0], [0, 0]]
    levels = np.arange(0.0, xmax, step)
    CS3 = plt.contourf(Z, levels, cmap=cmap)

    cbar = plt.colorbar(CS3, orientation="horizontal")
    cbar.set_ticks(x)
    cbar.ax.set_xticklabels(entropy_labels)
    cbar.set_label(_("Entropy"))

    s = tag + "graph_prse_MEAN" + _("-english")
    graph_filename = constants.picture_filename(s, es)
    plt.savefig(graph_filename, dpi=600)


def plot_size_graph(response_size, size_stdev, es):
    plt.clf()

    full_length = 100.0
    step = 0.1
    main_step = full_length / len(response_size)
    x = np.arange(0, full_length, main_step)

    # One main step less because levels go on sticks, not
    # on intervals.
    xmax = full_length - main_step + step
    ymax = constants.n_labels

    plt.errorbar(
        x,
        response_size,
        fmt="g-D",
        yerr=size_stdev,
        label=_("Average number of responses"),
    )
    plt.xlim(0, xmax)
    plt.ylim(0, ymax)
    plt.xticks(x, constants.memory_sizes)
    plt.yticks(np.arange(0, ymax + 1, 1), range(constants.n_labels + 1))

    plt.xlabel(_("Range Quantization Levels"))
    plt.ylabel(_("Size"))
    plt.legend(loc=1)
    plt.grid(True)

    graph_filename = constants.picture_filename("graph_size_MEAN" + _("-english"), es)
    plt.savefig(graph_filename, dpi=600)


def plot_behs_graph(no_response, no_correct, no_chosen, correct, es):

    for i in range(len(no_response)):
        total = (no_response[i] + no_correct[i] + no_chosen[i] + correct[i]) / 100.0
        no_response[i] /= total
        no_correct[i] /= total
        no_chosen[i] /= total
        correct[i] /= total

    plt.clf()

    full_length = 100.0
    step = 0.1
    main_step = full_length / len(constants.memory_sizes)
    x = np.arange(0.0, full_length, main_step)

    # One main step less because levels go on sticks, not
    # on intervals.
    xmax = full_length - main_step + step
    ymax = full_length
    width = 5  # the width of the bars: can also be len(x) sequence

    plt.bar(x, correct, width, label=_("Correct response chosen"))
    cumm = np.array(correct)
    plt.bar(x, no_chosen, width, bottom=cumm, label=_("Correct response not chosen"))
    cumm += np.array(no_chosen)
    plt.bar(x, no_correct, width, bottom=cumm, label=_("No correct response"))
    cumm += np.array(no_correct)
    plt.bar(x, no_response, width, bottom=cumm, label=_("No responses"))

    plt.xlim(-width, xmax + width)
    plt.ylim(0.0, ymax)
    plt.xticks(x, constants.memory_sizes)

    plt.xlabel(_("Range Quantization Levels"))
    plt.ylabel(_("Labels"))

    plt.legend(loc=0)
    plt.grid(axis="y")

    graph_filename = constants.picture_filename(
        "graph_behaviours_MEAN" + _("-english"), es
    )
    plt.savefig(graph_filename, dpi=600)


def plot_features_graph(domain, means, stdevs, es):
    """Draws the characterist shape of features per label.

    The graph is a dots and lines graph with error bars denoting standard deviations.
    """
    ymin = np.PINF
    ymax = np.NINF
    for i in constants.all_labels:
        yn = (means[i] - stdevs[i]).min()
        yx = (means[i] + stdevs[i]).max()
        ymin = ymin if ymin < yn else yn
        ymax = ymax if ymax > yx else yx
    main_step = 100.0 / domain
    xrange = np.arange(0, 100, main_step)
    fmts = constants.label_formats
    for i in constants.all_labels:
        plt.clf()
        plt.figure(figsize=(12, 5))
        plt.errorbar(xrange, means[i], fmt=fmts[i], yerr=stdevs[i], label=str(i))
        plt.xlim(0, 100)
        plt.ylim(ymin, ymax)
        plt.xticks(xrange, labels="")
        plt.xlabel(_("Features"))
        plt.ylabel(_("Values"))
        plt.legend(loc="right")
        plt.grid(True)
        filename = constants.features_name(es) + "-" + str(i).zfill(3) + _("-english")
        plt.savefig(constants.picture_filename(filename, es), dpi=600)


def plot_conf_matrix(matrix, tags, prefix, es):
    plt.clf()
    plt.figure(figsize=(6.4, 4.8))
    seaborn.heatmap(
        matrix,
        xticklabels=tags,
        yticklabels=tags,
        vmin=0.0,
        vmax=1.0,
        annot=False,
        cmap="Blues",
    )
    plt.xlabel(_("Prediction"))
    plt.ylabel(_("Label"))
    filename = constants.picture_filename(prefix, es)
    plt.savefig(filename, dpi=600)


def plot_memory(memory: AssociativeMemory, prefix, es, fold):
    plt.clf()
    plt.figure(figsize=(6.4, 4.8))
    seaborn.heatmap(
        memory.relation / memory.max_value,
        vmin=0.0,
        vmax=1.0,
        annot=False,
        cmap="coolwarm",
    )
    plt.xlabel(_("Characteristics"))
    plt.ylabel(_("Values"))
    filename = constants.picture_filename(prefix, es, fold)
    plt.savefig(filename, dpi=600)


def plot_memories(ams, es, fold):
    for label in ams:
        prefix = f"memory-{label}-state"
        plot_memory(ams[label], prefix, es, fold)


def get_label(memories, weights=None, entropies=None):
    if len(memories) == 1:
        return memories[0]
    random.shuffle(memories)
    if (entropies is None) or (weights is None):
        return memories[0]
    else:
        i = memories[0]
        entropy = entropies[i]
        weight = weights[i]
        penalty = entropy / weight if weight > 0 else float("inf")
        for j in memories[1:]:
            entropy = entropies[j]
            weight = weights[j]
            new_penalty = entropy / weight if weight > 0 else float("inf")
            if new_penalty < penalty:
                i = j
                penalty = new_penalty
        return i


def msize_features(features, msize, min_value, max_value):
    return np.round(
        (msize - 1) * (features - min_value) / (max_value - min_value)
    ).astype(np.int16)


def rsize_recall(recall, msize, min_value, max_value):
    return (max_value - min_value) * recall / (msize - 1) + min_value


TP = (0, 0)
FP = (0, 1)
FN = (1, 0)
TN = (1, 1)


def conf_sum(cms, t):
    return np.sum([cms[i][t] for i in range(len(cms))])


def memories_precision(cms):
    total = conf_sum(cms, TP) + conf_sum(cms, FN)
    if total == 0:
        return 0.0
    precision = 0.0
    for m in range(len(cms)):
        denominator = cms[m][TP] + cms[m][FP]
        if denominator == 0:
            m_precision = 1.0
        else:
            m_precision = cms[m][TP] / denominator
        weight = (cms[m][TP] + cms[m][FN]) / total
        precision += weight * m_precision
    return precision


def memories_recall(cms):
    total = conf_sum(cms, TP) + conf_sum(cms, FN)
    if total == 0:
        return 0.0
    recall = 0.0
    for m in range(len(cms)):
        m_recall = cms[m][TP] / (cms[m][TP] + cms[m][FN])
        weight = (cms[m][TP] + cms[m][FN]) / total
        recall += weight * m_recall
    return recall


def memories_accuracy(cms):
    total = conf_sum(cms, TP) + conf_sum(cms, FN)
    if total == 0:
        return 0.0
    accuracy = 0.0
    for m in range(len(cms)):
        m_accuracy = (cms[m][TP] + cms[m][TN]) / total
        weight = (cms[m][TP] + cms[m][FN]) / total
        accuracy += weight * m_accuracy
    return accuracy


def register_in_memory(memory, features_iterator):
    # print("Entered register_in_memory")
    # print(features_iterator)
    # memory.register(features_iterator)
    for features in features_iterator:
        # print("feats shape {}".format(features.shape))
        memory.register(features)


def memory_entropy(m, memory: AssociativeMemory):
    return m, memory.entropy


def recognize_by_memory(fl_pairs, ams, entropy):
    n_mems = constants.n_labels
    response_size = np.zeros(n_mems, dtype=int)
    cms = np.zeros((n_mems, 2, 2), dtype="int")
    behaviour = np.zeros((n_mems, constants.n_behaviours), dtype=np.float64)
    for features, label in fl_pairs:
        correct = label
        memories = []
        weights = {}
        for k in ams:
            recognized, weight = ams[k].recognize(features)
            if recognized:
                memories.append(k)
                weights[k] = weight
                response_size[correct] += 1
            # For calculation of per memory precision and recall
            cms[k][TP] += (k == correct) and recognized
            cms[k][FP] += (k != correct) and recognized
            cms[k][TN] += not ((k == correct) or recognized)
            cms[k][FN] += (k == correct) and not recognized
        if len(memories) == 0:
            # Register empty case
            behaviour[correct, constants.no_response_idx] += 1
        elif not (correct in memories):
            behaviour[correct, constants.no_correct_response_idx] += 1
        else:
            l = get_label(memories, weights, entropy)
            if l != correct:
                behaviour[correct, constants.no_correct_chosen_idx] += 1
            else:
                behaviour[correct, constants.correct_response_idx] += 1
    return response_size, cms, behaviour


def split_by_label(fl_pairs):
    label_dict = {}
    n_labels = constants.n_labels
    print("split by label")
    for label in range(n_labels):
        label_dict[label] = []
        # print(label)
    for features, label in fl_pairs:
        if label >= n_labels:
            print(label)
            continue
        label_dict[label].append(features)
    return label_dict.items()


def split_every(n, iterable):
    i = iter(iterable)
    piece = list(islice(i, n))
    while piece:
        yield piece
        piece = list(islice(i, n))


# added parameter: config
def get_ams_results(config, domain, trf, tef, trl, tel, fold):

    n_labels = constants.n_labels
    n_mems = n_labels
    ams = dict.fromkeys(range(n_mems))

    # Round the values
    msize = 64  # max_mem_size # para normalizar entre el maximo tamaño de memoria
    max_value = trf.max()
    other_value = tef.max()
    max_value = max_value if max_value > other_value else other_value

    min_value = trf.min()
    other_value = tef.min()
    min_value = min_value if min_value < other_value else other_value

    trf_rounded = msize_features(trf, msize, min_value, max_value)
    tef_rounded = msize_features(tef, msize, min_value, max_value)

    measures = np.zeros(constants.n_measures, dtype=np.float64)
    entropy = np.zeros(n_mems, dtype=np.float64)
    behaviour = np.zeros((constants.n_labels, constants.n_behaviours), dtype=np.float64)

    # Confusion matrix for calculating precision and recall per memory.
    cms = np.zeros((n_mems, 2, 2), dtype="int")

    # Create the required associative memories.
    sigma = 0.2
    for j in ams:
        tolerance = config[str(j) + "_tolerance"]
        iota = config[str(j) + "_iota"]
        kappa = config[str(j) + "_kappa"]
        msize = 64  # config[str(j)+"_memory_size"]
        ams[j] = AssociativeMemory(domain, msize, tolerance, sigma, iota, kappa)

    # Registration in serial, because I cant get Parallel working on C3 cluster
    for label, features_list in split_by_label(zip(trf_rounded, trl)):
        # if label >= n_labels:
        #    continue # no se que causa este bug pero a veces label = 22 y crashea porque solo hay ams del 0 al 21
        register_in_memory(ams[label], features_list)

    # TODO: Registration in parallel, per label.
    # print("About to Parallel call register_in_memory")
    # print(trf_rounded.shape)
    # print(trl.shape)
    # Parallel(n_jobs=22, require='sharedmem', verbose=50)(
    #    delayed(register_in_memory)(ams[label], features_list) \
    #        for label, features_list in split_by_label(zip(trf_rounded, trl)))

    # Calculate entropies
    means = []
    for m in ams:
        entropy[m] = ams[m].entropy
        means.append(ams[m].mean)

    # Recognition
    response_size = np.zeros(n_mems, dtype=int)
    split_size = 500

    # TODO: Parallel #2
    # for rsize, scms, sbehavs in \
    #     Parallel(n_jobs=1, verbose=50)(
    #        delayed(recognize_by_memory)(fl_pairs, ams, entropy) \
    #        for fl_pairs in split_every(split_size, zip(tef_rounded, tel)))
    #        response_size = response_size + rsize
    #        cms  = cms + scms
    #        behaviour = behaviour + sbehavs

    # Version secuencial #2
    for fl_pairs in split_every(split_size, zip(tef_rounded, tel)):
        rsize, scms, sbehavs = recognize_by_memory(fl_pairs, ams, entropy)
        response_size = response_size + rsize
        cms = cms + scms
        behaviour = behaviour + sbehavs

    counters = [np.count_nonzero(tel == i) for i in range(n_labels)]
    counters = np.array(counters)
    behaviour[:, constants.response_size_idx] = response_size / counters
    all_responses = len(tef_rounded) - np.sum(
        behaviour[:, constants.no_response_idx], axis=0
    )
    all_precision = np.sum(
        behaviour[:, constants.correct_response_idx], axis=0
    ) / float(all_responses)
    all_recall = np.sum(behaviour[:, constants.correct_response_idx], axis=0) / float(
        len(tef_rounded)
    )

    behaviour[:, constants.precision_idx] = all_precision
    behaviour[:, constants.recall_idx] = all_recall

    precision_sum = 0.0
    recall_sum = 0.0
    F1_sum = 0.0
    precisions = []
    recalls = []
    F1s = []

    for i in range(n_mems):
        print(cms[i])
        if (cms[i][TP] + cms[i][FP]) > 0:
            pre = cms[i][TP] / (cms[i][TP] + cms[i][FP])
            rec = cms[i][TP] / (cms[i][TP] + cms[i][FN])
        else:
            pre = 0.0
            rec = 0.0
        if (pre + rec) > 0:
            F1 = (2.0 * pre * rec) / (pre + rec)
        else:
            F1 = 0.0
        # measures[constants.precision_idx,i] = pre
        # measures[constants.recall_idx,i] = rec
        precisions.append(pre)
        recalls.append(rec)
        precision_sum += pre
        recall_sum += rec
        F1_sum += F1
        F1s.append(F1)

    score = (-1) * (F1_sum)  # el objetivo de smac es minimizar este valor
    # score = (-1)*( (W1*precision_sum) + (W2*recall_sum) ) # Modificacion para incluir pesos ponderando los dos objetivos Rec y Pre

    # Parseamos la configuración en forma de renglón para el csv
    outconfig = []
    print(
        "Núm memoria, tolerancia, iota, kappa, tamaño memoria, F1, Precision, Recall, Entropía"
    )
    for j in ams:
        tolerance = config[str(j) + "_tolerance"]
        iota = config[str(j) + "_iota"]
        kappa = config[str(j) + "_kappa"]
        msize = 64  # config[str(j)+"_memory_size"]

        row = [
            tolerance,
            iota,
            kappa,
            msize,
            F1s[j],
            precisions[j],
            recalls[j],
            entropy[j],
        ]
        print(",".join([str(j)] + [str(r) for r in row]))
        outconfig += row

    precisions = np.array(precisions)
    recalls = np.array(recalls)
    entropies = np.array(entropy)
    F1s = np.array(F1s)

    # Guardamos resultados en archivo statsfilename
    # print("\n\nScore = -1*(Suma de F1s), F1 (Promedio), F1 (desviación estándar), Precisión (promedio), Precisión (desviación estándar), Recall (promedio), Recall (desviación estándar), Entropía (promedio), Entropía (desviación estándar)")
    outdata1 = [
        score,
        F1s.mean(),
        F1s.std(),
        precisions.mean(),
        precisions.std(),
        recalls.mean(),
        recalls.std(),
        entropies.mean(),
        entropies.std(),
    ]
    outdata2 = []
    for val in outdata1 + outconfig:
        outdata2.append(str(val))
    run_results_str = ",".join(outdata2)
    run_results_str += "\n"
    with open(os.path.join(os.getcwd(), statsfilename), "a+") as outf:
        outf.write(run_results_str)

    print("Score {}".format(score))
    # positives = conf_sum(cms, TP) + conf_sum(cms, FP)
    # details = True
    # if positives == 0:
    #    print('No memory responded')
    #    measures[constants.precision_idx] = 1.0
    #    details = False
    # else:
    #    measures[constants.precision_idx] = memories_precision(cms)
    # measures[constants.recall_idx] = memories_recall(cms)
    # measures[constants.accuracy_idx] = memories_accuracy(cms)
    # measures[constants.entropy_idx] = np.mean(entropy)

    # if details:
    #    for i in range(n_mems):
    #        positives = cms[i][TP] + cms[i][FP]
    #        if positives == 0:
    #            print(f'Memory {i} of size {msize} in fold {fold} did not respond.')
    return (score, None, behaviour, cms)


# Cargamos una sola vez los datos de feats
filling_features = None
filling_labels = None
testing_features = None
testing_labels = None
es = None  # experiment settings
for fold in [0]:  # range(constants.n_folds):
    gc.collect()
    print(f"Fold: {fold}")
    suffix = constants.filling_suffix
    filling_features_filename = (
        "runs/" + constants.features_name(es) + suffix + "-00" + str(fold) + ".npy"
    )
    filling_labels_filename = (
        "runs/" + constants.labels_name(es) + suffix + "-00" + str(fold) + ".npy"
    )

    suffix = constants.testing_suffix
    testing_features_filename = (
        "runs/" + constants.features_name(es) + suffix + "-00" + str(fold) + ".npy"
    )
    testing_labels_filename = (
        "runs/" + constants.labels_name(es) + suffix + "-00" + str(fold) + ".npy"
    )

    filling_features = np.load(filling_features_filename)
    filling_labels = np.load(filling_labels_filename)
    testing_features = np.load(testing_features_filename)
    testing_labels = np.load(testing_labels_filename)

    #        if filling_features is None:
    #           filling_features = np.load(filling_features_filename)
    #        else:
    #            filling_features = np.append(filling_features, np.load(filling_features_filename))

    #        if filling_labels is None:
    #            filling_labels = np.load(filling_labels_filename)
    #        else:
    #            filling_labels = np.append(filling_labels, np.load(filling_labels_filename))

    #        if testing_features is None:
    #            testing_features = np.load(testing_features_filename)
    #        else:
    #            testing_features = np.append(testing_features, np.load(testing_features_filename))

    #        if testing_labels is None:
    #            testing_labels = np.load(testing_labels_filename)
    #        else:
    #            testing_labels = np.append(testing_labels, np.load(testing_labels_filename))

    print("Finished loading feats and labels for fold {}. Shapes: ".format(fold))
    print("filling_feats {}".format(filling_features.shape))
    print("filling_labels {}".format(filling_labels.shape))
    print("testing_features {}".format(testing_features.shape))
    print("testing_labels {}".format(testing_labels.shape))


# Recibe una configuracion y devuelve un valor real que consiste en
# Para cada memoria sumar las precisiones y recalls:
# (-1)*(SumaDeTodasLasPrecisiones + SumadeTodoslosRecall)
# Este valor es mínimo cuando el precision y recall de todas las memorias es 1
def evaluate_memory_config(config: Configuration) -> float:
    domain = constants.domain
    es = None  # constants.ExperimentSettings(0, 0, False, 0, 0.5, 0.0, 0.0)

    entropy = []
    precision = []
    recall = []
    accuracy = []
    all_precision = []
    all_recall = []
    all_cms = []
    fold = 0  # unused

    no_response = []
    no_correct_response = []
    no_correct_chosen = []
    correct_chosen = []
    response_size = []

    print("Testing the memories")
    # fold = 9

    score, _, _, _ = get_ams_results(
        config,
        domain,
        filling_features,
        testing_features,
        filling_labels,
        testing_labels,
        fold,
    )
    return score

    # measures_per_size = np.zeros(
    #    (len(constants.memory_sizes), constants.n_measures),
    #    dtype=np.float64)
    # behaviours = np.zeros(
    #    (constants.n_labels,
    #    len(constants.memory_sizes),
    #    constants.n_behaviours))
    # list_measures = []
    # list_cms = []
    # for midx, msize in enumerate(constants.memory_sizes):
    # score, measures, behaviour, cms = get_ams_results(config, domain, filling_features, testing_features, filling_labels, testing_labels, fold)
    # return score


# 3. Escenario
# start timer
start_time = time.time()

# Seleccionar variables de ambiente
scenario = Scenario(
    {
        "cs": cs,
        # "output-directory": smac_outdir,
        "run_obj": "quality",
        # "wallclock_limit": 1*86400, #863400 = 24hrs = 2 *  12 hrs = 12*60*60 secs
        # "n-workers":32,  # Use 32 workers
        "deterministic": "true",  # hace que solo 1 semilla se pruebe en cada ejecucion de la funcion objetivo
        "runcount_limit": 1,
    }
)


# 4. Fachada
# Escoge pipelines default o construye una propia
# from smac import BlackBoxFacade as BBFacade
# from smac import HyperparameterOptimizationFacade as HPOFacade
# from smac import MultiFidelityFacade as MFFacade
# from smac import AlgorithmConfigurationFacade as ACFacade
# from smac import RandomFacade as RFacade
# from smac import HyperbandFacade as HBFacade

smac = HPOFacade(
    scenario=scenario,
    tae_runner=evaluate_memory_config,
    initial_design=DefaultConfiguration,
)  # target_function=evaluate_memory_config)
best_found_config = smac.optimize()
print(best_found_config)

# end timer
end_time = time.time()
log(
    f"Ended SMAC Run with n_trials = 1. Total execution time: {end_time - start_time:.2f} seconds",
    fname_txt,
)

# EXP 2
# start timer
start_time = time.time()

# Seleccionar variables de ambiente
scenario = Scenario(
    {
        "cs": cs,
        # "output-directory": smac_outdir,
        "run_obj": "quality",
        # "wallclock_limit": 1*86400, #863400 = 24hrs = 2 *  12 hrs = 12*60*60 secs
        # "n-workers":32,  # Use 32 workers
        "deterministic": "true",  # hace que solo 1 semilla se pruebe en cada ejecucion de la funcion objetivo
        "runcount_limit": 10,
    }
)


# 4. Fachada
# Escoge pipelines default o construye una propia
# from smac import BlackBoxFacade as BBFacade
# from smac import HyperparameterOptimizationFacade as HPOFacade
# from smac import MultiFidelityFacade as MFFacade
# from smac import AlgorithmConfigurationFacade as ACFacade
# from smac import RandomFacade as RFacade
# from smac import HyperbandFacade as HBFacade

smac = HPOFacade(
    scenario=scenario,
    tae_runner=evaluate_memory_config,
    initial_design=DefaultConfiguration,
)  # target_function=evaluate_memory_config)
best_found_config = smac.optimize()
print(best_found_config)

# end timer
end_time = time.time()
log(
    f"Ended SMAC Run with n_trials = 10. Total execution time: {end_time - start_time:.2f} seconds",
    fname_txt,
)
