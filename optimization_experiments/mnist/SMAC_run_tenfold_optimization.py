# SMAC3
from ConfigSpace import Configuration, ConfigurationSpace
from smac.scenario.scenario import Scenario
from smac.facade.smac_bb_facade import SMAC4BB as BBFacade
from smac.facade.smac_hpo_facade import SMAC4HPO as HPOFacade
from smac.initial_design.default_configuration_design import DefaultConfiguration

# EAM
import constants
import convnet
from associative import AssociativeMemory

# Other
import os
import sys
import gc
import argparse
import numpy as np

# from joblib import Parallel, delayed
import matplotlib as mpl
import matplotlib.pyplot as plt
import random

# Directorio de salida
smac_outdir = "SMAC_tenfold_results"
# Historico de configuraciones intentadas
statsfilename = smac_outdir + "/configs_statistics.csv"

if not os.path.exists(smac_outdir):
    os.makedirs(smac_outdir)
if not os.path.exists(os.path.join(os.getcwd(), statsfilename)):
    with open(os.path.join(os.getcwd(), statsfilename), "w+") as outf:
        outf.write(
            "Ponderación de desempeño, Tamaño de memoria, Tolerancia, Sigma, Iota, Kappa, Precisión (promedio), Precisión (desviación estándar), Recall (promedio), Recall (desviación estándar), Entropía (promedio), Entropía (desviación estándar)\n"
        )


# SMAC necesita 4 componentes:
# Espacio de configuracion
# Funcion objetivo
# Escenario
# Fachada

hl = 2.0  # cota superior de parametros sigma, iota, kappa
maxtol = 64  # cota superior para parametro de tolerancia
cs = ConfigurationSpace(
    {
        "tolerance": (0, maxtol),  # Uniform Integer
        "iota": (0.0, hl),
        "kappa": (0.0, hl),
        "memory_size": (1, 100),
    }
)


def get_label(memories, entropies=None):
    # Random selection
    if entropies is None:
        i = random.atddrange(len(memories))
        return memories[i]
    else:
        i = memories[0]
        entropy = entropies[i]

        for j in memories[1:]:
            if entropy > entropies[j]:
                i = j
                entropy = entropies[i]

    return i


# Función auxiliar a la función objetivo
# Crea un sistema de memoria w-ams usando los parametros dados y lo evalua
def get_wams_results(
    msize, domain, lpm, trf, tef, trl, tel, tolerance, sigma, iota, kappa
):
    # Round the values
    max_value = trf.max()
    other_value = tef.max()
    max_value = max_value if max_value > other_value else other_value

    min_value = trf.min()
    other_value = tef.min()
    min_value = min_value if min_value < other_value else other_value

    trf_rounded = np.round(
        (trf - min_value) * (msize - 1) / (max_value - min_value)
    ).astype(np.int16)
    tef_rounded = np.round(
        (tef - min_value) * (msize - 1) / (max_value - min_value)
    ).astype(np.int16)

    n_labels = constants.n_labels
    nmems = int(n_labels / lpm)

    measures = np.zeros((constants.n_measures, nmems), dtype=np.float64)
    entropy = np.zeros((nmems,), dtype=np.float64)
    behaviour = np.zeros((constants.n_behaviours,))

    # Confusion matrix for calculating precision and recall per memory.
    cms = np.zeros((nmems, 2, 2))
    TP = (0, 0)
    FP = (0, 1)
    FN = (1, 0)
    TN = (1, 1)

    # Create the required associative memories.
    ams = dict.fromkeys(range(nmems))
    # tolerance = 5
    # sigma = 0.1
    # iota = 0.3
    # kappa = 1.5
    for j in ams:
        ams[j] = AssociativeMemory(
            domain, msize, tolerance, sigma, iota, kappa
        )  # AssociativeMemory(domain, msize)

    # Registration
    for features, label in zip(trf_rounded, trl):
        i = int(label / lpm)
        ams[i].register(features)

    # Calculate entropies
    for j in ams:
        entropy[j] = ams[j].entropy

    # Recognition
    response_size = 0

    for features, label in zip(tef_rounded, tel):
        correct = int(label / lpm)

        memories = []
        for k in ams:
            recognized, weight = ams[k].recognize(features)

            # For calculation of per memory precision and recall
            # if (k == correct) and recognized:
            #    cms[k][TP] += 1
            # elif k == correct:
            #    cms[k][FN] += 1
            # elif recognized:
            #    cms[k][FP] += 1
            # else:
            #    cms[k][TN] += 1

            # For calculation of behaviours, including overall precision and recall.
            # if recognized:
            #    memories.append(k)

            # For calculation of per memory precision and recall
            cms[k][TP] += (k == correct) and recognized
            cms[k][FP] += (k != correct) and recognized
            cms[k][TN] += not ((k == correct) or recognized)
            cms[k][FN] += (k == correct) and not recognized

            # For calculation of behaviours, including overall precision and recall.
            if recognized:
                memories.append(k)
                # weights[k] = weight
                # response_size[correct] += 1

        response_size += len(memories)
        if len(memories) == 0:
            # Register empty case
            behaviour[constants.no_response_idx] += 1
        elif not (correct in memories):
            behaviour[constants.no_correct_response_idx] += 1
        else:
            l = get_label(memories, entropy)
            if l != correct:
                behaviour[constants.no_correct_chosen_idx] += 1
            else:
                behaviour[constants.correct_response_idx] += 1

    behaviour[constants.mean_responses_idx] = response_size / float(len(tef_rounded))
    all_responses = len(tef_rounded) - behaviour[constants.no_response_idx]
    all_precision = (behaviour[constants.correct_response_idx]) / float(all_responses)
    all_recall = (behaviour[constants.correct_response_idx]) / float(len(tef_rounded))

    # print("Sanity check {} {}".format(constants.correct_response_idx, behaviour))
    # print("Sanity check {} {} {}".format(all_responses,all_precision,all_recall))

    behaviour[constants.precision_idx] = all_precision
    behaviour[constants.recall_idx] = all_recall

    precision_sum = 0.0
    recall_sum = 0.0
    F1_sum = 0.0
    precisions = []
    recalls = []
    F1s = []

    for i in range(nmems):
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
        measures[constants.precision_idx, i] = pre
        measures[constants.recall_idx, i] = rec
        precisions.append(pre)
        recalls.append(rec)
        precision_sum += pre
        recall_sum += rec
        F1_sum += F1
        F1s.append(F1)

    score = (-1) * (F1_sum)
    # score = (-1)*(precision_sum+recall_sum) # el objetivo de todo esto es minimizar este valor
    precisions = np.array(precisions)
    recalls = np.array(recalls)
    entropies = np.array(entropy)
    # Guardamos resultados en archivo statsfilename
    print(
        "Score(-1*F1sum), Tamaño de memoria, Tolerancia, Sigma, Iota, Kappa, Precisión (promedio), Precisión (desviación estándar), Recall (promedio), Recall (desviación estándar), Entropía (promedio), Entropía (desviación estándar)"
    )
    outdata1 = [
        score,
        msize,
        tolerance,
        sigma,
        iota,
        kappa,
        precisions.mean(),
        precisions.std(),
        recalls.mean(),
        recalls.std(),
        entropies.mean(),
        entropies.std(),
    ]
    outdata2 = []
    for val in outdata1:
        outdata2.append(str(val))
    run_results_str = ",".join(outdata2)
    run_results_str += "\n"
    print(run_results_str)
    with open(os.path.join(os.getcwd(), statsfilename), "a+") as outf:
        outf.write(run_results_str)

    return (score, measures, entropy, behaviour)


# Funcion objetivo
# Recibe una configuracion y devuelve un valor real que consiste en
# Para cada memoria sumar las precisiones y recalls:
# (-1)*(SumaDeTodasLasPrecisiones + SumadeTodoslosRecall)
# Este valor es mínimo cuando el precision y recall de todas las memorias es 1
def evaluate_memory_config(
    config: Configuration,
) -> float:  # (self, config: Configuration, seed: int) -> float:
    # def test_memories(domain, prefix, experiment):
    domain = constants.domain
    prefix = constants.partial_prefix
    experiment = 1
    tolerance = config["tolerance"]
    iota = config["iota"]
    kappa = config["kappa"]
    msize = config["memory_size"]

    labels_x_memory = constants.labels_per_memory[experiment]  # = 1
    n_memories = int(constants.n_labels / labels_x_memory)  # = n_labels

    if prefix == constants.partial_prefix:
        suffix = constants.filling_suffix
    elif prefix == constants.full_prefix:
        suffix = constants.training_suffix

    scores = []
    # i = constants.training_stages - 1
    for i in range(constants.training_stages):
        gc.collect()

        training_features_filename = prefix + constants.features_name + suffix
        training_features_filename = constants.data_filename(
            training_features_filename, i
        )
        training_labels_filename = prefix + constants.labels_name + suffix
        training_labels_filename = constants.data_filename(training_labels_filename, i)

        suffix = constants.testing_suffix
        testing_features_filename = prefix + constants.features_name + suffix
        testing_features_filename = constants.data_filename(
            testing_features_filename, i
        )
        testing_labels_filename = prefix + constants.labels_name + suffix
        testing_labels_filename = constants.data_filename(testing_labels_filename, i)

        training_features = np.load(training_features_filename)
        training_labels = np.load(training_labels_filename)
        testing_features = np.load(testing_features_filename)
        testing_labels = np.load(testing_labels_filename)

        # Each memory has precision and recall
        # measures_per_size = np.zeros((1, n_memories, constants.n_measures), dtype=np.float64)

        # An entropy value per memory size and memory.
        # entropies = np.zeros((1, n_memories), dtype=np.float64)
        # behaviours = np.zeros((1, constants.n_behaviours))
        sigma = 0.1
        score, measures, entropy, behaviour = get_wams_results(
            msize,
            domain,
            labels_x_memory,
            training_features,
            testing_features,
            training_labels,
            testing_labels,
            tolerance,
            sigma,
            iota,
            kappa,
        )
        scores.append(score)

    score_avg = sum(scores) / len(scores)
    return score_avg


# 3. Escenario
# Seleccionar variables de ambiente
scenario = Scenario(
    {
        "cs": cs,
        "run_obj": "quality",
        "wallclock_limit": 86400,  # 24 hrs
        "deterministic": "true",  # hace que solo 1 semilla se pruebe en cada ejecucion de la funcion objetivo
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
    scenario=scenario, tae_runner=evaluate_memory_config
)  # target_function=evaluate_memory_config)
best_found_config = smac.optimize()
print(best_found_config)
