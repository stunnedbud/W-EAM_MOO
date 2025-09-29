# SMAC3
from ConfigSpace import Configuration, ConfigurationSpace
from smac.scenario.scenario import Scenario
from smac.facade.smac_bb_facade import SMAC4BB as BBFacade

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
import matplotlib as mpl
import matplotlib.pyplot as plt
import random

import time
import csv


fname_txt = "/home/arivera/smac_runtimes/MNIST_ExecutionTimeLogs.txt"
fname_csv = "/home/arivera/smac_runtimes/MNIST_ExecutionTimeResultChart.csv"


def log(msg, fname):
    print(msg)
    with open(fname, "a+") as f:
        f.write(msg + "\n")


# Directorio de salida
smac_outdir = "SMAC_results"
# Historico de configuraciones intentadas
statsfilename = smac_outdir + "/configurations_statistics.csv"

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

hl = 5.0  # cota superior de parametros sigma, iota, kappa
maxtol = 1  # cota superior para parametro de tolerancia
cs = ConfigurationSpace(
    {
        "tolerance": (0, maxtol),  # Uniform Integer
        "sigma": (0.0, hl),  # Uniform Float
        "iota": (0.0, hl),
        "kappa": (0.0, hl),
        "memory_size": (1, 250),
        # "species": ["mouse", "cat", "dog"],   # Categorical
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
            if (k == correct) and recognized:
                cms[k][TP] += 1
            elif k == correct:
                cms[k][FN] += 1
            elif recognized:
                cms[k][FP] += 1
            else:
                cms[k][TN] += 1

            # For calculation of behaviours, including overall precision and recall.
            if recognized:
                memories.append(k)

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

    print("Sanity check {} {}".format(constants.correct_response_idx, behaviour))
    print("Sanity check {} {} {}".format(all_responses, all_precision, all_recall))

    behaviour[constants.precision_idx] = all_precision
    behaviour[constants.recall_idx] = all_recall

    precision_sum = 0.0
    recall_sum = 0.0
    precisions = []
    recalls = []
    for i in range(nmems):
        print(cms[i])
        if (cms[i][TP] + cms[i][FP]) > 0:
            pre = cms[i][TP] / (cms[i][TP] + cms[i][FP])
            rec = cms[i][TP] / (cms[i][TP] + cms[i][FN])
        else:
            pre = 0.0
            rec = 0.0
        measures[constants.precision_idx, i] = pre
        measures[constants.recall_idx, i] = rec
        precisions.append(pre)
        recalls.append(rec)
        precision_sum += pre
        recall_sum += rec

    score = (-1) * (
        precision_sum + recall_sum
    )  # el objetivo de todo esto es minimizar este valor
    precisions = np.array(precisions)
    recalls = np.array(recalls)
    entropies = np.array(entropy)
    # Guardamos resultados en archivo statsfilename
    print(
        "Ponderación de desempeño, Tamaño de memoria, Tolerancia, Sigma, Iota, Kappa, Precisión (promedio), Precisión (desviación estándar), Recall (promedio), Recall (desviación estándar), Entropía (promedio), Entropía (desviación estándar)"
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
    sigma = config["sigma"]
    iota = config["iota"]
    kappa = config["kappa"]
    msize = config["memory_size"]

    labels_x_memory = constants.labels_per_memory[experiment]  # = 1
    n_memories = int(constants.n_labels / labels_x_memory)  # = n_labels

    if prefix == constants.partial_prefix:
        suffix = constants.filling_suffix
    elif prefix == constants.full_prefix:
        suffix = constants.training_suffix

    i = constants.training_stages - 1
    # for i in range(constants.training_stages):
    gc.collect()

    training_features_filename = prefix + constants.features_name + suffix
    training_features_filename = constants.data_filename(training_features_filename, i)
    training_labels_filename = prefix + constants.labels_name + suffix
    training_labels_filename = constants.data_filename(training_labels_filename, i)

    suffix = constants.testing_suffix
    testing_features_filename = prefix + constants.features_name + suffix
    testing_features_filename = constants.data_filename(testing_features_filename, i)
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
    return score

    #
    # Processes running in parallel.
    #    list_measures_entropies = Parallel(n_jobs=constants.n_jobs, verbose=50)(
    #        delayed(get_ams_results)(midx, msize, domain, labels_x_memory, \
    #            training_features, testing_features, training_labels, testing_labels) \
    #                for midx, msize in enumerate(constants.memory_sizes))

    #    for j, measures, entropy, behaviour in list_measures_entropies:
    #        measures_per_size[j, :, :] = measures.T
    #        entropies[j, :] = entropy
    #        behaviours[j, :] = behaviour

    ##########################################################################################

    # Calculate precision and recall


"""

    average_entropy = []
    stdev_entropy = []

    average_precision = []
    stdev_precision = []
    average_recall = []
    stdev_recall = []

    all_precision = []
    all_recall = []

    no_response = []
    no_correct_response = []
    no_correct_chosen = []
    correct_chosen = []
    total_responses = []


        precision = np.zeros((1, n_memories+2), dtype=np.float64)
        recall = np.zeros((1, n_memories+2), dtype=np.float64)

        for j, s in enumerate(constants.memory_sizes):
            precision[j, 0:n_memories] = measures_per_size[j, : , constants.precision_idx]
            precision[j, constants.mean_idx(n_memories)] = measures_per_size[j, : , constants.precision_idx].mean()
            precision[j, constants.std_idx(n_memories)] = measures_per_size[j, : , constants.precision_idx].std()
            recall[j, 0:n_memories] = measures_per_size[j, : , constants.recall_idx]
            recall[j, constants.mean_idx(n_memories)] = measures_per_size[j, : , constants.recall_idx].mean()
            recall[j, constants.std_idx(n_memories)] = measures_per_size[j, : , constants.recall_idx].std()


        ###################################################################3##
        # Measures by memory size

        # Average entropy among al digits.
        average_entropy.append( entropies.mean(axis=1) )
        stdev_entropy.append( entropies.std(axis=1) )

        # Average precision as percentage
        average_precision.append( precision[:, constants.mean_idx(n_memories)] * 100 )
        stdev_precision.append( precision[:, constants.std_idx(n_memories)] * 100 )

        # Average recall as percentage
        average_recall.append( recall[:, constants.mean_idx(n_memories)] * 100 )
        stdev_recall.append( recall[:, constants.std_idx(n_memories)] * 100 )

        all_precision.append(behaviours[:, constants.precision_idx] * 100)
        all_recall.append(behaviours[:, constants.recall_idx] * 100)

        no_response.append(behaviours[:, constants.no_response_idx])
        no_correct_response.append(behaviours[:, constants.no_correct_response_idx])
        no_correct_chosen.append(behaviours[:, constants.no_correct_chosen_idx])
        correct_chosen.append(behaviours[:, constants.correct_response_idx])
        total_responses.append(behaviours[:, constants.mean_responses_idx])


    average_precision = np.array(average_precision)
    stdev_precision = np.array(stdev_precision)
    main_average_precision =[]
    main_stdev_precision = []

    average_recall=np.array(average_recall)
    stdev_recall = np.array(stdev_recall)
    main_average_recall = []
    main_stdev_recall = []

    all_precision = np.array(all_precision)
    main_all_average_precision = []
    main_all_stdev_precision = []

    all_recall = np.array(all_recall)
    main_all_average_recall = []
    main_all_stdev_recall = []

    average_entropy=np.array(average_entropy)
    stdev_entropy=np.array(stdev_entropy)
    main_average_entropy=[]
    main_stdev_entropy=[]

    no_response = np.array(no_response)
    no_correct_response = np.array(no_correct_response)
    no_correct_chosen = np.array(no_correct_chosen)
    correct_chosen = np.array(correct_chosen)
    total_responses = np.array(total_responses)

    main_no_response = []
    main_no_correct_response = []
    main_no_correct_chosen = []
    main_correct_chosen = []
    main_total_responses = []
    main_total_responses_stdev = []


    for i in range(1): # len(constants.memory_sizes)
        main_average_precision.append( average_precision[:,i].mean() )
        main_average_recall.append( average_recall[:,i].mean() )
        main_average_entropy.append( average_entropy[:,i].mean() )

        main_stdev_precision.append( stdev_precision[:,i].mean() )
        main_stdev_recall.append( stdev_recall[:,i].mean() )
        main_stdev_entropy.append( stdev_entropy[:,i].mean() )

        main_all_average_precision.append(all_precision[:, i].mean())
        main_all_stdev_precision.append(all_precision[:, i].std())
        main_all_average_recall.append(all_recall[:, i].mean())
        main_all_stdev_recall.append(all_recall[:, i].std())

        main_no_response.append(no_response[:, i].mean())
        main_no_correct_response.append(no_correct_response[:, i].mean())
        main_no_correct_chosen.append(no_correct_chosen[:, i].mean())
        main_correct_chosen.append(correct_chosen[:, i].mean())
        main_total_responses.append(total_responses[:, i].mean())
        main_total_responses_stdev.append(total_responses[:, i].std())

    main_behaviours = [main_no_response, main_no_correct_response, \
        main_no_correct_chosen, main_correct_chosen, main_total_responses]

    np.savetxt(constants.csv_filename('main_average_precision--{0}'.format(experiment)), \
        main_average_precision, delimiter=',')
    np.savetxt(constants.csv_filename('main_all_average_precision--{0}'.format(experiment)), \
        main_all_average_precision, delimiter=',')
    np.savetxt(constants.csv_filename('main_average_recall--{0}'.format(experiment)), \
        main_average_recall, delimiter=',')
    np.savetxt(constants.csv_filename('main_all_average_recall--{0}'.format(experiment)), \
        main_all_average_recall, delimiter=',')
    np.savetxt(constants.csv_filename('main_average_entropy--{0}'.format(experiment)), \
        main_average_entropy, delimiter=',')

    np.savetxt(constants.csv_filename('main_stdev_precision--{0}'.format(experiment)), \
        main_stdev_precision, delimiter=',')
    np.savetxt(constants.csv_filename('main_all_stdev_precision--{0}'.format(experiment)), \
        main_all_stdev_precision, delimiter=',')
    np.savetxt(constants.csv_filename('main_stdev_recall--{0}'.format(experiment)), \
        main_stdev_recall, delimiter=',')
    np.savetxt(constants.csv_filename('main_all_stdev_recall--{0}'.format(experiment)), \
        main_all_stdev_recall, delimiter=',')
    np.savetxt(constants.csv_filename('main_stdev_entropy--{0}'.format(experiment)), \
        main_stdev_entropy, delimiter=',')

    np.savetxt(constants.csv_filename('main_behaviours--{0}'.format(experiment)), \
        main_behaviours, delimiter=',')

    plot_pre_graph(main_average_precision, main_average_recall, main_average_entropy,\
        main_stdev_precision, main_stdev_recall, main_stdev_entropy, action=experiment)

    plot_pre_graph(main_all_average_precision, main_all_average_recall, \
        main_average_entropy, main_all_stdev_precision, main_all_stdev_recall,\
            main_stdev_entropy, 'overall', action=experiment)

    plot_size_graph(main_total_responses, main_total_responses_stdev, action=experiment)

    plot_behs_graph(main_no_response, main_no_correct_response, main_no_correct_chosen,\
        main_correct_chosen, action=experiment)

    print('Test complete')
"""
# return 1 - accuracy


# EXP 1

# start timer
start_time = time.time()

# 3. Escenario
scenario = Scenario(
    {
        "cs": cs,
        # "output-directory": smac_outdir,
        "run_obj": "quality",
        # "wallclock_limit":2*43200,  #24hrs = 2 *  12 hrs = 12*60*60 secs
        # "n-workers":32,  # Use 32 workers
        "deterministic": "true",  # hace que solo 1 semilla se pruebe en cada ejecucion de la funcion objetivo
        "runcount_limit": 1,
    }
)

smac = BBFacade(
    scenario=scenario, tae_runner=evaluate_memory_config
)  # target_function=evaluate_memory_config)

best_found_config = smac.optimize()
print(best_found_config)


# end timer
end_time = time.time()
log(
    f"Ended SMAC Global Run with n_trials = 1. Total execution time: {end_time - start_time:.2f} seconds",
    fname_txt,
)


# EXP 2

# start timer
start_time = time.time()

# 3. Escenario
scenario = Scenario(
    {
        "cs": cs,
        # "output-directory": smac_outdir,
        "run_obj": "quality",
        "deterministic": "true",  # hace que solo 1 semilla se pruebe en cada ejecucion de la funcion objetivo
        "runcount_limit": 10,
    }
)

smac = BBFacade(
    scenario=scenario, tae_runner=evaluate_memory_config
)  # target_function=evaluate_memory_config)

best_found_config = smac.optimize()
print(best_found_config)

# end timer
end_time = time.time()
log(
    f"Ended SMAC Global Run with n_trials = 10. Total execution time: {end_time - start_time:.2f} seconds",
    fname_txt,
)
