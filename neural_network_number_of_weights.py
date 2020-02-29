import sys, time
import logging
import pickle

import pandas as pd
from mlrose import (
    NeuralNetwork,
    GeomDecay,
    ArithDecay,
    ExpDecay
)
from sklearn.metrics import (
    balanced_accuracy_score,
    f1_score,
)
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import ParameterGrid

from utils import (
    load_data,
    log_filename
)

STEM = __file__[:-3]
print(STEM)

def get_param_grids():
    lrs = [10**i for i in range(-4,-2)]
    gd_params = {
        'learning_rate': [0.0001],
        'max_iters': [40]
    }
    rhc_params = {
        'learning_rate': [0.1],
        'restarts': [100], #[10] 
        'max_iters': [4000]
    }
    sa_params = {
        'learning_rate': [0.1],
        'schedule': [
            #AirithDecay(),
            ExpDecay(),
            #ExpDecay(exp_const=0.01),
            #ExpDecay(exp_const=0.001),
            #ExpDecay(exp_const=0.002),
            #ExpDecay(exp_const=0.005),
            #ExpDecay(exp_const=0.002, init_temp=.9),
            #ExpDecay(exp_const=0.002, init_temp=.8),
            #ExpDecay(exp_const=0.002, init_temp=.7),
        ],
        'max_iters': [4000]
             
    }
    ga_params = {
        'learning_rate': [0.1], #, 0.01],
        'pop_size': [400], #[200, 400], #[200]
        'mutation_prob': [0.9], #[0.1]
        'clip_max': [100],
        'max_iters': [1000]
    }

    params = {
        'gradient_descent': gd_params,
        'random_hill_climb': rhc_params,
        'simulated_annealing': sa_params,
        'genetic_alg': ga_params
    }
    return params

def run_experiments(problems):
    lfname = log_filename(problems, STEM)
    logging.basicConfig(filename='./nn_logs/' + lfname + '.txt', level=logging.DEBUG)
    algo_runs = {
        'gradient_descent': 3,
        #'random_hill_climb': 3,
        #'simulated_annealing': 3,
        #'genetic_alg': 3
    }
    param_grids = get_param_grids()
    results = {}
    for problem in problems:
        results[problem] = {}
        X_train, X_test, y_train, y_test = load_data(problem)
        ss = StandardScaler()
        X_train_scaled = ss.fit_transform(X_train)
        X_test_scaled = ss.transform(X_test)
        for algo in algo_runs.keys():
            results[problem][algo] = {}
            pg = ParameterGrid(param_grids[algo])
            for i, kwargs in enumerate(pg):
                logging.info('problem ' + problem + ' algo ' + algo + ' i ' + str(i) + str(kwargs))
                results[problem][algo][i] = {}
                print(algo)
                for run in range(algo_runs[algo]):
                    results[problem][algo][i][run] = {}
                    mdl = NeuralNetwork(hidden_nodes = [25, 50, 25], #[5, 10, 5],
                                        activation='relu',
                                        algorithm=algo, #'gradient_descent',
                                        bias=True,
                                        is_classifier=True,
                                        early_stopping=True, #False,
                                        #early_stopping=True,
                                        #clip_max=5,
                                        max_attempts=10,
                                        random_state=0,
                                        curve=True,
                                        **kwargs
                                       )

                    start_time = time.time()
                    logging.info('start time ' + str(time.time()))
                    mdl.fit(X_train_scaled, y_train)
                    end_time = time.time()
                    duration = end_time - start_time
                    logging.info('duration ' + str(duration))
                    y_pred_train = mdl.predict(X_train_scaled)
                    train_acc =  balanced_accuracy_score(y_train, y_pred_train)
                    train_f1 = f1_score(y_train, y_pred_train)
                    results[problem][algo][i][run]['fit_time'] = duration
                    results[problem][algo][i][run]['curve'] = mdl.fitness_curve
                    print("***Number of weights")
                    print(mdl.fitted_weights.shape)
                    results[problem][algo][i][run]['balanced_accuracy'] = train_acc
                    results[problem][algo][i][run]['F1_score'] = train_f1
                    #results[problem][algo][i][run]['train'] = {}
                    #results[problem][algo][i][run]['train']['balanced_accuracy'] = train_acc
                    #results[problem][algo][i][run]['train']['F1_score'] = train_f1
                    #y_pred_test= mdl.predict(X_test_scaled)
                    #test_acc = balanced_accuracy_score(y_test, y_pred_test)
                    #test_f1 = f1_score(y_test, y_pred_test)
                    #results[problem][algo][i][run]['test'] = {}
                    #results[problem][algo][i][run]['test']['balanced_accuracy'] = test_acc
                    #results[problem][algo][i][run]['test']['F1_score'] = test_f1
                    logging.info('train ba' + str(train_acc))
                    #logging.info('test_ba' + str(test_acc))

    with open('./nn_output/' + STEM + '_nn_results_dict.pkl', 'wb') as f:
        pickle.dump(results, f, pickle.HIGHEST_PROTOCOL)


if __name__=="__main__":
    problems = []
    if len(sys.argv) > 1:
        problems = sys.argv[1:]
    run_experiments(problems)
