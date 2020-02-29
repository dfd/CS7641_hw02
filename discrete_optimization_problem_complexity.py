import copy
import logging
import sys
import multiprocessing as mp
import time

from mmlrose.opt_probs import (
    TSPOpt,
    DiscreteOpt
)

from mmlrose.algorithms import (
    random_hill_climb,
    simulated_annealing,
    genetic_alg,
    mimic
)

from mlrose import (
    TravellingSales,
    FourPeaks,
    SixPeaks,
    ContinuousPeaks,
    Knapsack,
    MaxKColor,
    Queens,
    #random_hill_climb,
    #simulated_annealing,
    #genetic_alg,
    #mimic,
    ExpDecay,
    GeomDecay,
    ArithDecay,
    #TSPOpt,
    #DiscreteOpt
)
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from utils import (
    file_join,
    log_filename
)

idx =  pd.IndexSlice

STEM = __file__[:-3]
print(STEM)


def get_problems():
    return {
        'TSP': {
            'class': TravellingSales,
            'opt_prob': TSPOpt,
            'experiments': [
                {'length': 15, #10,  # fixed 2/17 after seeing error
                 'kwargs': {
                    'coords': #[(0, 0), (3, 0), (3, 2), (2, 4), (1, 3),
                              # (7, 9), (2, 5), (4, 1), (2, 3), (0, 7)
                              #]
                     # https://people.sc.fsu.edu/~jburkardt/datasets/tsp/p01_sxy.txt
                     [
                     (-65.0865638413727368,       36.0624693073746769),
                     (-0.1358203773809326,      28.7292896751977480),
                     (-14.6577381710829471,       43.3895496964974043),
                     (-29.0584693142401171,       43.2167287683090606),
                     (-43.0700258454450875,      -14.5548396888330487),
                     (-36.0366489745023770,       21.6135482886620949),
                     (-0.0000000400893815,        0.0000000358808126),
                     (-50.4808382862985496,       -7.3744722432402208),
                     (-50.5859026832315024,       21.5881966132975371),
                     (-21.4983260706612533,       -7.3194159498090388),
                     (-64.7472605264735108,      -21.8981713360336698),
                     (-72.0785319657452987,       -0.1815834632498404),
                     (-28.8732862244731230,       -0.0000008724121069),
                     (-79.2915791686897506,       21.4033307581457670),
                     (-57.5687244704708050,       43.2505562436354225),
                     ]
                 }
                },
            ]
        },
        'Four_Peaks': {
            'class': FourPeaks,
            'opt_prob': DiscreteOpt,
            'experiments': [
                {'length': 30,
                 'kwargs': {
                    't_pct': 0.1
                 },
                },
                {'length': 40,
                 'kwargs': {
                    't_pct': 0.1
                 },
                },
                 {'length': 50,
                 'kwargs': {
                    't_pct': 0.1
                 },
                },
                 {'length': 60,
                 'kwargs': {
                    't_pct': 0.1
                 },
                },
            ]
        },
        'Six_Peaks': {
            'class': SixPeaks,
            'opt_prob': DiscreteOpt,
            'experiments': [
                {'length': 30,
                 'kwargs': {
                    't_pct': 0.1
                 }
                },
            ]
        },
        'Continuous_Peaks': {
            'class': ContinuousPeaks,
            'opt_prob': DiscreteOpt,
            'experiments': [
                {'length': 30,
                 'kwargs': {
                    't_pct': 0.1
                 }
                },
            ]
        },
        'Knapsack': {
            'class': Knapsack,
            'opt_prob': DiscreteOpt,
            'experiments': [
                {'length': 40,
                 'kwargs':
                 {
                    'weights':[
                        1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
                        1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
                        1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
                        1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
                    ],
                    'values': [
                        1, 10, 2, 9, 3, 8, 4, 7, 5, 6,
                        1, 10, 2, 9, 3, 8, 4, 7, 5, 6,
                        1, 10, 2, 9, 3, 8, 4, 7, 5, 6,
                        1, 10, 2, 9, 3, 8, 4, 7, 5, 6,
                    ],
                    'max_weight_pct': 0.25
                 }
                 #{
                 #   'weights':[1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                 #   'values': [1, 10, 2, 9, 3, 8, 4, 7, 5, 6],
                 #   'max_weight_pct': 0.25
                 #}
                },
                #{
                # 'length': 10,
                # 'kwargs':
                # {
                #    'weights':[1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                #    'values': [1, 10, 2, 9, 3, 8, 4, 7, 5, 6],
                #    'max_weight_pct': 0.5
                # },
                #},
                #{
                # 'length': 5,
                # 'kwargs':
                # {
                #    'weights':[1, 2, 3, 4, 5],
                #    'values': [1, 10, 2, 9, 3],
                #    'max_weight_pct': 0.25
                # },
                #},
                #{
                # 'length': 5,
                # 'kwargs':
                # {
                #    'weights':[1, 2, 3, 4, 5],
                #    'values': [1, 10, 2, 9, 3],
                #    'max_weight_pct': 0.5
                # }
                #}
            ]
        },
        'k_Color': {
            'class': MaxKColor,
            'opt_prob': DiscreteOpt,
            'experiments': [
                {'length': 40,
                 'kwargs': {
                    'edges': [(0, 1), (0, 2), (0, 4), (1, 3), (2, 0), (2, 3), (3, 4),
                              (5, 6), (5, 7), (5, 9), (6, 8), (7, 5), (7, 8), (8, 9),
                              (0, 9),
                              (10, 11), (10, 12), (10, 14), (11, 13), (12, 10), (12, 13), (13, 14),
                              (15, 16), (15, 17), (15, 19), (16, 18), (17, 15), (17, 18), (18, 19),
                              (10, 19),
                              (20, 21), (20, 22), (20, 24), (21, 23), (22, 20), (22, 23), (23, 24),
                              (25, 26), (25, 27), (25, 29), (26, 28), (27, 25), (27, 28), (28, 29),
                              (20, 29),
                              (30, 31), (30, 32), (30, 34), (31, 33), (32, 30), (32, 33), (33, 34),
                              (35, 36), (35, 37), (35, 39), (36, 38), (37, 35), (37, 38), (38, 39),
                              (30, 39),
                             ]
                 }
                },
            ]
        },
        'Eight_Queens': {
            'class': Queens,
            'opt_prob': DiscreteOpt,
            'experiments': [
                {'length': 8,
                 'kwargs': {}
                },
            ]
        },
    }


def get_optimizers():
    return {
        'RHC': {
            'function': random_hill_climb,
            'kwargs': [
                {}
            ],
        },
        'SA': {
            'function': simulated_annealing,
            'kwargs': [
                {'schedule': ExpDecay()},
                {'schedule': ExpDecay(exp_const=0.01)},
                {'schedule': ExpDecay(exp_const=0.002)},
                {'schedule': GeomDecay()},
                {'schedule': GeomDecay(decay=.98)},
                {'schedule': GeomDecay(decay=.96)},
                {'schedule': ArithDecay()},
                {'schedule': ArithDecay(decay=0.001)},
                {'schedule': ArithDecay(decay=1/5000)},
                {'schedule': ExpDecay(exp_const=1/5000)},
            ]
        },
        'GA': {
            'function': genetic_alg,
            'kwargs': [
                {'pop_size': 200,
                 'mutation_prob': 0.1                 
                },
                {'pop_size': 200,
                 'mutation_prob': 0.2
                },
                {'pop_size': 200,
                 'mutation_prob': 0.05
                },
                {'pop_size': 800,
                 'mutation_prob': 0.1
                },
            ]
        },
        'MIMIC': {
            'function': mimic,
            'kwargs': [
                {'pop_size': 200,
                 'keep_pct': 0.2,
                 #'fast_mimic': True
                },
                {'pop_size': 200,
                 'keep_pct': 0.4,
                 #'fast_mimic': True
                },
                {'pop_size': 400,
                 'keep_pct': 0.2,
                },
             ]
        
        }
    }

#### add # of runs for each problem X algorithm
def get_runs_per_problem():
    return {
        'TSP': 10,
        'Four_Peaks': 10,
        'Six_Peaks': 10,
        'Continuous_Peaks': 10,
        'Knapsack': 10,
        'k_Color': 10,
        'Eight_Queens': 10,
    }


def get_func_kwargs_per_problem():
    return {
        'Four_Peaks': {
            'GA': 2,
            'MIMIC': 0,
            'SA': 5,
            'RHC': 0
        },
        'Knapsack': {
            'GA': 0,
            'MIMIC': 0,
            'SA': 6,
            'RHC': 0
        },
        'k_Color': {
            'GA': 3,
            'MIMIC': 2,
            'SA': 9,
            'RHC': 0
        },
    
    }

def kwarg_str(kwargs):
    s = ''
    for val in kwargs.values():
        s += str(val)
    return s


def run_opt(params):
    problem_fit = params['problem_fit']
    max_attempts = params['max_attempts']
    #max_iters = params['max_iters']
    run = params['run']
    func_kwargs = params['func_kwargs']
    func = params['func']
    start_time = time.time()
    best_state, best_fitness, curve, eval_curve = (
        func(
            problem_fit,
            max_attempts=max_attempts,
            #max_iters=max_iters,
            curve=True,
            random_state=run,
            **func_kwargs
        )
    )
    end_time = time.time()
    duration = end_time - start_time
    outcome = {
        'best_state': best_state,
        'best_fitness': best_fitness,
        'curve': curve,
        'eval_curve': eval_curve,
        'duration': duration 
    }
    return outcome



def run_experiments(problems):
    lfname = log_filename(problems, STEM)
    logging.basicConfig(filename='./logs/' + lfname + '.txt', level=logging.DEBUG)
    problems_dict = get_problems()
    optimizers = get_optimizers()
    runs_per_problem = get_runs_per_problem()
    func_kwargs_per_problem = get_func_kwargs_per_problem()
    best_state_collection = {}
    best_fitness_collection = {}
    curves = {}
    eval_curves = {}
    times = {}
    iters = {}

    for problem in problems:
        print('problem', problem)
        best_state_collection[problem] = {}
        best_fitness_collection[problem] = {}
        curves[problem] = {}
        eval_curves[problem] = {}
        times[problem] = {}
        iters[problem] = {}
        prob_dict = problems_dict[problem]
        #print('prob_dict', prob_dict)
        for exp_id, experiment in enumerate(prob_dict['experiments']):
            print('exp_id', exp_id)
            kwargs = experiment['kwargs']
            problem_kwargs = kwarg_str(kwargs)
            logging.info('problem ' + problem + ' experiment id ' + str(exp_id) + ' for ' + problem_kwargs)
            best_state_collection[problem][exp_id] = {}
            best_fitness_collection[problem][exp_id] = {}
            curves[problem][exp_id] = {}
            eval_curves[problem][exp_id] = {}
            times[problem][exp_id] = {}
            iters[problem][exp_id] = {}
            fitness = prob_dict['class']
            opt_prob = prob_dict['opt_prob']
            for opt_name, optimizer in optimizers.items():
                best_state_collection[problem][exp_id][opt_name] = {}
                best_fitness_collection[problem][exp_id][opt_name] = {}
                curves[problem][exp_id][opt_name] = {}
                eval_curves[problem][exp_id][opt_name] = {}
                times[problem][exp_id][opt_name] = {}
                iters[problem][exp_id][opt_name] = {}
                func = optimizer['function']
                #for f_kwargs_id, func_kwargs in enumerate(optimizer['kwargs']):
                f_kwargs_id = func_kwargs_per_problem[problem][opt_name]
                func_kwargs = optimizer['kwargs'][f_kwargs_id]
                print('fitness', fitness.__name__)
                print('kwargs', kwargs)
                print('func', func)
                print('func_kwargs', func_kwargs)
                func_kwargs_str = kwarg_str(func_kwargs)
                logging.info('problem ' + problem + 'experiment id ' + str(exp_id) + ' fkwargs_id ' + str(f_kwargs_id) + ' is ' + func_kwargs_str)
                fit_func = fitness(**kwargs)
                problem_fit = opt_prob(experiment['length'], fit_func)
                print('problem_fit', problem_fit)
                best_state_collection[problem][exp_id][opt_name][f_kwargs_id] = {}
                best_fitness_collection[problem][exp_id][opt_name][f_kwargs_id] = {}
                curves[problem][exp_id][opt_name][f_kwargs_id] = {}
                eval_curves[problem][exp_id][opt_name][f_kwargs_id] = {}
                times[problem][exp_id][opt_name][f_kwargs_id] = {}
                iters[problem][exp_id][opt_name][f_kwargs_id] = {}

                param_list = []

                for run in range(runs_per_problem[problem]):
                    params = {
                        'problem_fit': problem_fit,
                        'max_attempts': 200, #1000,
                        'max_iters': 10000,
                        'run': run,
                        'func_kwargs': func_kwargs,
                        'func': func
                    }
                    param_list.append(params)

                logging.info('time ' + str(time.time()))
                with mp.Pool() as pool:
                    outcome = pool.map(run_opt, param_list)
                    pool.close()
                    pool.join()
                    #outcome = run_opt(params)
                    #start_time = time.time()
                    #best_state, best_fitness, curve = (
                    #    func(
                    #        problem_fit,
                    #        max_attempts=1000,
                    #        max_iters=10000,
                    #        curve=True,
                    #        random_state=run,
                    #        **func_kwargs
                    #    )
                    #)
                    #end_time = time.time()
                    #time = end_time - start_time
                    #best_state_collection[problem][problem_kwargs][opt_name][func_kwargs_str][run] = best_state
                    #best_fitness_collection[problem][problem_kwargs][opt_name][func_kwargs_str][run] = best_fitness
                    #curves[problem][problem_kwargs][opt_name][func_kwargs_str][run] = curve
                    #times[problem][problem_kwargs][opt_name][func_kwargs_str][run] = time 
                    #best_state_collection[problem][problem_kwargs][opt_name][func_kwargs_str][run] = outcome['best_state']
                    #best_fitness_collection[problem][problem_kwargs][opt_name][func_kwargs_str][run] = outcome['best_fitness']
                    #curves[problem][problem_kwargs][opt_name][func_kwargs_str][run] = outcome['curve']
                    #times[problem][problem_kwargs][opt_name][func_kwargs_str][run] = outcome['duration']

                for run in range(runs_per_problem[problem]):
                    best_state_collection[problem][exp_id][opt_name][f_kwargs_id][run] = outcome[run]['best_state']
                    best_fitness_collection[problem][exp_id][opt_name][f_kwargs_id][run] = outcome[run]['best_fitness']
                    curves[problem][exp_id][opt_name][f_kwargs_id][run] = outcome[run]['curve']
                    eval_curves[problem][exp_id][opt_name][f_kwargs_id][run] = outcome[run]['eval_curve']
                    times[problem][exp_id][opt_name][f_kwargs_id][run] = outcome[run]['duration']
                    iters[problem][exp_id][opt_name][f_kwargs_id][run] = len(outcome[run]['curve'])

    #best_state_df = pd.DataFrame.from_dict(
    #    {(i, j, k, l, m): best_state_collection[i][j][k][l][m]
    #     for i in best_state_collection.keys()
    #     for j in best_state_collection[i].keys()
    #     for k in best_state_collection[i][j].keys()
    #     for l in best_state_collection[i][j][k].keys()
    #     for m in best_state_collection[i][j][k][l].keys()
    #     }, orient='index'
    #    )

    best_fitness_df = pd.DataFrame.from_dict(
        {(i, j, k, l, m): best_fitness_collection[i][j][k][l][m]
         for i in best_fitness_collection.keys()
         for j in best_fitness_collection[i].keys()
         for k in best_fitness_collection[i][j].keys()
         for l in best_fitness_collection[i][j][k].keys()
         for m in best_fitness_collection[i][j][k][l].keys()
         }, orient='index'
        )
    best_fitness_df.columns = ['fitness']
    best_fitness_df.index = pd.MultiIndex.from_tuples(best_fitness_df.index,
                                                names=('problem', 'experiment',
                                                       'algo', 'params', 'run'))
    #best_fitness_df.to_pickle('./output/best_fitness_df.pkl')
    for problem in problems:
        for exp_id, experiment in enumerate(prob_dict['experiments']):
            
            fig, ax = plt.subplots(figsize=(6, 4))
            plt.subplots_adjust(bottom=.2)
            #f, ax = plt.subplots(1, 1)
            #plt.subplots_adjust(bottom=.3)
            summary = best_fitness_df.loc[idx[(problem, exp_id)], :].unstack().droplevel(1)
            #summary = best_fitness_df.loc[idx[(problem)], :].unstack()
            summary.plot(kind='bar', rot=45, ax=ax, legend=False, color="C0", edgecolor='black')
            plt.xlabel('Each Run Grouped By Algorithm')
            plt.ylabel('fitness')
            plt.suptitle('Fitness Result on ' + problem + '\nfor ' + str(runs_per_problem[problem]) + ' runs of each Algorithm ')
            #plt.legend(loc='lower right')
            plt.savefig('./plots/' + STEM + '_end_fitness_by_run_' + file_join([problem, exp_id]) + '.png')
            plt.clf()
 
    curves_df = pd.DataFrame.from_dict(
        {(i, j, k, l, m): curves[i][j][k][l][m]
         for i in curves.keys()
         for j in curves[i].keys()
         for k in curves[i][j].keys()
         for l in curves[i][j][k].keys()
         for m in curves[i][j][k][l].keys()
         }, orient='index'
        )
    curves_df = curves_df.ffill(axis=1)
    curves_df.index = pd.MultiIndex.from_tuples(curves_df.index,
                                                names=('problem', 'experiment',
                                                       'algo', 'params', 'run'))
    avg_curves = curves_df.mean(level=list(range(4)))
    for problem in problems:
        for exp_id, experiment in enumerate(prob_dict['experiments']):
            
            fig, ax = plt.subplots(figsize=(6, 4))
            plt.subplots_adjust(bottom=.15)
            #plot_df = avg_curves.loc[idx[problem, exp_id, : , ], :]
            plot_df = avg_curves.loc[idx[problem, exp_id, : , ], :].droplevel(level=[0, 1, 3])
            plot_df.T.plot(ax=ax)
            plt.xlabel('iteration')
            plt.ylabel('fitness')
            plt.suptitle('Fitness on ' + problem + ' by iteration')
            plt.legend(loc='lower right')
            plt.savefig('./plots/' + STEM + '_fitness_curves_' + file_join([problem, exp_id]) + '.png')
            plt.clf()
    #curves_df.to_pickle('./output/curves_df.pkl')
    #print(curves_df.unstack(-1))

    eval_curves_df = pd.DataFrame.from_dict(
        {(i, j, k, l, m): eval_curves[i][j][k][l][m]
         for i in eval_curves.keys()
         for j in eval_curves[i].keys()
         for k in eval_curves[i][j].keys()
         for l in eval_curves[i][j][k].keys()
         for m in eval_curves[i][j][k][l].keys()
         }, orient='index'
        )
    eval_curves_df.index = pd.MultiIndex.from_tuples(eval_curves_df.index,
                                                names=('problem', 'experiment',
                                                       'algo', 'params', 'run'))
    avg_eval_curves = eval_curves_df.mean(level=list(range(4)))
    #print(avg_eval_curves)
    for problem in problems:
        for exp_id, experiment in enumerate(prob_dict['experiments']):
            
            fig, ax = plt.subplots(figsize=(6, 4))
            plt.subplots_adjust(bottom=.15)
            plot_df = avg_eval_curves.loc[idx[problem, exp_id, : , ], :].droplevel(level=[0, 1, 3])
            plot_df.T.plot(ax=ax)
            plt.xlabel('iteration')
            plt.ylabel('evaluations')
            plt.yscale('log')
            plt.suptitle('Evaluations on ' + problem + ' by Iteration')
            plt.legend(loc='lower right')
            plt.savefig('./plots/' + STEM + '_eval_curves_' + file_join([problem, exp_id]) + '.png')
            plt.clf()
    colors = ['C0', 'C1', 'C2', 'C3']

    #print('curves')
    #print(curves_df.head())
    #print('evals')
    #print(eval_curves_df.head())

    for problem in problems:
        for exp_id, experiment in enumerate(prob_dict['experiments']):
            fig, ax = plt.subplots(figsize=(6, 4))
            plt.subplots_adjust(bottom=.15)
            #for i, algo in enumerate(algos): # change
            for i, (opt_name, optimizer) in enumerate(optimizers.items()):
                color = colors[i]
                f_kwargs_id = func_kwargs_per_problem[problem][opt_name]
                for run in range(runs_per_problem[problem]):
                    fitness = curves_df.loc[idx[problem, exp_id, opt_name, f_kwargs_id, run], :]
                    #print('fitness')
                    #print(fitness)
                    evals = eval_curves_df.loc[idx[problem, exp_id, opt_name, f_kwargs_id, run], :]
                    #print('evals')
                    #print(evals)
                    #plt.plot(evals, fitness)
                    #plt.show()
                    if run == 0:
                        ax.plot(evals, fitness, color=color, linestyle=':', label=opt_name)
                    else:
                        ax.plot(evals, fitness, color=color, linestyle=':', label='_nolegend_')
                    ax.set_xlabel('evaluations')
                    ax.set_ylabel('fitness')
                    ax.set_xscale('log')
                    ax.set_title('Fitness by Evaluation for ' + problem)
            ax.legend()        
            plt.savefig('./plots/' + STEM + '_fitness_by_eval_' + file_join([problem, exp_id]) + '.png')
            plt.clf()

    times_df = pd.DataFrame.from_dict(
        {(i, j, k, l, m): times[i][j][k][l][m]
         for i in times.keys()
         for j in times[i].keys()
         for k in times[i][j].keys()
         for l in times[i][j][k].keys()
         for m in times[i][j][k][l].keys()
         }, orient='index'
        )
    times_df.columns = ['time']
    times_df.index = pd.MultiIndex.from_tuples(times_df.index,
                                                names=('problem', 'experiment',
                                                       'algo', 'params', 'run'))
    iters_df = pd.DataFrame.from_dict(
        {(i, j, k, l, m): iters[i][j][k][l][m]
         for i in iters.keys()
         for j in iters[i].keys()
         for k in iters[i][j].keys()
         for l in iters[i][j][k].keys()
         for m in iters[i][j][k][l].keys()
         }, orient='index'
        )
    iters_df.columns = ['iterations']
    iters_df.index = pd.MultiIndex.from_tuples(iters_df.index,
                                                names=('problem', 'experiment',
                                                       'algo', 'params', 'run'))
    combined = pd.merge(left=times_df,
                        right=iters_df,
                        how='left',
                        left_index=True,
                        right_index=True
                        )

    #combined.to_pickle('./output/combined.pkl')
    avgtimes = {}
    for problem in problems:
        avgtimes[problem] = {}
        for exp_id, experiment in enumerate(prob_dict['experiments']):
            avgtimes[problem][exp_id] = {}
            fig, ax = plt.subplots(figsize=(6, 4))
            plt.subplots_adjust(bottom=.15)
            #fig, ax = plt.subplots()
            plot_df = combined.loc[idx[(problem, exp_id)], :]
            #print(plot_df)
            for opt_name, optimizer in optimizers.items():
                df = plot_df.loc[idx[opt_name], :]
                avgtime = df['time'].sum()/df['iterations'].sum()
                avgtimes[problem][exp_id][opt_name] = avgtime

                ax.scatter(df['iterations'], df['time'], label=f'{opt_name}: {avgtime:.2e} s/iter')
                ax.set_yscale('log')
            plt.xlabel('iterations')
            plt.ylabel('completion time (seconds)')
            plt.suptitle('Time vs Iterations for each algorithm on ' + problem)
            plt.legend()
            plt.savefig('./plots/' + STEM + '_time_iterations_' + file_join([problem, exp_id]) + '.png')


    combined = pd.merge(left=times_df,
                        right=best_fitness_df,
                        how='left',
                        left_index=True,
                        right_index=True
                        )

    #combined.to_pickle('./output/combined.pkl')
    for problem in problems:
        for exp_id, experiment in enumerate(prob_dict['experiments']):
            fig, ax = plt.subplots(figsize=(6, 4))
            plt.subplots_adjust(bottom=.15)
            #fig, ax = plt.subplots()
            plot_df = combined.loc[idx[(problem, exp_id)], :]
            #print(plot_df)
            for opt_name, optimizer in optimizers.items():
                avgtime = avgtimes[problem][exp_id][opt_name]
                df = plot_df.loc[idx[opt_name], :]
                ax.scatter(df['time'], df['fitness'], label=f'{opt_name}: {avgtime:.2e} s/iter')
                ax.set_xscale('log')
            plt.xlabel('time (seconds)')
            plt.ylabel('Fitness')
            plt.suptitle('Fitness vs Time for each algorithm on ' + problem)
            plt.legend()
            plt.savefig('./plots/' + STEM + '_fitness_time_' + file_join([problem, exp_id]) + '.png')




    #best_state_df.to_csv('./results/' + lfname + '_best_states.csv')
    best_fitness_df.to_csv('./results/' + lfname + '_best_fitness.csv')
    curves_df.to_csv('./results/' + lfname + '_curves.csv')
    eval_curves_df.to_csv('./results/' + lfname + '_eval_curves.csv')
    times_df.to_csv('./results/' + lfname + '_times.csv')
    iters_df.to_csv('./results/' + lfname + '_iterations.csv')


    #logging.info(str(best_state_collection))
    logging.info(str(best_fitness_collection))
    logging.info(str(curves))
    logging.info(str(times))
                    







if __name__=="__main__":
    problems = []
    if len(sys.argv) > 1:
        problems = sys.argv[1:]
    run_experiments(problems)
