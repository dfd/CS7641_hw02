import pickle

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

idx =  pd.IndexSlice

#STEM = __file__[-3]
#print(STEM)

def run_analysis():
    with open('./nn_output/neural_network_final_nn_results_dict.pkl', 'rb') as f:
        results = pickle.load(f)

    #print(results)
    raw_data = results['two_features']

    # get all curves into a list of np.arrays
    #curves_list = []
    curves_df = pd.DataFrame.from_dict(
        {(i, j, k): raw_data[i][j][k]['curve']
                    for i in raw_data.keys()
                    for j in raw_data[i].keys()
                    for k in raw_data[i][j].keys()
            }, orient='index'
    )
    curves_df.index = pd.MultiIndex.from_tuples(curves_df.index,
                                                names=('algorithm', 'hyperparameters', 'run'))
                    
    avg_curves = curves_df.mean(level=0)

    fig, ax = plt.subplots(figsize=(6,4)) 
    print(avg_curves)
    avg_curves.T.plot(ax=ax, color=['C0', 'C1', 'C2', 'C3'])
    curves_df.T.plot(ax=ax, color=['C0']*3 + ['C1']*3 + ['C2']*3 + ['C3']*3, style=':', legend=False)
    ax.set_ylim(-8, 0)
    ax.set_xlabel('iteration')
    ax.set_ylabel('Negative Log Loss')
    plt.title('Fitness curves for means and individual runs per Algorithm')
    plt.savefig('./nn_plots/nn_final_total_fitness_curve.png')

    fig, ax = plt.subplots(figsize=(6,4)) 
    #print(avg_curves)
    avg_curves.T.plot(ax=ax, color=['C0', 'C1', 'C2', 'C3'])
    curves_df.T.plot(ax=ax, color=['C0']*3 + ['C1']*3 + ['C2']*3 + ['C3']*3, style=':', legend=False)
    ax.set_ylim(-3, 0)
    ax.set_xlim(0, 2000)
    ax.set_xlabel('iteration')
    ax.set_ylabel('Negative Log Loss')
    plt.title('Zoom In: Fitness curves for means and individual runs per Algorithm')
    plt.savefig('./nn_plots/nn_final_zoom_fitness_curve.png')

    fit_time_df = pd.DataFrame
    
    ba_df = pd.DataFrame.from_dict(
        {(i, j, k): raw_data[i][j][k]['balanced_accuracy']
                    for i in raw_data.keys()
                    for j in raw_data[i].keys()
                    for k in raw_data[i][j].keys()
            }, orient='index'
    )
    ba_df.index = pd.MultiIndex.from_tuples(curves_df.index,
                                                names=('algorithm', 'hyperparameters', 'run'))

    ft_df = pd.DataFrame.from_dict(
        {(i, j, k): raw_data[i][j][k]['fit_time']
                    for i in raw_data.keys()
                    for j in raw_data[i].keys()
                    for k in raw_data[i][j].keys()
            }, orient='index'
    )
    ft_df.index = pd.MultiIndex.from_tuples(curves_df.index,
                                                names=('algorithm', 'hyperparameters', 'run'))

    combined = pd.merge(left=ba_df,
                        right=ft_df,
                        how='left',
                        left_index=True,
                        right_index=True
                        )

    combined.columns = ['balanced_accuracy', 'fit_time']


    fig, ax = plt.subplots(figsize=(6, 4))
    plt.subplots_adjust(bottom=.15)
    for algo in raw_data.keys():   
        df = combined.loc[idx[algo], :]
        ax.scatter(df['fit_time'], df['balanced_accuracy'], label=algo)

    ax.set_xscale('log')
    plt.xlabel('fit time (seconds)')
    plt.ylabel('Balanced Accuracy')
    plt.suptitle('Fitness vs Time for each algorithm with Neural Networks')
    plt.legend()
    plt.plot()
    plt.savefig('./nn_plots/nn_final_time_ba.png')
    #plt.show()

    f1_df = pd.DataFrame.from_dict(
        {(i, j, k): raw_data[i][j][k]['F1_score']
                    for i in raw_data.keys()
                    for j in raw_data[i].keys()
                    for k in raw_data[i][j].keys()
            }, orient='index'
    )
    f1_df.index = pd.MultiIndex.from_tuples(curves_df.index,
                                                names=('algorithm', 'hyperparameters', 'run'))

    combined = pd.merge(left=f1_df,
                        right=ft_df,
                        how='left',
                        left_index=True,
                        right_index=True
                        )

    combined.columns = ['F1_score', 'fit_time']


    fig, ax = plt.subplots(figsize=(6, 4))
    plt.subplots_adjust(bottom=.15)
    for algo in raw_data.keys():   
        df = combined.loc[idx[algo], :]
        ax.scatter(df['fit_time'], df['F1_score'], label=algo)

    ax.set_xscale('log')
    plt.xlabel('fit time (seconds)')
    plt.ylabel('F1 Score')
    plt.suptitle('Fitness vs Time for each algorithm with Neural Networks')
    plt.legend()
    plt.plot()
    plt.savefig('./nn_plots/nn_final_time_f1.png')
    plt.show()
 
if __name__=="__main__":
    run_analysis()
