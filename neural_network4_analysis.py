import pickle

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def run_analysis():
    with open('./nn_output/neural_network4_nn_results_dict.pkl', 'rb') as f:
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
        
                      

        print(curves_df)
        curves_df.T.plot()
        plt.show()


        # put rest of data into a DataFrame 

if __name__=="__main__":
    run_analysis()
