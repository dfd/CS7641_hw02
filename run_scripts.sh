source activate CS7641_A2

# generate data
python ./generate_data.py

# neural network scripts
./run_nn5.sh
./run_nn5_analysis.sh
./run_nn8.sh
./run_nn8_analysis.sh
./run_nn9.sh
./run_nn9_analysis.sh
./run_nn_final.sh
./run_nn_final_analysis.sh

# discrete problems
./run_discrete9.sh
./run_discrete10.sh
./run_discrete_hps.sh
./run_discrete_problem_complexity.sh

