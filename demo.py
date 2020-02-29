import mlrose
import numpy as np

fitness = mlrose.Queens()

def queens_max(state):

    # Initialize counter
    fitness_cnt = 0

    # for all pairs of queens
    for i in range(len(state) - 1):
        for j in range(i + 1, len(state)):

            # Check for horizontal, diagonal-up and diagonal-down attacks
            if (state[j] != state[i]) \
                    and (state[j] != state[i] + (j - i)) \
                    and (state[j] != state[i] - (j - i)):

                # if no attacks, then increment counter
                fitness_cnt += 1

    return fitness_cnt


def queens_problem(max_attempts, max_iters):
    fitness_cust = mlrose.CustomFitness(queens_max)

    problem = mlrose.DiscreteOpt(length = 8, fitness_fn=fitness, maximize = False, max_val=8)
    #problem = mlrose.DiscreteOpt(length=8, fitness_fn=queens_max, maximize=True, max_val=8)

    # Define decay schedule
    schedule = mlrose.ExpDecay()

    # Define initial state
    init_state = np.array([0, 1, 2, 3, 4, 5, 6, 7])

    # Solve problem using simulated annealing
    best_state, best_fitness = mlrose.simulated_annealing(problem, schedule=schedule,
                                                          max_attempts=max_attempts, max_iters=max_iters,
                                                          init_state=init_state, random_state=1)
    print(best_state)
    print(best_fitness)


def tsp_problem():
    coords_list = [(1, 1), (4, 2), (5, 2), (6, 4), (4, 4), (3, 6), (1, 5), (2, 3)]
    fitness_coords = mlrose.TravellingSales(coords=coords_list)

    # Define optimization problem object
    problem_fit = mlrose.TSPOpt(length=8, fitness_fn=fitness_coords, maximize=True)

    best_state, best_fitness = mlrose.genetic_alg(problem_fit, random_state=2)

    print(best_state)
    print(best_fitness)

    best_state, best_fitness = mlrose.genetic_alg(problem_fit, mutation_prob=0.2,
                                                  max_attempts=100, random_state=2)


if __name__=="__main__":
    print('queens 10,100')
    queens_problem(10, 100)
    print('queens 100,1000')
    queens_problem(100, 1000)
    print('tsp')
    tsp_problem()




