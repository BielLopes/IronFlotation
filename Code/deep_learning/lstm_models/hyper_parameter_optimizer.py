from mealpy.swarm_based import GWO
import numpy as np

def fitness_function(solution):
    return np.sum(solution**2)

problem = {
    "fit_func": fitness_function,
    "lb": [-100, ],
    "ub": [100, ],
    "minmax": "min",
    "log_to": None,
    "save_population": False,
}

term_dict1 = {
   "mode": "FE",
   "quantity": 10    # 100000 number of function evaluation
}


## Run the algorithm
model = GWO.OriginalGWO(epoch=100, pop_size=50, pr=0.03)
best_position, best_fitness = model.solve(problem, termination=term_dict1)
print(f"Best solution: {best_position}, Best fitness: {best_fitness}")