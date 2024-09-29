import random
import numpy as np
import pandas as pd
from scipy.stats import cauchy

class EP_Individual:
    def __init__(self, dimensions, bounds):
        self.position = np.random.uniform(bounds[0], bounds[1], dimensions)
        self.strategy = np.random.uniform(0.1, 0.5, dimensions)  # mutation strengths
        self.fitness = None

def objective_fun1(x):
    return sum(100.0*(x[1:]-x[:-1]**2.0)**2.0 + (1-x[:-1])**2.0)

def objective_fun2(x):
    # Implement Griewanks function
    sum = 0
    prod = 1
    for i in range(len(x)):
        sum += x[i]**2/4000
        prod *= np.cos(x[i]/np.sqrt(i+1))
    return sum - prod + 1

def cauchy_mutate(individual, scale_factor):
    tau = 1 / np.sqrt(2 * len(individual.position))
    tau_prime = 1 / np.sqrt(2 * np.sqrt(len(individual.position)))
    
    individual.strategy *= np.exp(tau_prime * np.random.normal(0, 1) + tau * np.random.normal(0, 1, len(individual.strategy)))
    individual.strategy = np.maximum(individual.strategy, 1e-8)
    
    individual.position += individual.strategy * cauchy.rvs(loc=0, scale=scale_factor, size=len(individual.position))
    return individual

def tournament_selection(population, fitness, tournament_size):
    """
    Perform tournament selection to choose a parent from the population.

    Parameters:
    population (list of numpy.ndarray): The population of individuals.
    fitness (list of float): The fitness values of the individuals in the population.
    tournament_size (int): The number of individuals to be selected for the tournament.

    Returns:
    numpy.ndarray: The selected parent individual.
    """
    selected_indices = np.random.choice(len(population), tournament_size, replace=False)
    selected_fitness = [fitness[i] for i in selected_indices]
    best_index = selected_indices[np.argmin(selected_fitness)]
    return population[best_index], best_index

def print_summary(runs, fitness_obj1, fitness_obj2):
    # First column
    #runs_column = []
    #for i in range(runs):
    #    runs_column.append(f'Run {i+1}')
    
    # Mean and standard deviation
    mean_obj1 = np.mean(fitness_obj1)
    std_obj1 = np.std(fitness_obj1)
    mean_obj2 = np.mean(fitness_obj2)
    std_obj2 = np.std(fitness_obj2)

    # Create a dictionary with the two lists as values
    #data = {'Fun': '', 'Fitness Obj 1': '', 'Fitness Obj 2': ''}
    data = {'': ['Mean'], 'Fitness Obj 1': [mean_obj1], 'Fitness Obj 2': [mean_obj2]}

    # Create a pandas DataFrame from the dictionary
    data_table = pd.DataFrame(data)

    # Create a new DataFrame with the mean and concatenate it with (data_table)
    #mean_row = pd.DataFrame({'': ['Mean'], 'Fitness Obj 1': [mean_obj1], 'Fitness Obj 2': [mean_obj2]})
    #data_table = pd.concat([data_table, mean_row], ignore_index=True)

    # Create a new DataFrame with the stander deviation and concatenate it with (data_table)
    std_row = pd.DataFrame({'': ['STD'], 'Fitness Obj 1': [std_obj1], 'Fitness Obj 2': [std_obj2]})
    data_table = pd.concat([data_table, std_row], ignore_index=True)

    return data_table


def EP(parameters):
    # Unpack parameters
    generations, dim, bounds, mu, lambda_, seed, obj_no = parameters
    scale_factor = 3.0  # Cauchy mutation scale factor

    # Set random seed
    random.seed(seed)

    # INITIALIZATION: Initialize population
    EP_population = [EP_Individual(dim, bounds) for _ in range(mu)]
    #variance = np.random.uniform(low=0, high=1, size=(mu, dim))
    best_individual = None
    best_fitness = None

    # Evolution loop
    for generation in range(generations):
        if generation % 10 == generation: scale_factor = scale_factor / 2
        # Create offspring
        offspring = np.zeros((lambda_, dim))
        offspring_fitness = np.zeros(lambda_)

        # EVALUATION: Evaluate population fitness
        for idx in range(len(EP_population)):
            EP_population[idx].fitness = objective_fun1(EP_population[idx].position) if obj_no == 1 else objective_fun2(EP_population[idx].position)
            #fitness[i] = objective_fun1(population[i]) if obj_no == 1 else objective_fun2(population[i])
        
        # Create offspring through mutation
        offspring = []
        for i in range(lambda_):
            # Extract positions and fitnesses from EP_population
            positions = [individual.position for individual in EP_population]
            fitnesses = [individual.fitness for individual in EP_population]
            # Select parents
            selected_parent, paretn_idx = tournament_selection(positions, fitnesses, 2)

            # Cauchy mutation
            child = EP_Individual(dim, bounds)
            child.position = EP_population[paretn_idx].position.copy()
            child.strategy = EP_population[paretn_idx].strategy.copy()
            child = cauchy_mutate(child, scale_factor)
            child.fitness = objective_fun1(child.position) if obj_no == 1 else objective_fun2(child.position)
            offspring.append(child)

        # Combine population and offspring, and evaluate fitness
        EP_population += offspring
        for idx in range(len(EP_population)):
            EP_population[idx].fitness = objective_fun1(EP_population[idx].position) if obj_no == 1 else objective_fun2(EP_population[idx].position)
        
        # Select mu best individuals
        EP_population = sorted(EP_population, key=lambda x: x.fitness)[:mu]
    
        # TRACK BEST SOLUTION: Update best individual and best fitness
        current_best = EP_population[0]
        if not best_fitness or current_best.fitness < best_fitness:
            best_individual = child.position
            best_fitness = child.fitness

    return best_individual, best_fitness

def ES(parameters):
    # Unpack parameters
    generations, dim, bounds, mu, lambda_, seed, obj_no = parameters

    # Set random seed
    random.seed(seed)

    # Initialize population and variance
    population = np.random.uniform(low=bounds[0], high=bounds[1], size=(mu, dim))
    """ Modify the strategy of create variance """
    sigma = [[4.0] * dim for _ in range(mu)]  
    
    # Generate T and t_prime
    τ = (np.sqrt(2 * np.sqrt(dim))) ** -1
    τ_prime = (np.sqrt(2 * dim)) ** -1
    
    # EVALUATION: Evaluate population fitness
    fitness = np.zeros(mu) # Initialize fitness
    for i in range(mu):
        fitness[i] = objective_fun1(population[i]) if obj_no == 1 else objective_fun2(population[i])

    best_individual = []
    best_fitness = []

    # Evolution loop
    for generation in range(generations):
        #print(f"Generation {generation+1} of {generations}...", end="\r")
        # Create offspring
        offspring = np.zeros((lambda_, dim))
        offspring_variance = np.zeros((lambda_, dim))
        offspring_fitness = np.zeros(lambda_)

        for i in range(lambda_):
            # Select parents
            parents1, parent1_id = tournament_selection(population, fitness, 2)
            parents2, parent2_id = tournament_selection(population, fitness, 2)

            # Recombination: use intermediate recombination
            combine = (parents1 + parents2) / 2
            combine_variance = [(a + b) / 2 for a,b in zip(sigma[parent1_id], sigma[parent2_id])]

            # Mutation variance of offspring
            offspring_variance[i] = combine_variance * np.exp(τ_prime * np.random.normal(0, 1, dim) + τ * np.random.normal(0, 1))

            # calculate diagonal matrix
            diag_matrix_temp = np.diag(offspring_variance[i])
            diag_matrix = np.diag(diag_matrix_temp)

            # Mutation of offspring
            offspring[i] = combine + np.random.normal(0, diag_matrix, dim)

            # Evaluate offspring
            offspring_fitness[i] = objective_fun1(offspring[i]) if obj_no == 1 else objective_fun2(offspring[i])
            
            # Update best individual and best fitness
            if not best_fitness or offspring_fitness[i] < best_fitness:
                best_individual = offspring[i]
                best_fitness = offspring_fitness[i]

        # Combine population and offspring
        population = np.vstack((population, offspring))
        sigma = np.vstack((sigma, offspring_variance))
        fitness = np.concatenate((fitness, offspring_fitness))

        # Select mu best individuals
        best_indices = np.argsort(fitness)[:mu]
        population = population[best_indices]
        fitness = fitness[best_indices]
        sigma = [sigma[i] for i in best_indices]

    return best_individual, best_fitness
 
def main():
    mu = 15  # Number of parents
    lambda_ = 15  # Number of offspring
    dimention_li = [20, 50]  # Number of dimensions
    generations = 50  # Number of generations
    times = 5  # Number of runs
    bounds = [-30, 30]  # Search space

    # generate 30 random seeds with determine incremental value
    seeds = [i+2 for i in range(times)]

    # Iterate over objective functions, once for each objective function
    fitness_obj1_d20 = np.zeros((times,2)) # Store the fitness values for objective function 1
    fitness_obj1_d50 = np.zeros((times,2))
    fitness_obj2_d20 = np.zeros((times,2)) # Store the fitness values for objective function 2
    fitness_obj2_d50 = np.zeros((times,2))

    for i in range(2):
        print()

        # for EP optmization algorithm     
        for run in range(times):
            print('--------------------------------------')
            print(f"Run {run+1}/{times}...", end="\n")


            for dim in dimention_li:
                EP_parameters = [generations, dim, bounds, mu, lambda_, seeds[run], i+1]  # (i) objective function number, 1 = objective_fun1, 2 = objective_fun2
                EP_best_individual, EP_best_fitness = EP(EP_parameters)
                
                if i == 0:
                    if(dim == 20):
                        fitness_obj1_d20[run][0] = EP_best_fitness
                    else:
                        fitness_obj1_d50[run][0] = EP_best_fitness
                else:
                    if(dim == 20):
                        fitness_obj2_d20[run][0] = EP_best_fitness
                    else:
                        fitness_obj2_d50[run][0] = EP_best_fitness

                print(f"EP:  Objective function {i+1}, Dimension {dim}, Run {run+1}/{times}: Best fitness: {EP_best_fitness}")
                print()
                
        # for ES optmization algorithm 
        #for run in range(times):
            #print(f"Run {run+1}/{times}...", end="\r")
            #print(f"ES: Objective function {i+1}, Dimension {dim}, Run {run+1}/{times}...", end="\n")
            for dim in dimention_li:
                ES_parameters = [generations, dim, bounds, mu, lambda_, seeds[run], i+1] # (i) objective function number, 1 = objective_fun1, 2 = objective_fun2
                ES_best_individual, ES_best_fitness = ES(ES_parameters)

                if i == 0:
                    if(dim == 20):
                        fitness_obj1_d20[run][1] = ES_best_fitness
                    else:
                        fitness_obj1_d50[run][1] = ES_best_fitness
                else:
                    if(dim == 20):
                        fitness_obj2_d20[run][1] = ES_best_fitness
                    else:
                        fitness_obj2_d50[run][1] = ES_best_fitness

                print(f"ES:  Objective function {i+1}, Dimension {dim}, Run {run+1}/{times}: Best fitness: {ES_best_fitness}")
                print()
            
            # Store be results in a list
            #if i == 0:
            #    fitness_obj1.append([round(EP_best_fitness), round(ES_best_fitness)])
            #else:
            #    fitness_obj2.append([round(EP_best_fitness,5), round(ES_best_fitness,5)])
    
    # Print the summary
    EP_d20_table = print_summary(times, [item[0] for item in fitness_obj1_d20], [item[0] for item in fitness_obj2_d20])
    EP_d50_table = print_summary(times, [item[0] for item in fitness_obj1_d50], [item[0] for item in fitness_obj2_d50])
    ES_d20_table = print_summary(times, [item[1] for item in fitness_obj1_d20], [item[1] for item in fitness_obj2_d20])
    ES_d50_table = print_summary(times, [item[1] for item in fitness_obj1_d50], [item[1] for item in fitness_obj2_d50])
    
    print("\nEP Dim(20) Results:")
    print(EP_d20_table)
    print("\nEP Dim(50) Results:")
    print(EP_d50_table)
    print("\nES Dim(20) Results:")
    print(ES_d20_table)
    print("\nES Dim(50) Results:")
    print(ES_d50_table)
    #print(ES_data_table)

                



if __name__ == "__main__":
    main()