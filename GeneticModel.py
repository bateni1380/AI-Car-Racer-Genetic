import random
from typing import Callable, List, Any
import numpy as np
class Gene:
    def __init__(self, values: list, objective_val: float):
        self.values = values
        self.objective_val = objective_val
    def copy(self):
        return Gene(self.values.copy(), float(self.objective_val))


class GeneticAlgorithmModel:
    def __init__(self,  metrics=[]):
        self.__crossover_fun: Callable[[Gene, Gene], (Gene, Gene)] = None
        self.__mutation_fun: Callable[[Gene], Gene] = None
        self.__crossover_num = 0
        self.__mutation_num = 0

        self.population: list[Gene] = []
        self.objectives: list[float] = []
        self.epoch = 0
        self.metrics = metrics


    def compile(self, crossover_fun, mutation_fun, initial_population_fun, crossover_coeff, mutation_coeff):
        self.__crossover_fun = crossover_fun
        self.__mutation_fun = mutation_fun
        self.population = initial_population_fun()
        self.objectives = [t.objective_val for t in self.population]
        self.__crossover_num = len(self.population) * crossover_coeff
        self.__mutation_num = len(self.population) * mutation_coeff
        self.__print_on_epoch(self.epoch, self.metrics)

    def __choose_weighted(self, k):
        objectives_sum = sum(self.objectives)
        return np.random.choice(
            self.population, size=k, replace=False,
            p=[u / objectives_sum for u in self.objectives])

    def __choose_best(self, k):
        l1 = self.population.copy()
        l1.sort(key=lambda e: e.objective_val, reverse=True)
        return l1[-k:]

    def extend(self, genes):
        self.population.extend(genes)
        self.objectives.extend([t.objective_val for t in genes])

    def remove(self, gene):
        i1 = self.population.index(gene)
        self.population.pop(i1)
        self.objectives.pop(i1)

    def __print_on_epoch(self, epoch, metrics):
        print('Epoch #' + str(epoch) + ' -->', end='')
        if 'best_objective' in metrics or 'best_objective_val' in metrics:
            t = np.argmax(self.objectives)
            if 'best_objective_val' in metrics:
                print(' best_objective_val=', self.objectives[t], end='\n')
            if 'best_objective' in metrics:
                print('best_objective=', self.population[t].values, end='\n')
        print('\n')

    def step(self):
        
        self.objectives = [t.objective_val for t in self.population]

        crossover_index = 0
        while crossover_index < self.__crossover_num/2:
            parent1, parent2 = self.__choose_weighted(2)
            child1, child2 = self.__crossover_fun(parent1.copy(), parent2.copy())
            self.extend([child1, child2])
            if 'crossovers' in self.metrics:
                print(
                    f'Crossovering:\n  {parent1.values} \n  {parent2.values} \n  childs: \n  {child1.values} \n  {child2.values}')
            crossover_index += 2

        mutation_index = 0
        while mutation_index < self.__mutation_num:
            to_mutate = random.choice(self.population)
            mutated = self.__mutation_fun(to_mutate.copy())
            self.extend([mutated])
            if 'mutates' in self.metrics:
                print(f'Mutating:\n  {to_mutate.values} \n  mutated: \n  {mutated.values}')
            mutation_index += 1

        self.population = self.__choose_best(len(self.population))
        self.objectives = [t.objective_val for t in self.population]
        self.epoch += 1
        self.__print_on_epoch(self.epoch,self. metrics)
        
        

