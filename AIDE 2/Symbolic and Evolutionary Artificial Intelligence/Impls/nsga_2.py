from abc import ABC, abstractmethod
from typing import override
import numpy as np
from tqdm.auto import tqdm
import matplotlib.pyplot as plt

def dominates(p1: np.ndarray, p2: np.ndarray) -> bool:
    """
    Check if solution p1 dominates solution p2.
    A solution p1 dominates p2 if it is no worse in all objectives and better in at least one.
    """
    return np.all(p1 <= p2) and np.any(p1 < p2)

def crowding_distance_sort(front: list[np.ndarray], front_indices: list[int]) -> list[int]:
    """
    Sorts a front of solutions based on crowding distance.
    The crowding distance is a measure of how close solutions are to each other in the objective space.
    Solutions with higher crowding distance are preferred.
    """
    if len(front) == 0:
        return []
    distances = np.zeros(len(front))
    num_objectives = front[0].shape[0]

    for i in range(num_objectives):
        sorted_indices, sorted_front = zip(*sorted(enumerate(front), key=lambda x: x[1][i]))
        distances[sorted_indices[0]] = float('inf')  # Boundary solutions
        distances[sorted_indices[-1]] = float('inf')  # Boundary solutions
        range_i = sorted_front[-1][i] - sorted_front[0][i] + 1e-10  # Avoid division by zero
        for j in range(1, len(sorted_front) - 1):
            distances[sorted_indices[j]] += (sorted_front[j + 1][i] - sorted_front[j - 1][i]) / range_i

    return [front_indices[i] for i in np.argsort(-distances)]
    

def fast_non_dominated_sort(population: list[np.ndarray]):
    s = [[] for _ in range(len(population))]
    n = np.zeros(len(population), dtype=int)
    fronts = [[]] # indices into population
    ranks = [0 for _ in range(len(population))] # indices into fronts
    for i, p in enumerate(population):
        for j, q in enumerate(population):
            if dominates(p, q):
                s[i].append(j)
            elif dominates(q, p):
                n[i] += 1
        if n[i] == 0:
            ranks[i] = 0
            fronts[0].append(i)
    while True:
        next_front = []
        for i in fronts[-1]:
            for j in s[i]:
                n[j] -= 1
                if n[j] == 0:
                    ranks[j] = len(fronts)
                    next_front.append(j)
        if len(next_front) == 0:
            break
        fronts.append(next_front)
    return fronts, ranks

class Individual(ABC):
    @abstractmethod
    def evaluate(self) -> np.ndarray:
        """
        Evaluate the individual and return its objective values.
        This method should be implemented by subclasses.
        """
        pass

    @abstractmethod
    def mate(self, other: 'Individual') -> 'Individual':
        """
        Create a new individual by mating this individual with another.
        This method should be implemented by subclasses.
        """
        pass

    @staticmethod
    @abstractmethod
    def random_init() -> 'Individual':
        """
        Create a new individual with random values.
        This method should be implemented by subclasses.
        """
        pass

def binary_tournament_selection(population: list[np.ndarray]) -> int:
    i, j = np.random.choice(len(population), size=2, replace=False)
    if dominates(population[i], population[j]):
        return i
    elif dominates(population[j], population[i]):
        return j
    else:
        # If neither dominates the other, select randomly
        return np.random.choice([i, j])

def reproduce(population: list[Individual], objectives: list[np.ndarray]) -> Individual:
    father = population[binary_tournament_selection(objectives)]
    mother = population[binary_tournament_selection(objectives)]
    child = father.mate(mother)
    return child

def lexicographic_sort(front_indices: list[int], objectives: list[np.ndarray]) -> list[int]:
    """
    Sorts the population based on lexicographic order of their objective values.
    This is a simple sorting method that can be used to determine the best individuals.
    """
    sorted_indices = sorted(range(len(objectives)), key=lambda i: tuple(objectives[i]))
    return [front_indices[i] for i in sorted_indices]

def nsga_2(individual_class: type[Individual], population_size: int, generations: int):
    """
    NSGA-II algorithm implementation.
    :param individual_class: The class of individuals to be used in the population.
    :param population_size: The size of the population.
    :param generations: The number of generations to run the algorithm.
    """
    # Initialize population
    population = [individual_class.random_init() for _ in range(population_size)]
    objectives = [ind.evaluate() for ind in population]
    loading_bar = tqdm(range(generations), desc="Running NSGA-II", unit="gen", total=generations)
    for generation in loading_bar:
        new_population = [
            reproduce(population, objectives) for _ in range(population_size)
        ]
        new_objectives = [ind.evaluate() for ind in new_population]
        combined_population = population + new_population
        combined_objectives = objectives + new_objectives
        fronts, ranks = fast_non_dominated_sort(combined_objectives)
        loading_bar.set_postfix({
            "Fronts": len(fronts),})
        next_population = []
        next_objectives = []
        for fi in range(len(fronts)):
            if len(next_population) + len(fronts[fi]) > population_size:
                fronts[fi] = crowding_distance_sort([combined_objectives[i] for i in fronts[fi]], fronts[fi])
                fronts[fi] = fronts[fi][:population_size - len(next_population)]
            next_population.extend([combined_population[i] for i in fronts[fi]])
            next_objectives.extend([combined_objectives[i] for i in fronts[fi]])
            assert len(next_population) <= population_size, "Population exceeded size limit"
            if len(next_population) == population_size:
                break
        population = next_population
        objectives = next_objectives
    fronts, ranks = fast_non_dominated_sort(objectives)
    for i, front in enumerate(fronts):
        fronts[i] = lexicographic_sort(front, [objectives[j] for j in front])
    return population, objectives, fronts, ranks


if __name__ == "__main__":
    class ExampleIndividual(Individual):
        def __init__(self, values: np.ndarray):
            self.values = values
        
        @override
        def evaluate(self) -> np.ndarray:
            return np.array([
                1.0 - np.exp(-np.sum((self.values - 0.33**0.5)**2)),
                1.0 - np.exp(-np.sum((self.values + 0.33**0.5)**2)),
            ])
        
        @override
        def mate(self, other: 'ExampleIndividual') -> 'ExampleIndividual':
            # crossover
            values = zip(self.values, other.values)
            new_values = np.array([a if np.random.rand() < 0.5 else b for a, b in values])

            # mutation
            if np.random.rand() < 0.5:  # mutation probability
                new_values += np.random.normal(0, 0.5, new_values.shape)
            return ExampleIndividual(new_values)

        @override
        def random_init() -> 'ExampleIndividual':
            values = np.random.rand(3)
            return ExampleIndividual(values)
        
    population, objectives, fronts, ranks = \
        nsga_2(ExampleIndividual, population_size=128, generations=5)
    
    print("Final Population:")
    for idx, (ind, obj, rank) in enumerate(zip(population, objectives, ranks)):
        print(f"Individual[{idx}]: {ind.values}, Objectives: {obj}, Rank: {rank}")
        
    # Plotting the results
    plt.figure(figsize=(10, 6))
    plt.suptitle('NSGA-II Results')
    plt.subplot(1, 2, 1)
    for fi, front in enumerate(fronts):
        front_objectives = [objectives[i] for i in front]
        front_objectives = np.array(front_objectives)
        plt.plot(front_objectives[:, 0], front_objectives[:, 1], '--o', alpha=0.5, label=f"Front {fi}")
    plt.title('Objective Space')
    plt.xlabel('Objective 1')
    plt.ylabel('Objective 2')
    plt.legend()
    plt.grid()
    plt.subplot(1, 2, 2, projection='3d')
    for fi, front in enumerate(fronts):
        front_values = [population[i].values for i in front]
        front_values = np.array(front_values)
        plt.plot(front_values[:, 0], front_values[:, 1], front_values[:, 2], 'o', alpha=0.5, label=f"Front {fi}")
    plt.title('Search Space')
    plt.legend()
    plt.grid()
    plt.show()