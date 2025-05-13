# core/optimizer.py
from algorithms.heuristics.spt import SPTHeuristic
from algorithms.metaheuristics.simulated_annealing import SimulatedAnnealing  # when ready
from algorithms.heuristics.glouton import GreedyHeuristic
from algorithms.heuristics.lpt import LPTHeuristic
from algorithms.heuristics.IdleFillingHeuristic import IdleFillingHeuristic
from algorithms.metaheuristics.tabu_search import TabuSearch
import copy, math, random
from models.machine import Machine
from algorithms.metaheuristics.genetic_algorithm import GeneticAlgorithm

def optimize(instance, class_id=None, strategy='SPT', **params):
    if strategy.upper() == 'SPT':
        _, _, makespan = SPTHeuristic().optimize(instance)  # Extract makespan
        return makespan
    elif strategy.upper() == 'SA':
        return SimulatedAnnealing(class_id=class_id, **params).optimize(instance)
    elif strategy == "Greedy":
        _, _, makespan = GreedyHeuristic().optimize(instance)  # Extract makespan
        return makespan
    elif strategy == "LPT":
        _, _, makespan = LPTHeuristic().optimize(instance)  # Extract makespan
        return makespan
    elif strategy == "IdleFilling":
        _, _, makespan = IdleFillingHeuristic().optimize(instance)  # Extract makespan
        return makespan
    elif strategy == "TS":
        # Extract tabu search specific parameters
        tabu_tenure = params.get('tabu_tenure', 20)
        max_iterations = params.get('max_iterations', 500)
        max_iterations_no_improvement = params.get('max_iterations_no_improvement', 100)
        num_neighbors = params.get('num_neighbors', 20)
        enable_hypertuning = params.get('enable_hypertuning', True)
        
        # Create and run the tabu search optimizer
        optimizer = TabuSearch(
            tabu_tenure=tabu_tenure,
            max_iterations=max_iterations,
            max_iterations_no_improvement=max_iterations_no_improvement,
            num_neighbors=num_neighbors,
            class_id=class_id,
            enable_hypertuning=enable_hypertuning
        )
        return optimizer.optimize(instance)
    elif strategy=='GA':
        return GeneticAlgorithm(**params).optimize(instance)
    else:
        raise ValueError(f"Unknown strategy: {strategy}")

