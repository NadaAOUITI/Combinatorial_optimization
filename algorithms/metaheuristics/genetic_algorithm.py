from .base import Metaheuristic
import copy
import random
import numpy as np
from tqdm import tqdm
from algorithms.heuristics.IdleFillingHeuristic import IdleFillingHeuristic
from algorithms.metaheuristics.tabu_search import TabuSearch
from algorithms.metaheuristics.simulated_annealing import SimulatedAnnealing
from algorithms.metaheuristics.neighborsgeneration import calculate_makespan, schedule_to_machines, generate_diverse_neighbors
import matplotlib.pyplot as plt
import time
import json
from itertools import product
from concurrent.futures import ThreadPoolExecutor

class GeneticAlgorithm(Metaheuristic):
    def __init__(self, population_size=200, max_generations=200, crossover_rate=0.85, mutation_rate=0.25,
                 elite_size=10, class_id=1, enable_hypertuning=True, time_limit=60):
        self.population_size = population_size
        self.max_generations = max_generations
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.elite_size = elite_size
        self.class_id = class_id
        self.enable_hypertuning = enable_hypertuning
        self.time_limit = time_limit  # Time limit in seconds
        
        # For tracking makespan over generations
        self.generation_history = []
        self.makespan_history = []
        self.best_makespan_per_gen = []  # Track best makespan per generation
        self.best_recipe = None  # Store the best parameter combination

    def optimize(self, instance):
        """
        Optimize the schedule using a Genetic Algorithm.
        """
        # Start timing
        start_time = time.time()

        # Apply hypertuning only if enabled
        if self.enable_hypertuning:
            self.hypertune_parameters(instance)

        # Generate initial solution using TabuSearch with minimal iterations
        tabu = TabuSearch(max_iterations=50, max_iterations_no_improvement=25, enable_hypertuning=True)
        _, tabu_schedule, tabu_makespan = tabu.optimize(instance)

        print(f"Initial TabuSearch solution makespan: {tabu_makespan}")

        # Initialize population based on the tabu solution
        population = self.initialize_population(instance, tabu_schedule)

        # Evaluate the initial population
        fitness_scores = self.evaluate_population(population)

        # Find the initial best solution
        best_idx = fitness_scores.index(max(fitness_scores))
        best_schedule = copy.deepcopy(population[best_idx])
        best_makespan = calculate_makespan(best_schedule)

        # Progress bar
        progress_bar = tqdm(desc="Genetic Algorithm", total=self.max_generations)

        # Track generations without improvement for early stopping
        generations_no_improvement = 0

        # Main GA loop
        for generation in range(self.max_generations):
            # Check if time limit exceeded
            if time.time() - start_time > self.time_limit:
                print(f"\nTime limit of {self.time_limit} seconds reached. Stopping...")
                break

            # Elitism - keep the best solutions
            elite = self.select_elite(population, fitness_scores)

            # Create new population through selection, crossover, and mutation
            new_population = elite.copy()

            # Fill the rest of the population
            while len(new_population) < self.population_size:
                # Selection
                parent1 = self.roulette_wheel_selection(population, fitness_scores)
                parent2 = self.roulette_wheel_selection(population, fitness_scores)

                # Crossover
                if random.random() < self.crossover_rate:
                    child1, child2 = self.two_point_crossover(parent1, parent2, instance)
                else:
                    child1, child2 = copy.deepcopy(parent1), copy.deepcopy(parent2)

                # Mutation
                if random.random() < self.mutation_rate:
                    child1 = self.mutate(child1, instance)
                if random.random() < self.mutation_rate:
                    child2 = self.mutate(child2, instance)

                # Add children to the new population
                new_population.append(child1)
                if len(new_population) < self.population_size:
                    new_population.append(child2)

            # Update the population
            population = new_population

            # Re-evaluate fitness scores
            fitness_scores = self.evaluate_population(population)

            # Update the best solution if needed
            current_best_idx = fitness_scores.index(max(fitness_scores))
            current_best_makespan = calculate_makespan(population[current_best_idx])

            if current_best_makespan < best_makespan:
                best_schedule = copy.deepcopy(population[current_best_idx])
                best_makespan = current_best_makespan
                generations_no_improvement = 0
            else:
                generations_no_improvement += 1

            # Stop if no improvement for a long time
            if generations_no_improvement > 5:  # Reduce the threshold for early stopping
                print(f"No improvement for 5 generations. Stopping early...")
                break

            # Update progress bar
            progress_bar.update(1)

        progress_bar.close()

        return best_schedule, best_makespan

    def evaluate_population(self, population):
        """
        Evaluate the fitness of the population in parallel.
        """
        def fitness(individual):
            return 1.0 / calculate_makespan(individual)

        with ThreadPoolExecutor() as executor:
            fitness_scores = list(executor.map(fitness, population))

        return fitness_scores

    def initialize_population(self, instance, seed_schedule=None):
        """
        Initialize population with a mix of:
        - 50% solutions from Tabu Search
        - 50% completely random solutions
        """
        population = []

        # Generate 50% of the population using Tabu Search
        tabu = TabuSearch(max_iterations=50, max_iterations_no_improvement=25, enable_hypertuning=False)
        _, tabu_schedule, _ = tabu.optimize(instance)
        tabu_count = self.population_size // 2
        for _ in range(tabu_count):
            solution = self.mutate(copy.deepcopy(tabu_schedule), instance)
            population.append(solution)

        # Generate 50% completely random solutions
        random_count = self.population_size - len(population)
        for _ in range(random_count):
            solution = self.create_random_solution(instance)
            population.append(solution)

        return population

    def create_random_solution(self, instance):
        """Create a completely random solution"""
        # Start with IdleFilling
        idlefilling = IdleFillingHeuristic()
        _, initial_schedule, _ = idlefilling.optimize(instance)
        
        # Randomize all machine assignments
        random_solution = copy.deepcopy(initial_schedule)
        for job in random_solution:
            job['machine_id'] = random.randint(1, instance.num_machines)
            
        return random_solution
    
    def select_elite(self, population, fitness_scores):
        """Select the elite (best) individuals from the population"""
        combined = list(zip(population, fitness_scores))
        sorted_population = [x[0] for x in sorted(combined, key=lambda x: x[1], reverse=True)]
        return [copy.deepcopy(individual) for individual in sorted_population[:self.elite_size]]
    
    def roulette_wheel_selection(self, population, fitness_scores):
        """Select an individual using roulette wheel selection (higher fitness = higher chance)"""
        total_fitness = sum(fitness_scores)
        if total_fitness == 0:
            return copy.deepcopy(random.choice(population))
            
        selection_point = random.uniform(0, total_fitness)
        current_sum = 0
        
        for individual, fitness in zip(population, fitness_scores):
            current_sum += fitness
            if current_sum >= selection_point:
                return copy.deepcopy(individual)
        
        # Fallback (should never reach here)
        return copy.deepcopy(random.choice(population))
    
    def two_point_crossover(self, parent1, parent2, instance):
        """Perform two-point crossover ensuring machine and mold constraints are respected"""
        # Get job IDs for reference
        job_ids = [job['job_id'] for job in parent1]

        if len(job_ids) <= 2:
            # Not enough jobs for two-point crossover, just return copies
            return copy.deepcopy(parent1), copy.deepcopy(parent2)

        # Choose two crossover points
        crossover_points = sorted(random.sample(range(len(job_ids)), 2))

        # Initialize children as copies of parents
        child1 = copy.deepcopy(parent1)
        child2 = copy.deepcopy(parent2)

        # Exchange machine assignments in the crossover section
        for i in range(crossover_points[0], crossover_points[1]):
            job_id = job_ids[i]

            # Find job index in both children
            idx1 = next(idx for idx, job in enumerate(child1) if job['job_id'] == job_id)
            idx2 = next(idx for idx, job in enumerate(child2) if job['job_id'] == job_id)

            # Swap machine assignments
            child1[idx1]['machine_id'] = parent2[idx2]['machine_id']
            child2[idx2]['machine_id'] = parent1[idx1]['machine_id']

        return child1, child2
    
    def mutate(self, schedule, instance):
        """Mutate a schedule by moving a random job to a different machine."""
        mutated_schedule = copy.deepcopy(schedule)

        # Select a random job to mutate
        job_index = random.randint(0, len(mutated_schedule) - 1)

        # Get the current machine assignment for the selected job
        current_machine = mutated_schedule[job_index]['machine_id']

        # Create a list of possible machines excluding the current machine (1 to num_machines)
        possible_machines = [m for m in range(1, instance.num_machines + 1) if m != current_machine]

        # Assign the job to a new machine
        if possible_machines:  # Make sure there's at least one other machine
            new_machine = random.choice(possible_machines)
            mutated_schedule[job_index]['machine_id'] = new_machine

        return mutated_schedule
    
    def local_improvement(self, schedule, instance):
        """Apply a quick local search to improve a schedule"""
        improved_schedule = copy.deepcopy(schedule)
        current_makespan = calculate_makespan(improved_schedule)
        
        # Try moving each job to a different machine and keep the best move
        for i in range(len(improved_schedule)):
            original_machine = improved_schedule[i]['machine_id']
            
            for machine in range(1, instance.num_machines + 1):
                if machine != original_machine:
                    # Try this move
                    improved_schedule[i]['machine_id'] = machine
                    new_makespan = calculate_makespan(improved_schedule)
                    
                    if new_makespan < current_makespan:
                        # Keep this move
                        current_makespan = new_makespan
                    else:
                        # Revert the move
                        improved_schedule[i]['machine_id'] = original_machine
        
        return improved_schedule
    
    def hypertune_parameters(self, instance):
        """Apply pre-tuned parameters based on instance class and size"""
        num_jobs = len(instance.jobs)

        # Aggressive settings specifically to target the gap to lower bound
        self.population_size = 500
        self.max_generations = 500
        self.crossover_rate = 0.85
        self.mutation_rate = 0.3
        self.elite_size = 20
        self.time_limit = 60  # 5 minutes max
        if instance.num_molds==2:  # If the gap is very large, increase exploration
                self.mutation_rate = min(0.6, self.mutation_rate + 0.1)
                self.crossover_rate = max(0.6, self.crossover_rate - 0.1)
                self.population_size = min(800, self.population_size + 100)
        elif instance.num_molds==5:  # If the gap is moderate, balance exploration and exploitation
                self.mutation_rate = min(0.4, max(0.2, self.mutation_rate))
                self.crossover_rate = min(0.9, max(0.7, self.crossover_rate))
        else: 
                self.mutation_rate = max(0.1, self.mutation_rate - 0.05)
                self.crossover_rate = min(0.95, self.crossover_rate + 0.05)