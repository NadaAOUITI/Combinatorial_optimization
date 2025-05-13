from .base import Metaheuristic
import copy
import random
from tqdm import tqdm
from algorithms.heuristics.IdleFillingHeuristic import IdleFillingHeuristic
from algorithms.metaheuristics.neighborsgeneration import generate_diverse_neighbors, calculate_makespan, schedule_to_machines
import matplotlib.pyplot as plt

class TabuSearch(Metaheuristic):
    def __init__(self, tabu_tenure=30000, max_iterations=20000, max_iterations_no_improvement=10000, num_neighbors=20, class_id=1, enable_hypertuning=True):
        self.tabu_tenure = tabu_tenure
        self.max_iterations = max_iterations
        self.max_iterations_no_improvement = max_iterations_no_improvement
        self.num_neighbors = num_neighbors
        self.class_id = class_id  # Store class_id for hypertuning
        self.enable_hypertuning = enable_hypertuning  # Flag to enable/disable hypertuning
        
        # For tracking makespan over iterations
        self.iteration_history = []
        self.makespan_history = []

    def optimize(self, instance):
        # Apply hypertuning only if enabled
        if self.enable_hypertuning:
            self.hypertune_parameters(instance)
            
        # Generate initial solution using IdleFilling heuristic
        idlefilling = IdleFillingHeuristic()
        machines, initial_schedule, initial_makespan = idlefilling.optimize(instance)
        
        # Initialize best and current solutions
        best_schedule = copy.deepcopy(initial_schedule)
        best_makespan = initial_makespan
        current_schedule = copy.deepcopy(initial_schedule)
        current_makespan = initial_makespan
        
        # Initialize tabu list (store configurations as hashes)
        tabu_list = []
        
        # To see if neighbors are useful
        makespan_counts = {}  # Dictionary to count makespan frequencies
        
        # Record initial state
        self.iteration_history.append(0)
        self.makespan_history.append(best_makespan)
        
        # Initialize counters
        iterations = 0
        iterations_no_improvement = 0
        
        # Progress bar
        progress_bar = tqdm(desc="Tabu Search", total=self.max_iterations)
        
        while iterations < self.max_iterations and iterations_no_improvement < self.max_iterations_no_improvement:
            # Generate neighbors
            neighbors = generate_diverse_neighbors(current_schedule, instance.num_machines, self.num_neighbors)
            
            # Best non-tabu neighbor
            best_neighbor = None
            best_neighbor_makespan = float('inf')
            
            for neighbor in neighbors:
                # Create a hash for the neighbor configuration
                neighbor_hash = tuple(sorted((job['job_id'], job['machine_id']) for job in neighbor))
                neighbor_makespan = calculate_makespan(neighbor)
                
                # Count makespan frequencies
                if neighbor_makespan in makespan_counts:
                    makespan_counts[neighbor_makespan] += 1
                else:
                    makespan_counts[neighbor_makespan] = 1
                
                # Check if this neighbor is not in tabu list or satisfies aspiration criterion
                if neighbor_hash not in tabu_list or neighbor_makespan < best_makespan:
                    # If this is the best neighbor so far
                    if neighbor_makespan < best_neighbor_makespan:
                        best_neighbor = neighbor
                        best_neighbor_makespan = neighbor_makespan
            
            # If we found a valid neighbor
            if best_neighbor:
                # Move to that neighbor
                current_schedule = copy.deepcopy(best_neighbor)
                current_makespan = best_neighbor_makespan
                
                # Create a hash for the current configuration
                current_hash = tuple(sorted((job['job_id'], job['machine_id']) for job in current_schedule))
                
                # Add to tabu list
                tabu_list.append(current_hash)
                if len(tabu_list) > self.tabu_tenure:
                    tabu_list.pop(0)  # Remove oldest entry
                
                # Update best solution if needed
                if current_makespan < best_makespan:
                    best_schedule = copy.deepcopy(current_schedule)
                    best_makespan = current_makespan
                    iterations_no_improvement = 0
                    
                    # Record when we find a new best solution
                    self.iteration_history.append(iterations)
                    self.makespan_history.append(best_makespan)
                else:
                    iterations_no_improvement += 1
            else:
                # No valid neighbor found - could be a local optimum or all neighbors are tabu
                iterations_no_improvement += 1
            
            iterations += 1
            progress_bar.update(1)
            
        
        progress_bar.close()

        # Display makespan frequencies
        print("\n📊 Makespan Frequencies Across All Neighbors:")
        for mk, count in sorted(makespan_counts.items()):
            print(f"Makespan {mk} : {count} neighbors")
        
        print(f"\nTotal iterations: {iterations}")
        print(f"Best makespan: {best_makespan}")
        print(f"Initial makespan: {initial_makespan}")
        print(f"Improvement: {initial_makespan - best_makespan} ({(initial_makespan - best_makespan) / initial_makespan * 100:.2f}%)")

        # Convert best schedule to machine assignments
        best_machines = schedule_to_machines(best_schedule, instance.num_machines, instance.jobs)

        return best_machines, best_schedule, best_makespan
    


    def hypertune_parameters(self, instance):
        """Apply pre-tuned parameters based on instance class and size"""
        num_jobs = len(instance.jobs)
        
        if self.class_id in [1, 2, 3]:
            self.tabu_tenure = 10 
            self.max_iterations = 100
            self.max_iterations_no_improvement = 50
            self.num_neighbors = 10
        else:
            # Default hypertuning for bimodal classes
            self.tabu_tenure =10
            self.max_iterations = 100
            self.num_neighbors = 10
            if instance.num_molds < 5: #few molds number 4
                self.max_iterations_no_improvement = 50
            else: # many molds and it doesn' matter how many jobs number 5
                self.max_iterations_no_improvement = 10
            