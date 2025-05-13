# Simulated Annealing Parameter Tuning Tool
# This script performs a grid search over key SA parameters and visualizes the results

import copy
import math
import random
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
import itertools
import time
import os
import json  # To save best parameters
from algorithms.metaheuristics.simulated_annealing import SimulatedAnnealing  # Import your SA class
from utils.file_io import parse_all_instances

class SAParameterTuner:
    def __init__(self, instance, num_trials=3, save_directory='tuning_results'):
        """
        Initialize the parameter tuning tool.
        
        Args:
            instance: The problem instance to optimize
            num_trials: Number of times to run each parameter combination
            save_directory: Directory to save results and plots
        """
        self.instance = instance
        self.num_trials = num_trials
        self.save_directory = save_directory
        self.results_df = None
        
        # Create save directory if it doesn't exist
        if not os.path.exists(save_directory):
            os.makedirs(save_directory)
    
    def setup_grid_search(self, parameter_grid):
        """
        Set up the grid search with parameter combinations to try.
        
        Args:
            parameter_grid: Dictionary with parameter names as keys and lists of values to try
        
        Returns:
            List of parameter combinations to try
        """
        self.parameter_grid = parameter_grid
        
        # Generate all parameter combinations
        parameter_names = list(parameter_grid.keys())
        parameter_values = list(parameter_grid.values())
        combinations = list(itertools.product(*parameter_values))
        
        # Convert to list of dictionaries
        self.param_combinations = []
        for combo in combinations:
            param_dict = {}
            for i, param_name in enumerate(parameter_names):
                param_dict[param_name] = combo[i]
            self.param_combinations.append(param_dict)
        
        print(f"Set up grid search with {len(self.param_combinations)} parameter combinations")
        return self.param_combinations
    
    def evaluate_combination(self, params, trial_num):
        """
        Evaluate a single parameter combination.
        
        Args:
            params: Dictionary of parameters to use
            trial_num: Trial number for this combination
            
        Returns:
            Dictionary with results
        """
        # Create an instance of the SA algorithm with these parameters
        sa = SimulatedAnnealing(**params)
        
        # Measure execution time
        start_time = time.time()
        
        # Run the algorithm
        _, _, makespan = sa.optimize(self.instance)
        
        execution_time = time.time() - start_time
        
        # Return the results
        result = {
            'trial': trial_num,
            'makespan': makespan,
            'execution_time': execution_time,
            **params  # Include all parameters in the result
        }
        
        return result
    
    def run_grid_search(self, parallel=True):
        """
        Run the grid search across all parameter combinations.
        
        Args:
            parallel: Whether to use parallel processing
            
        Returns:
            DataFrame with results
        """
        print(f"Running grid search with {len(self.param_combinations)} combinations, {self.num_trials} trials each")
        
        # Generate tasks: each task is a parameter combination and trial number
        tasks = []
        for params in self.param_combinations:
            for trial in range(self.num_trials):
                tasks.append((params, trial))
        
        results = []
        if parallel and cpu_count() > 1:
            # Run in parallel
            num_processes = min(cpu_count(), 8)  # Limit to 8 processes max
            print(f"Running in parallel with {num_processes} processes")
            with Pool(num_processes) as pool:
                results = list(tqdm(pool.imap(self.evaluate_combination, tasks), total=len(tasks), desc="Evaluating parameter combinations"))
        else:
            # Run sequentially
            print("Running sequentially")
            for params, trial in tqdm(tasks, desc="Evaluating parameter combinations"):
                results.append(self.evaluate_combination(params, trial))
        
        # Convert results to DataFrame
        self.results_df = pd.DataFrame(results)
        
        # Save raw results
        self.results_df.to_csv(f"{self.save_directory}/grid_search_results.csv", index=False)
        
        return self.results_df
    
    def get_best_parameters(self):
        """
        Get the best parameter combination based on mean makespan.
        
        Returns:
            Dictionary with the best parameters
        """
        if self.results_df is None:
            raise ValueError("No results to analyze. Run grid_search first.")
        
        # Group by parameter combinations and calculate mean makespan
        param_columns = [col for col in self.results_df.columns if col not in ['trial', 'makespan', 'execution_time']]
        aggregated = self.results_df.groupby(param_columns).agg({'makespan': 'mean'}).reset_index()
        
        # Get the row with the minimum mean makespan
        best_row = aggregated.loc[aggregated['makespan'].idxmin()]
        
        # Extract parameter values
        best_params = {param: best_row[param] for param in param_columns}
        
        print("\n🏆 Best Parameter Combination:")
        for param, value in best_params.items():
            print(f"  - {param}: {value}")
        
        print(f"\nPerformance Metrics:")
        print(f"  - Mean Makespan: {best_row['makespan']:.2f}")
        
        return best_params

# Main function to run the parameter tuning
if __name__ == "__main__":
    # Parse the input file and get the first class instance
    input_file = 'data/instance_A_test.txt'  # Replace with the actual path
    instances = parse_all_instances(input_file)
    first_class_instance = instances[0]  # Assuming the first instance belongs to the first class

    # Initialize the tuner
    tuner = SAParameterTuner(first_class_instance)

    # Define the parameter grid
    parameter_grid = {
        'initial_temp': [500, 1000, 2000, 5000],
        'cooling_rate': [0.9, 0.95, 0.97, 0.99],
        'max_iterations': [100, 300, 500, 700],
        'min_temp': [0.1, 1, 5]
    }

    # Set up and run the grid search
    tuner.setup_grid_search(parameter_grid)
    tuner.run_grid_search()

    # Get the best parameters
    best_params = tuner.get_best_parameters()

    # Save the best parameters to a file
    with open("best_sa_parameters.json", "w") as f:
        json.dump(best_params, f, indent=4)

    print("\n✅ Best parameters saved to 'best_sa_parameters.json'")