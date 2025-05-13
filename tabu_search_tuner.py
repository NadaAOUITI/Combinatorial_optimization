import os
import json
from algorithms.metaheuristics.tabu_search import TabuSearch
from utils.file_io import parse_all_instances 

class TabuSearchTuner:
    def __init__(self, instance, save_directory='tuning_results'):
        """
        Initialize the Tabu Search tuning tool.

        Args:
            instance: The problem instance to optimize.
            save_directory: Directory to save results and plots.
        """
        self.instance = instance
        self.save_directory = save_directory

       
        if not os.path.exists(save_directory):
            os.makedirs(save_directory)

    def setup_grid_search(self, parameter_grid):
        """
        Set up the grid search with parameter combinations to try.

        Args:
            parameter_grid: Dictionary with parameter names as keys and lists of values to try.

        Returns:
            List of parameter combinations to try.
        """
        from itertools import product

        self.parameter_grid = parameter_grid
        parameter_names = list(parameter_grid.keys())
        parameter_values = list(parameter_grid.values())
        combinations = list(product(*parameter_values))

        # Convert to list of dictionaries
        self.param_combinations = [
            dict(zip(parameter_names, combo)) for combo in combinations
        ]

        print(f"Set up grid search with {len(self.param_combinations)} parameter combinations.")
        return self.param_combinations

    def run_grid_search(self):
        """
        Run the grid search across all parameter combinations.

        Returns:
            List of results with parameter combinations and their makespan.
        """
        results = []

        print(f"Running grid search with {len(self.param_combinations)} combinations.")
        for params in self.param_combinations:
            print(f"Testing parameters: {params}")

            # Initialize Tabu Search with the current parameter combination
            ts = TabuSearch(
                tabu_tenure=params['tabu_tenure'],
                max_iterations=params['max_iterations'],
                max_iterations_no_improvement=params['max_iterations_no_improvement'],
                num_neighbors=params['num_neighbors'],
                enable_hypertuning=False  # Disable built-in hypertuning for manual tuning
            )

            # Run Tabu Search on the instance
            _, _, makespan = ts.optimize(self.instance)

            # Record the result
            results.append({'params': params, 'makespan': makespan})
            print(f"Result: Makespan = {makespan}")

        # Save raw results
        results_file = os.path.join(self.save_directory, "grid_search_results.json")
        with open(results_file, "w") as f:
            json.dump(results, f, indent=4)

        print(f" Saved grid search results to {results_file}")
        return results

    def get_best_parameters(self, results):
        """
        Get the best parameter combination based on the makespan.

        Args:
            results: List of results with parameter combinations and their makespan.

        Returns:
            Dictionary with the best parameters.
        """
        best_result = min(results, key=lambda x: x['makespan'])
        best_params = best_result['params']
        print("\n🏆 Best Parameter Combination:")
        for param, value in best_params.items():
            print(f"  - {param}: {value}")
        print(f"\nPerformance Metrics:")
        print(f"  - Makespan: {best_result['makespan']:.2f}")
        return best_params


# Main function to run the parameter tuning
if __name__ == "__main__":
    # Parse the input file and get the first class instance
    #input_file = 'data/instance_A_test.txt'  # Replace with the actual path
    input_file = 'data/manymoldsmanyjobs.txt'  # Replace with the actual path
    instances = parse_all_instances(input_file)
    for idx, instance in enumerate(instances):
        print(f"\n🔍 Tuning parameters for Instance #{idx + 1}")

        # Create a unique folder for each instance's results
        save_dir = f'tuning_results/instance_{idx + 1}'

        # Initialize the tuner with the current instance
        tuner = TabuSearchTuner(instance, save_directory=save_dir)

        # Define the parameter grid
        parameter_grid = {
            'tabu_tenure': [10, 20, 50],
            'max_iterations': [100, 500, 1000],
            'max_iterations_no_improvement': [50, 100, 200],
            'num_neighbors': [10, 20, 50]
        }

        # Run grid search
        tuner.setup_grid_search(parameter_grid)
        results = tuner.run_grid_search()

        # Get best parameters
        best_params = tuner.get_best_parameters(results)

        # Save best parameters for this instance
        best_params_file = os.path.join(save_dir, "best_tabu_parameters.json")
        with open(best_params_file, "w") as f:
            json.dump(best_params, f, indent=4)

        print(f"✅ Saved best parameters for instance #{idx + 1} to {best_params_file}")