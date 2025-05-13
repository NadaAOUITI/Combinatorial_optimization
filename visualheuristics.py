import os
import matplotlib.pyplot as plt
import numpy as np

# Function to load BB data (lower bounds from the file)
def load_bb_data(filepath):
    optimal_solutions = []
    try:
        with open(filepath, "r") as f:
            for line in f:
                line = line.strip()
                if line:  # Skip empty lines
                    parts = [part for part in line.split() if part]
                    if len(parts) >= 5:
                        # Parse n, m, class_id, instance_id, lower_bound
                        lower_bound = float(parts[4])  # Use float for lower_bound
                        optimal_solutions.append(lower_bound)
        
        print(f"Loaded {len(optimal_solutions)} optimal solutions from {filepath}")
        return optimal_solutions
    
    except Exception as e:
        print(f"Error loading BB data: {e}")
        return []

# Function to load heuristic results from the file
def load_heuristic_data(filepath):
    heuristic_data = []
    try:
        with open(filepath, "r") as f:
            header_found = False
            for line in f:
                line = line.strip()
                if not line or line.startswith("="):  # Skip empty lines and separator lines
                    continue
                
                if not header_found:
                    header_found = True  # Skip the header line
                    continue
                
                parts = [part for part in line.split() if part]
                if len(parts) >= 6:
                    # Parse class_id, instance_id, and heuristic results
                    class_id = int(parts[0])
                    instance_id = int(parts[1])
                    spt = float(parts[2])
                    greedy = float(parts[3])
                    lpt = float(parts[4])
                    idle_filling = float(parts[5])
                    heuristic_data.append({
                        "Class": class_id,
                        "Instance": instance_id,
                        "SPT": spt,
                        "Greedy": greedy,
                        "LPT": lpt,
                        "IdleFilling": idle_filling
                    })
        
        print(f"Loaded {len(heuristic_data)} heuristic results from {filepath}")
        return heuristic_data
    
    except Exception as e:
        print(f"Error loading heuristic data: {e}")
        return []

# Function to process the heuristic data and calculate gaps
def process_heuristic_data(heuristic_data, optimal_solutions):
    for idx, entry in enumerate(heuristic_data):
        lb = optimal_solutions[idx] if idx < len(optimal_solutions) else float('inf')
        entry["LB"] = lb
        
        if lb == float('inf'):
            print(f"Warning: Missing lower bound for Class {entry['Class']}, Instance {entry['Instance']}")
        
        # Calculate gaps and determine optimality
        for heuristic in ["SPT", "Greedy", "LPT", "IdleFilling"]:
            if lb > 0:
                gap = ((entry[heuristic] - lb) / lb) * 100
            else:
                gap = float('inf')
            entry[f"{heuristic}_Gap"] = gap
            entry[f"{heuristic}_Optimal"] = gap <= 1
    
    return heuristic_data

# Function to count optimal solutions by class and heuristic
def count_optimal_solutions(processed_data):
    # Initialize counters for each class and heuristic
    optimal_counts = {}
    for class_id in range(1, 7):  # Classes 1-6
        optimal_counts[class_id] = {
            "SPT": 0, 
            "Greedy": 0, 
            "LPT": 0, 
            "IdleFilling": 0,
            "Total": 0  # To keep track of total instances per class
        }
    
    # Count optimal solutions
    for entry in processed_data:
        class_id = entry["Class"]
        if class_id in optimal_counts:
            optimal_counts[class_id]["Total"] += 1
            for heuristic in ["SPT", "Greedy", "LPT", "IdleFilling"]:
                if entry[f"{heuristic}_Optimal"]:
                    optimal_counts[class_id][heuristic] += 1
    
    # Calculate percentages
    for class_id in optimal_counts:
        total = optimal_counts[class_id]["Total"]
        if total > 0:
            for heuristic in ["SPT", "Greedy", "LPT", "IdleFilling"]:
                # Store both raw count and percentage
                optimal_counts[class_id][f"{heuristic}_Percent"] = (optimal_counts[class_id][heuristic] / total) * 100
    
    return optimal_counts

# Function to plot the optimal solution counts
def plot_optimal_solutions(optimal_counts):
    # Create a figure with subplots - one for uniform and one for binomial
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    
    # Classes
    uniform_classes = [1, 2, 3]
    binomial_classes = [4, 5, 6]
    heuristics = ["SPT", "Greedy", "LPT", "IdleFilling"]
    colors = ['blue', 'green', 'red', 'purple']
    
    # Plot for Uniform Classes (C1, C2, C3)
    x = np.arange(len(uniform_classes))
    width = 0.2  # Width of the bars
    
    for i, heuristic in enumerate(heuristics):
        percentages = [optimal_counts[c][f"{heuristic}_Percent"] for c in uniform_classes]
        ax1.bar(x + (i-1.5)*width, percentages, width, label=heuristic, color=colors[i])
    
    ax1.set_xticks(x)
    ax1.set_xticklabels(["C1", "C2", "C3"])
    ax1.set_title("Uniform Classes (C1, C2, C3)")
    ax1.set_xlabel("Class")
    ax1.set_ylabel("Percentage of Optimal Solutions (%)")
    ax1.set_ylim(0, 100)
    ax1.legend()
    ax1.grid(True, linestyle='--', alpha=0.6)
    
    # Plot for Binomial Classes (C4, C5, C6)
    x = np.arange(len(binomial_classes))
    
    for i, heuristic in enumerate(heuristics):
        percentages = [optimal_counts[c][f"{heuristic}_Percent"] for c in binomial_classes]
        ax2.bar(x + (i-1.5)*width, percentages, width, label=heuristic, color=colors[i])
    
    ax2.set_xticks(x)
    ax2.set_xticklabels(["C4", "C5", "C6"])
    ax2.set_title("Binomial Classes (C4, C5, C6)")
    ax2.set_xlabel("Class")
    ax2.set_ylabel("Percentage of Optimal Solutions (%)")
    ax2.set_ylim(0, 100)
    ax2.legend()
    ax2.grid(True, linestyle='--', alpha=0.6)
    
    # Add an overall title
    plt.suptitle("Percentage of Optimal Solutions by Class and Heuristic", fontsize=16)
    plt.tight_layout()
    
    # Save the figure
    plt.savefig("optimal_solutions_comparison.png", dpi=300, bbox_inches="tight")
    print("Graph saved as 'optimal_solutions_comparison.png'")
    
    # Show the plot
    plt.show()

# Main function
def main():
    # Load optimal solutions (lower bounds)
    bb_filepath = 'data/res_B&B2_instancesA.txt'  # Path to the lower bound file
    if not os.path.exists(bb_filepath):
        print(f"Error: File {bb_filepath} does not exist.")
        return
    
    optimal_solutions = load_bb_data(bb_filepath)
    
    # Load heuristic results
    results_filepath = 'results.txt'  # Path to the results file
    if not os.path.exists(results_filepath):
        print(f"Error: File {results_filepath} does not exist.")
        return
    
    heuristic_data = load_heuristic_data(results_filepath)
    
    # Process heuristic data with optimal solutions
    processed_data = process_heuristic_data(heuristic_data, optimal_solutions)
    
    # Count optimal solutions by class and heuristic
    optimal_counts = count_optimal_solutions(processed_data)
    
    # Print summary of optimal solutions
    print("\nSummary of Optimal Solutions by Class:")
    for class_id in range(1, 7):
        print(f"\nClass {class_id}:")
        for heuristic in ["SPT", "Greedy", "LPT", "IdleFilling"]:
            count = optimal_counts[class_id][heuristic]
            percent = optimal_counts[class_id][f"{heuristic}_Percent"]
            print(f"  {heuristic}: {count} optimal solutions ({percent:.2f}%)")
    
    # Plot the results
    plot_optimal_solutions(optimal_counts)

if __name__ == "__main__":
    main()
