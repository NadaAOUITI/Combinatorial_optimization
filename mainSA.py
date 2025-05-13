# main.py

import os
import matplotlib.pyplot as plt
from utils.file_io import parse_all_instances
from core.optimizer import *
from Visualization.gantt_chart import create_gantt_chart
from models.instance import Instance
import time

#-------------------------------Solve instance--------------------------------------------------
def solve_instance(instance: 'Instance', strategy, show_gantt=False, class_id=None):
    start_time = time.time()
    _, schedule, makespan = optimize(instance, class_id, strategy=strategy)  # Ignore the first return value

    end_time = time.time()
    execution_time = end_time - start_time

    if show_gantt:
        fig = create_gantt_chart(
            job_schedule=schedule,
            num_machines=instance.num_machines,
            makespan=makespan,
            instance_number=instance.instance_number,
            strategy=strategy
        )
        # plt.show()  # Uncomment if you want to display the Gantt chart

    return makespan, execution_time
#-------------------------------Save results to file--------------------------------------------------
def save_results_to_file(instances, strategies, filename, start_idx=0):
    with open(filename, "w") as log_file:
        header = f"{'Class':<8}{'Instance':<10}" + "".join(f"{s:<10}" for s in strategies) + "\n"
        log_file.write(header)
        log_file.write("=" * len(header) + "\n")
        
        for i, instance in enumerate(instances, start=start_idx+1):
            row = f"{str(instance.job_class):<8}{str(instance.instance_number):<10}"
            for strategy in strategies:
                makespan = solve_instance(instance, strategy, show_gantt=(i == 1 ))
                row += f"{makespan:<10}"
            log_file.write(row + "\n")
            
            if i % 100 == 0:
                print(f"Saved {i} instances to {filename}")
#-------------------------------End of save results to file--------------------------------------------------
#------------------------------Metaheuristics method file & calculates gap--------------------------------------------------
def save_metaheuristic_results(instances, strategies, lower_bounds_file, filename, start_idx=0):
    """Save metaheuristic results with lower bounds and gaps"""
    lower_bounds = {}
    with open(lower_bounds_file, 'r') as lb_file:
        for line in lb_file:
            parts = line.strip().split()
            if len(parts) >= 7:
                key = f"{parts[0]}_{parts[1]}_{parts[2]}_{parts[3]}"
                lb = int(parts[4])
                lower_bounds[key] = lb

    # Initialize gap counters
    gap_buckets = {f"{i}%": 0 for i in range(0, 6)}
    total_evaluated = 0

    with open(filename, "w") as log_file:
        # Write header
        header = (
            f"{'Jobs':<6}{'Molds':<6}{'Class':<8}{'Inst':<6}{'LB':<10}" +
            "".join(f"{s:<15}" for s in strategies) +
            "".join(f"Gap_{s:<15}" for s in strategies) + "\n"
        )
        log_file.write(header)
        log_file.write("=" * len(header) + "\n")

        for i, instance in enumerate(instances, start=start_idx+1):
            key = f"{instance.num_jobs}_{instance.num_molds}_{instance.job_class}_{instance.instance_number}"
            lb = lower_bounds.get(key, 0)

            row = (
                f"{instance.num_jobs:<6}{instance.num_molds:<6}"
                f"{str(instance.job_class):<8}{str(instance.instance_number):<6}"
                f"{lb:<10}"
            )

            gaps = []
            for strategy in strategies:
                makespan, _ = solve_instance(instance, strategy, show_gantt=(i == 1))
                row += f"{makespan:<15}"

                if lb > 0:
                    gap = ((makespan - lb) / lb) * 100
                    rounded_gap = round(gap)
                    if 0 <= rounded_gap <= 5:
                        gap_buckets[f"{rounded_gap}%"] += 1
                    total_evaluated += 1
                    gaps.append(f"{gap:.2f}%")
                else:
                    gaps.append("N/A")

            row += "".join(f"{gap:<15}" for gap in gaps)
            log_file.write(row + "\n")

            if i % 100 == 0:
                print(f"Saved {i} instances to {filename}")

    # Print final distribution
    print("\n📊 Gap distribution summary (out of", total_evaluated, "instances with known LB):")
    for percent, count in gap_buckets.items():
        ratio = (count / total_evaluated) * 100 if total_evaluated else 0
        print(f"  Gap = {percent:<3}: {count:<4} instances ({ratio:.2f}%)")
#------------------------------End of metaheuristics method file & calculates gap--------------------------------------------------
def main():
    instance_file = 'data/instance_A_test.txt'
    instances = parse_all_instances(instance_file)
    #-------Lower bounds file for metaheuristics------------------
    lower_bounds_file = 'data/res_B&B2_instancesA.txt'
    #end of lower bounds file for metaheuristics------------------
    if not instances:
        print(" No valid instances found in the data file.")
        return
   
    #---------------------------Testing metaheuristics with file & gap --------------------------------------------------
    strategies = ['SA']
     # Process in chunks if needed
    chunk_size = 1000
    for chunk_idx in range(0, len(instances), chunk_size):
        chunk = instances[chunk_idx:chunk_idx + chunk_size]
        file_num = (chunk_idx // chunk_size) + 1
        filename = f"{strategies[0]}_results_part{file_num}.txt"
        
        print(f"\nProcessing instances {chunk_idx + 1} to {min(chunk_idx + chunk_size, len(instances))}")
        save_metaheuristic_results(
            chunk, 
            strategies, 
            lower_bounds_file, 
            filename, 
            chunk_idx
        )
   

    print("\n✅ All metaheuristic results saved successfully!")

if __name__ == "__main__":
    main()