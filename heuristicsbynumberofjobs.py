import matplotlib.pyplot as plt
from collections import defaultdict
import seaborn as sns
import numpy as np

# Load optimal solutions with job info
optimal_solutions = []
job_instance_count = defaultdict(int)

with open("data/res_B&B2_instancesA.txt", "r") as f:
    lines = f.readlines()
    for line in lines:
        parts = line.split()
        if len(parts) >= 5:
            n, m, class_id, instance_id, lower_bound = map(int, parts[:5])
            # Store tuple with all relevant info (jobs, molds, class, instance, optimal value)
            optimal_solutions.append((n, m, class_id, instance_id, lower_bound))
            # Count instances per job number
            job_instance_count[n] += 1

# Load heuristic results
heuristic_results = {"SPT": [], "LPT": [], "Greedy": [], "IdleFilling": []}
with open("results.txt", "r") as f:
    lines = f.readlines()[2:]  # Skip header lines
    for line in lines:
        parts = line.split()
        if len(parts) >= 6:
            spt, greedy, lpt, idle_filling = map(int, parts[2:6])
            heuristic_results["SPT"].append(spt)
            heuristic_results["Greedy"].append(greedy)
            heuristic_results["LPT"].append(lpt)
            heuristic_results["IdleFilling"].append(idle_filling)

# Count optimal solutions per heuristic per job count
heuristic_job_counts = defaultdict(lambda: defaultdict(int))
for idx, (n, _, _, _, optimal_value) in enumerate(optimal_solutions):
    for heuristic, results in heuristic_results.items():
        if idx < len(results):  # Make sure we don't go out of bounds
            heuristic_value = results[idx]
            if abs(heuristic_value - optimal_value) / optimal_value <= 0.01:
                heuristic_job_counts[heuristic][n] += 1

# Plot by number of jobs
sns.set(style="whitegrid")
colors = sns.color_palette("Set2", 4)  # 4 heuristics

# Get unique job counts and sort them
job_counts = sorted(set(n for n, _, _, _, _ in optimal_solutions))
bar_width = 0.18
fig, ax = plt.subplots(figsize=(14, 8))

# Background: total instances per job count
for i, job_count in enumerate(job_counts):
    ax.bar(job_count, job_instance_count[job_count], width=0.75, color="lightgrey", zorder=0, alpha=0.7)

# Foreground: heuristic bars
for i, (heuristic, job_performance) in enumerate(heuristic_job_counts.items()):
    values = [job_performance.get(n, 0) for n in job_counts]
    bar_x = [n + (i - 1.5) * bar_width for n in job_counts]
    ax.bar(bar_x, values, width=bar_width, label=heuristic, color=colors[i], zorder=3)

# Axes and labels
ax.set_xticks(job_counts)
ax.set_xticklabels([f"N={n}" for n in job_counts])
ax.set_xlabel("Number of Jobs", fontsize=14)
ax.set_ylabel("Number of Optimal Solutions", fontsize=14)
ax.set_title("Heuristic Performance by Number of Jobs", fontsize=16)
ax.legend(fontsize=12)
plt.tight_layout()
plt.savefig("heuristic_by_jobs.png")
plt.show()

# Calculate and plot performance ratios (percentage of optimal solutions)
fig, ax = plt.subplots(figsize=(14, 8))

for i, (heuristic, job_performance) in enumerate(heuristic_job_counts.items()):
    ratio_values = [job_performance.get(n, 0) / job_instance_count[n] * 100 for n in job_counts]
    ax.plot(job_counts, ratio_values, marker='o', linewidth=2, markersize=10, 
            label=heuristic, color=colors[i])

ax.set_xticks(job_counts)
ax.set_xticklabels([f"N={n}" for n in job_counts])
ax.set_xlabel("Number of Jobs", fontsize=14)
ax.set_ylabel("Percentage of Optimal Solutions (%)", fontsize=14)
ax.set_title("Heuristic Performance Ratio by Number of Jobs", fontsize=16)
ax.grid(True, linestyle='--', alpha=0.7)
ax.legend(fontsize=12)
plt.tight_layout()
plt.savefig("heuristic_ratio_by_jobs.png")
plt.show()

# Create a combined visualization for better comparison across job counts
fig, axes = plt.subplots(1, len(job_counts), figsize=(20, 6), sharey=True)
fig.suptitle("Heuristic Performance by Number of Jobs", fontsize=16)

# Define consistent colors for each heuristic
heuristic_colors = {
    "IdleFilling": colors[0],
    "Greedy": colors[1],
    "LPT": colors[2],
    "SPT": colors[3]
}

for i, job_count in enumerate(job_counts):
    # Create a dict to count optimal solutions for each heuristic for this job count
    job_heuristic_counts = {h: 0 for h in heuristic_results.keys()}
    
    # Count optimal solutions for this job count
    for idx, (n, _, _, _, optimal_value) in enumerate(optimal_solutions):
        if n == job_count and idx < len(list(heuristic_results.values())[0]):
            for heuristic, results in heuristic_results.items():
                heuristic_value = results[idx]
                if abs(heuristic_value - optimal_value) / optimal_value <= 0.01:
                    job_heuristic_counts[heuristic] += 1
    
    # Plot for this job count
    heuristics = list(job_heuristic_counts.keys())
    values = list(job_heuristic_counts.values())
    
    # Use consistent colors for each heuristic
    bars = axes[i].bar(range(len(heuristics)), values, 
              color=[heuristic_colors[h] for h in heuristics])
    
    axes[i].set_title(f"N={job_count}")
    axes[i].set_xticks(range(len(heuristics)))
    axes[i].set_xticklabels(heuristics, rotation=45)
    
    # Add percentage labels on top of each bar
    for bar_idx, bar in enumerate(bars):
        height = bar.get_height()
        percentage = (height / job_instance_count[job_count]) * 100
        axes[i].text(bar.get_x() + bar.get_width()/2., height + 5,
                    f'{percentage:.1f}%', ha='center', va='bottom', fontsize=9)
    
    # Add total instance count
    axes[i].text(0.5, 0.9, f"Total: {job_instance_count[job_count]}", 
                transform=axes[i].transAxes, ha='center')

# Add common labels
fig.text(0.5, 0.01, "Heuristics", fontsize=14, ha='center')
fig.text(0.01, 0.5, "Number of Optimal Solutions", fontsize=14, va='center', rotation='vertical')
plt.tight_layout(rect=[0.02, 0.03, 1, 0.95])
plt.savefig("heuristic_by_jobs_combined.png")
plt.show()

# Create a comprehensive table-like visualization with ratio values
fig, ax = plt.subplots(figsize=(14, 8))

x = np.arange(len(job_counts))
width = 0.2
offset = np.linspace(-0.3, 0.3, 4)

for i, (heuristic, job_performance) in enumerate(heuristic_job_counts.items()):
    ratio_values = [job_performance.get(n, 0) / job_instance_count[n] * 100 for n in job_counts]
    bars = ax.bar(x + offset[i], ratio_values, width, label=heuristic, color=colors[i])
    
    # Add percentage labels
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{height:.1f}%', ha='center', va='bottom', fontsize=9)

ax.set_xticks(x)
ax.set_xticklabels([f"N={n}" for n in job_counts])
ax.set_xlabel("Number of Jobs", fontsize=14)
ax.set_ylabel("Percentage of Optimal Solutions (%)", fontsize=14)
ax.set_title("Comparison of Heuristic Performance Ratio by Number of Jobs", fontsize=16)
ax.legend(fontsize=12)
ax.grid(True, axis='y', linestyle='--', alpha=0.7)

plt.tight_layout()
plt.savefig("heuristic_ratio_comparison_by_jobs.png")
plt.show()