import matplotlib.pyplot as plt
from collections import defaultdict
import seaborn as sns

# Load optimal solutions with mold info
optimal_solutions = []
mold_instance_count = defaultdict(int)

with open("data/res_B&B2_instancesA.txt", "r") as f:
    lines = f.readlines()
    for line in lines:
        parts = line.split()
        if len(parts) >= 5:
            n, m, class_id, instance_id, lower_bound = map(int, parts[:5])
            # Store tuple with all relevant info (jobs, molds, class, instance, optimal value)
            optimal_solutions.append((n, m, class_id, instance_id, lower_bound))
            # Count instances per mold number
            mold_instance_count[m] += 1

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

# Count optimal solutions per heuristic per mold count
heuristic_mold_counts = defaultdict(lambda: defaultdict(int))
for idx, (_, m, _, _, optimal_value) in enumerate(optimal_solutions):
    for heuristic, results in heuristic_results.items():
        heuristic_value = results[idx]
        if abs(heuristic_value - optimal_value) / optimal_value <= 0.01:
            heuristic_mold_counts[heuristic][m] += 1

# Plot by number of molds
sns.set(style="whitegrid")
colors = sns.color_palette("Set2", 4)  # 4 heuristics

# Get unique mold counts and sort them
mold_counts = sorted(set(m for _, m, _, _, _ in optimal_solutions))
bar_width = 0.18
fig, ax = plt.subplots(figsize=(12, 6))

# Background: total instances per mold count
for i, mold_count in enumerate(mold_counts):
    ax.bar(mold_count - 0.1, mold_instance_count[mold_count], width=0.75, color="lightgrey", zorder=0)

# Foreground: heuristic bars
for i, (heuristic, mold_performance) in enumerate(heuristic_mold_counts.items()):
    values = [mold_performance.get(m, 0) for m in mold_counts]
    bar_x = [m + (i - 1.5) * bar_width for m in mold_counts]
    ax.bar(bar_x, values, width=bar_width, label=heuristic, color=colors[i], zorder=3)

# Axes and labels
ax.set_xticks(mold_counts)
ax.set_xticklabels([f"M={m}" for m in mold_counts])
ax.set_xlabel("Number of Molds", fontsize=12)
ax.set_ylabel("Number of Optimal Solutions", fontsize=12)
ax.set_title("Heuristic Performance by Number of Molds", fontsize=14)
ax.legend()
plt.tight_layout()
plt.savefig("heuristic_by_molds.png")
plt.show()

# Additional analysis: By number of jobs
job_instance_count = defaultdict(int)
for n, _, _, _, _ in optimal_solutions:
    job_instance_count[n] += 1

# Count optimal solutions per heuristic per job count
heuristic_job_counts = defaultdict(lambda: defaultdict(int))
for idx, (n, _, _, _, optimal_value) in enumerate(optimal_solutions):
    for heuristic, results in heuristic_results.items():
        heuristic_value = results[idx]
        if abs(heuristic_value - optimal_value) / optimal_value <= 0.01:
            heuristic_job_counts[heuristic][n] += 1

# Plot by number of jobs
job_counts = sorted(set(n for n, _, _, _, _ in optimal_solutions))
fig, ax = plt.subplots(figsize=(14, 6))

# Background: total instances per job count
for i, job_count in enumerate(job_counts):
    ax.bar(job_count - 0.1, job_instance_count[job_count], width=0.75, color="lightgrey", zorder=0)

# Foreground: heuristic bars
bar_width = 0.18
for i, (heuristic, job_performance) in enumerate(heuristic_job_counts.items()):
    values = [job_performance.get(n, 0) for n in job_counts]
    bar_x = [n + (i - 1.5) * bar_width for n in job_counts]
    ax.bar(bar_x, values, width=bar_width, label=heuristic, color=colors[i], zorder=3)

# Axes and labels
ax.set_xticks(job_counts)
ax.set_xticklabels([f"N={n}" for n in job_counts])
ax.set_xlabel("Number of Jobs", fontsize=12)
ax.set_ylabel("Number of Optimal Solutions", fontsize=12)
ax.set_title("Heuristic Performance by Number of Jobs", fontsize=14)
ax.legend()
plt.tight_layout()
plt.savefig("heuristic_by_jobs.png")
plt.show()

# Create a combined visualization for better comparison across mold counts
fig, axes = plt.subplots(1, len(mold_counts), figsize=(18, 6), sharey=True)
fig.suptitle("Heuristic Performance by Number of Molds", fontsize=16)

for i, mold_count in enumerate(mold_counts):
    # Create a dict to count optimal solutions for each heuristic for this mold count
    mold_heuristic_counts = {h: 0 for h in heuristic_results.keys()}
    
    # Count optimal solutions for this mold count
    for idx, (_, m, _, _, optimal_value) in enumerate(optimal_solutions):
        if m == mold_count:
            for heuristic, results in heuristic_results.items():
                heuristic_value = results[idx]
                if abs(heuristic_value - optimal_value) / optimal_value <= 0.01:
                    mold_heuristic_counts[heuristic] += 1
    
    # Plot for this mold count
    axes[i].bar(range(len(mold_heuristic_counts)), 
                list(mold_heuristic_counts.values()), 
                color=colors)
    axes[i].set_title(f"M={mold_count}")
    axes[i].set_xticks(range(len(mold_heuristic_counts)))
    axes[i].set_xticklabels(mold_heuristic_counts.keys(), rotation=45)
    
    # Add total instance count
    axes[i].text(0.5, 0.9, f"Total: {mold_instance_count[mold_count]}", 
                transform=axes[i].transAxes, ha='center')

# Add common labels
fig.text(0.5, 0.01, "Heuristics", fontsize=14, ha='center')
fig.text(0.01, 0.5, "Number of Optimal Solutions", fontsize=14, va='center', rotation='vertical')
plt.tight_layout(rect=[0.02, 0.03, 1, 0.95])
plt.savefig("heuristic_by_molds_combined.png")
plt.show()

# Calculate and plot performance ratios (percentage of optimal solutions)
# This gives a better comparison independent of instance counts
fig, ax = plt.subplots(figsize=(12, 6))

for i, (heuristic, mold_performance) in enumerate(heuristic_mold_counts.items()):
    ratio_values = [mold_performance.get(m, 0) / mold_instance_count[m] * 100 for m in mold_counts]
    ax.plot(mold_counts, ratio_values, marker='o', linewidth=2, markersize=8, label=heuristic, color=colors[i])

ax.set_xticks(mold_counts)
ax.set_xticklabels([f"M={m}" for m in mold_counts])
ax.set_xlabel("Number of Molds", fontsize=12)
ax.set_ylabel("Percentage of Optimal Solutions (%)", fontsize=12)
ax.set_title("Heuristic Performance Ratio by Number of Molds", fontsize=14)
ax.grid(True, linestyle='--', alpha=0.7)
ax.legend()
plt.tight_layout()
plt.savefig("heuristic_ratio_by_molds.png")
plt.show()