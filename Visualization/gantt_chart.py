# Visualization/gantt_chart.py
import matplotlib.pyplot as plt
import numpy as np

# --------------------------------------------------- Gantt Chart for Scheduling ------------------------------------------------------------

def create_gantt_chart(job_schedule, num_machines, makespan, instance_number,strategy):
    """
    Creates a Gantt chart visualization for the job schedule with mold constraints.
    
    Args:
        job_schedule (list of dict): A list containing the job scheduling information for each job.
        num_machines (int): The number of machines.
        makespan (int): The makespan (total completion time).
        instance_number (int): The instance identifier, used in the chart title.
    
    Returns:
        fig (matplotlib.figure.Figure): The generated Gantt chart figure.
    """
    
    # Prepare figure and axis
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Define colors for each unique mold (this way, each mold gets a distinct color)
    unique_molds = set(job['mold_id'] for job in job_schedule)
    colors = plt.cm.tab20(np.linspace(0, 1, len(unique_molds)))
    mold_color_map = {mold: colors[i] for i, mold in enumerate(unique_molds)}
    
    # Plot each job as a colored bar on the Gantt chart
    for job in job_schedule:
        ax.barh(
            y=job['machine_id'],  # The machine assigned to the job
            width=job['duration'],  # Job duration
            left=job['start_time'],  # Job start time
            color=mold_color_map[job['mold_id']],  # Color by mold
            edgecolor='black',  # Border around each bar
            alpha=0.7  # Transparency for readability
        )
        
        # Add job ID and mold ID as text labels on the bars
        ax.text(
            job['start_time'] + job['duration']/2,  # Horizontal center of the bar
            job['machine_id'],  # Machine ID for vertical positioning
            f"J{job['job_id']}(M{job['mold_id']})",  # Job and Mold info
            ha='center',  # Horizontal alignment
            va='center',  # Vertical alignment
            color='black',  # Text color
            fontweight='bold'  # Make the text bold
        )
    
    # Customize the chart's appearance
    ax.set_xlabel('Time')  # X-axis label (Time)
    ax.set_ylabel('Machine')  # Y-axis label (Machines)
    ax.set_title(f'Gantt Chart - Instance {instance_number} -{strategy} Scheduling with Mold Constraints')  # Title
    
    # Set machine labels on the Y-axis and adjust x-axis limits based on makespan
    ax.set_yticks(range(1, num_machines + 1))  # Y-ticks: one for each machine
    ax.set_xlim(0, makespan + 1)  # X-axis: Time range from 0 to makespan + 1 for padding
    ax.grid(True, axis='x', linestyle='--', alpha=0.7)  # Grid on X-axis for readability
    
    # Add a legend that identifies each mold with its corresponding color
    mold_patches = [plt.Rectangle((0,0),1,1, color=mold_color_map[m]) for m in unique_molds]
    ax.legend(mold_patches, [f"Mold {m}" for m in unique_molds], 
              loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=5)  # Adjust the legend placement
    
    # Tight layout to ensure no clipping of labels
    plt.tight_layout()
    
    return fig  # Return the figure for further use (e.g., saving or displaying)
