# metaheuristics/neighborsgeneration.py

from models.machine import Machine
import copy
import random

# -----------------------------Calculating Makespan---------------------------------------------------------------------------------

def calculate_makespan(schedule):
    """
    Calculate the makespan (maximum completion time) of a schedule.
    
    Args:
        schedule (list): The schedule to evaluate
        
    Returns:
        float: The makespan of the schedule
    """
    if not schedule:
        return 0
    return max(job['end_time'] for job in schedule)

# ---------------------------------------Recalculating the schedule of the machine--------------------------------------

def recalculate_schedule(schedule, num_machines):
    """
    Recalculate the start and end times of jobs in the schedule
    to ensure the schedule is feasible with respect to machine and mold constraints.
    
    Args:
        schedule (list): The schedule to recalculate
        num_machines (int): Number of machines
        
    Returns:
        list: The recalculated schedule
    """
    # Initialize empty machines and mold release times
    machine_times = {i+1: 0 for i in range(num_machines)}
    mold_times = {}
    
    # Group jobs by machine and sort by start time to preserve order
    machine_jobs = {}
    for job in schedule:
        machine_id = job['machine_id']
        if machine_id not in machine_jobs:
            machine_jobs[machine_id] = []
        machine_jobs[machine_id].append(job)
    
    # Process each machine's jobs
    recalculated_schedule = []
    for machine_id, jobs in machine_jobs.items():
        for job in jobs:
            mold_id = job['mold_id']
            duration = job['duration']
            
            # Determine the earliest start time considering both machine and mold availability
            earliest_start = max(machine_times.get(machine_id, 0), mold_times.get(mold_id, 0))
            
            # Update job timing
            job['start_time'] = earliest_start
            job['end_time'] = earliest_start + duration
            
            # Update machine and mold availability
            machine_times[machine_id] = job['end_time']
            mold_times[mold_id] = job['end_time']
            
            recalculated_schedule.append(job)
    
    return recalculated_schedule

# ----------------------------------Assigning jobs to machines---------------------------------------------------------------------------

def schedule_to_machines(schedule, num_machines, jobs):
    """
    Convert a schedule representation to a list of Machine objects with assigned jobs.
    
    Args:
        schedule (list): The schedule to convert
        num_machines (int): Number of machines
        jobs (list): List of Job objects
        
    Returns:
        list: List of Machine objects with assigned jobs
    """
    # Create empty machines
    machines = [Machine(i + 1) for i in range(num_machines)]
    
    # Group jobs by machine
    machine_jobs = {}
    for job in schedule:
        machine_id = job['machine_id']
        if machine_id not in machine_jobs:
            machine_jobs[machine_id] = []
        machine_jobs[machine_id].append(job)
    
    # Sort jobs by start time within each machine
    for machine_id in machine_jobs:
        machine_jobs[machine_id].sort(key=lambda x: x['start_time'])
    
    # Find the corresponding job objects and assign them to machines
    job_dict = {job.job_id: job for job in jobs}
    
    for machine_id, jobs_on_machine in machine_jobs.items():
        for job_info in jobs_on_machine:
            job_obj = job_dict.get(job_info['job_id'])
            if job_obj:
                machines[machine_id - 1].assign_job(job_obj)  # Adjust for 0-indexing
    
    return machines

# -----------------------------------------Generating neighbors using different strategies-----------------------------------------------

def swap_jobs(schedule, num_machines):
    """
    Generate a neighboring solution by swapping two random jobs between machines.
    
    Args:
        schedule (list): Current schedule
        num_machines (int): Number of machines
        
    Returns:
        list: A new schedule with jobs swapped between machines
    """
    if len(schedule) < 2:
        return copy.deepcopy(schedule)
    
    neighbor = copy.deepcopy(schedule)
    
    # Select two random jobs from different machines
    machine1_jobs = [i for i, job in enumerate(neighbor) if job['machine_id'] == 1]
    machine2_jobs = [i for i, job in enumerate(neighbor) if job['machine_id'] == 2]
    
    if not machine1_jobs or not machine2_jobs:
        return neighbor
        
    job1_idx = random.choice(machine1_jobs)
    job2_idx = random.choice(machine2_jobs)
    
    # Swap machine assignments
    neighbor[job1_idx]['machine_id'] = 2
    neighbor[job2_idx]['machine_id'] = 1
    
    # Recalculate schedule with new machine assignments
    return recalculate_schedule(neighbor, num_machines)

def move_job(schedule, num_machines):
    """
    Generate a neighboring solution by moving a random job to the other machine.
    
    Args:
        schedule (list): Current schedule
        num_machines (int): Number of machines
        
    Returns:
        list: A new schedule with one job moved to a different machine
    """
    if not schedule:
        return []
        
    neighbor = copy.deepcopy(schedule)
    
    # Select a random job
    job_idx = random.randrange(len(neighbor))
    
    # Move to other machine
    current_machine = neighbor[job_idx]['machine_id']
    neighbor[job_idx]['machine_id'] = 3 - current_machine  # Toggle between 1 and 2
    
    # Recalculate schedule
    return recalculate_schedule(neighbor, num_machines)

def swap_adjacent_jobs(schedule, num_machines):
    """
    Generate a neighboring solution by swapping two adjacent jobs on the same machine.
    
    Args:
        schedule (list): Current schedule
        num_machines (int): Number of machines
        
    Returns:
        list: A new schedule with adjacent jobs swapped
    """
    neighbor = copy.deepcopy(schedule)
    
    # Select a random machine
    machine_id = random.randint(1, num_machines)
    
    # Get jobs on this machine sorted by start time
    machine_jobs = [(i, job) for i, job in enumerate(neighbor) if job['machine_id'] == machine_id]
    machine_jobs.sort(key=lambda x: x[1]['start_time'])
    
    if len(machine_jobs) < 2:
        return neighbor
    
    # Select a random position (avoiding the last position)
    pos = random.randint(0, len(machine_jobs) - 2)
    
    # Get the indices of the two adjacent jobs
    idx1 = machine_jobs[pos][0]
    idx2 = machine_jobs[pos + 1][0]
    
    # Swap their positions
    temp_info = neighbor[idx1].copy()
    neighbor[idx1] = neighbor[idx2].copy()
    neighbor[idx2] = temp_info
    
    # Recalculate schedule
    return recalculate_schedule(neighbor, num_machines)

def subtour_reversal(schedule, num_machines):
    """
    Generate a neighboring solution by reversing a subsequence of jobs on one machine.
    
    Args:
        schedule (list): Current schedule
        num_machines (int): Number of machines
        
    Returns:
        list: A new schedule with a subsequence reversed
    """
    neighbor = copy.deepcopy(schedule)
    
    # Select a random machine
    machine_id = random.randint(1, num_machines)
    
    # Get all jobs scheduled on this machine
    machine_jobs_indices = [i for i, job in enumerate(neighbor) if job['machine_id'] == machine_id]
    
    if len(machine_jobs_indices) <= 1:
        return neighbor
    
    # Choose two random positions to determine the subsequence to reverse
    if len(machine_jobs_indices) == 2:
        pos1, pos2 = 0, 1
    else:
        pos1, pos2 = sorted(random.sample(range(len(machine_jobs_indices)), 2))
    
    # Get the actual indices in the schedule
    subsequence_indices = machine_jobs_indices[pos1:pos2+1]
    
    # Get the subsequence
    subsequence = [neighbor[i] for i in subsequence_indices]
    
    # Reverse the subsequence
    reversed_subsequence = list(reversed(subsequence))
    
    # Put the reversed subsequence back
    for i, idx in enumerate(subsequence_indices):
        neighbor[idx] = reversed_subsequence[i]
    
    # Recalculate schedule
    return recalculate_schedule(neighbor, num_machines)

def generate_neighbor(schedule, num_machines):
    """
    Generate a neighboring solution using one of several neighbor generation strategies.
    
    Args:
        schedule (list): Current schedule
        num_machines (int): Number of machines
    
    Returns:
        list: A new neighbor schedule
    """
    # Choose a random neighbor generation strategy
    strategies = [
        swap_jobs,
        move_job,
        swap_adjacent_jobs,
        subtour_reversal
    ]
    
    chosen_strategy = random.choice(strategies)
    return chosen_strategy(schedule, num_machines)


def generate_diverse_neighbors(schedule, num_machines, num_neighbors=100):
    """
    Generate a diverse set of neighbors using multiple strategies.
    
    Args:
        schedule (list): Current schedule
        num_machines (int): Number of machines
        num_neighbors (int): Number of neighbors to generate
    
    Returns:
        list: List of neighboring schedules
    """
    neighbors = []
    seen_configurations = set()  # To avoid duplicates
    
    # Calculate the original makespan for reference
    original_makespan = calculate_makespan(schedule)
    
    attempts = 0
    max_attempts = num_neighbors * 10  # Set a limit to avoid infinite loops
    
    while len(neighbors) < num_neighbors and attempts < max_attempts:
        attempts += 1
        new_neighbor = generate_neighbor(schedule, num_machines)
        
        # Calculate new makespan
        new_makespan = calculate_makespan(new_neighbor)
        
        # Create a hash of the machine assignments to detect duplicates
        config_hash = tuple(sorted((job['job_id'], job['machine_id']) for job in new_neighbor))
        
        # Only accept neighbor if it's unique and not much worse than original
        # Allow some worse solutions to escape local optima but within reason
        if config_hash not in seen_configurations and new_makespan <= original_makespan * 1.2:
            seen_configurations.add(config_hash)
            neighbors.append(new_neighbor)
    
    return neighbors