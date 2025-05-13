# File: utils/file_io.py

from models.instance import Instance
from models.job import Job
from typing import List

def read_all_instances(filename: str) -> List[str]:
    """
    Reads the raw contents of a file and splits it into raw instance blocks.
    Assumes that each instance is separated by a blank line.
    
    Args:
        filename (str): Path to the file containing instances.
        
    Returns:
        List[str]: A list of raw instance data blocks as strings.
    """
    with open(filename, 'r') as file:
        raw_instances = file.read().strip().split("\n\n")
    return raw_instances


def parse_all_instances(filename: str) -> List[Instance]:
    """
    Robust parser for mold-constrained scheduling instances.
    Header format: N M Class InstanceNumber
    Where:
    - N = number of jobs
    - M = number of molds (not machines)
    """
    raw_instances = read_all_instances(filename)
    instances = []

    for raw_instance in raw_instances:
        lines = [line.strip() for line in raw_instance.split('\n') if line.strip()]
        
        # Parse header (first line)
        header = lines[0].split()
        num_jobs = int(header[0])
        num_molds = int(header[1])  # CHANGED FROM num_machines
        job_class = int(header[2])
        instance_number = int(header[3])

        # Process all numerical data
        all_numbers = []
        for line in lines[1:]:
            all_numbers.extend(map(int, line.split()))
        
        if len(all_numbers) < 2 * num_jobs:
            raise ValueError(f"Instance {instance_number} has insufficient data")

        durations = all_numbers[:num_jobs]
        molds = all_numbers[num_jobs:2*num_jobs]
        
        # Validate mold numbers don't exceed num_molds
        if max(molds) > num_molds:
            raise ValueError(f"Instance {instance_number} contains mold {max(molds)} but only {num_molds} molds declared")

        jobs = [Job(i+1, durations[i], molds[i]) for i in range(num_jobs)]
        instance = Instance(num_jobs, num_molds, job_class, instance_number, jobs)  # CHANGED
        instances.append(instance)

    return instances