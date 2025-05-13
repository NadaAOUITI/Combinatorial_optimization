class Machine:
    def __init__(self, machine_id):
        self.machine_id = machine_id
        self.current_jobs = []  # Jobs assigned to this machine
        self.total_time = 0  # Total execution time of assigned jobS

    def assign_job(self, job):
        """Assign a job to this machine"""
        self.current_jobs.append(job)
        self.total_time += job.duration

    def __repr__(self):
        return f"Machine(ID={self.machine_id}, Jobs={self.current_jobs}, TotalTime={self.total_time})"
