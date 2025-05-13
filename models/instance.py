class Instance:
    def __init__(self, num_jobs,num_molds,  job_class, instance_number, jobs):
        self.num_jobs = num_jobs
        self.num_molds= num_molds
        self.num_machines = 2  # FIXED: This is machines, not molds
        self.job_class = job_class
        self.instance_number = instance_number
        self.jobs = jobs  # List of Job objects

    def __repr__(self):
        job_str = "\n".join(str(job) for job in self.jobs)
        return (f"\n🔹 Instance Information:\n"
                f"   Number of Jobs     : {self.num_jobs}\n"
                f"   Number of Machines : {self.num_machines}\n"
                f"   Job Class          : {self.job_class}\n"
                f"   Instance Number    : {self.instance_number}\n"
                f"   Jobs List:\n{job_str}"
                f" Number of Molds : {self.num_molds}\n")
               