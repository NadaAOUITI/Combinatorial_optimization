class Job:
    def __init__(self, job_id, duration, mold):
        self.job_id = job_id
        self.duration = duration
        self.mold = mold

    def __repr__(self):
        return f"  Job ID: {self.job_id} | Duration: {self.duration} | Mold: {self.mold}\n"