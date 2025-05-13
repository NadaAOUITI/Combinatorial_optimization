from models.machine import Machine

class SPTHeuristic:
    def optimize(self, instance):
        machines = [Machine(i + 1) for i in range(instance.num_machines)]
        remaining_jobs = instance.jobs[:]  # Copy of all jobs
        job_schedule = []

        mold_release_times = {}
        machine_release_times = [0] * instance.num_machines

        while remaining_jobs:
            # Step 1: Get the shortest job duration among remaining jobs
            min_duration = min(job.duration for job in remaining_jobs)

            # Step 2: Filter all jobs with that duration
            shortest_jobs = [job for job in remaining_jobs if job.duration == min_duration]

            # Step 3: Among them, choose the one with earliest mold availability
            best_job = None
            best_mold_time = float('inf')

            for job in shortest_jobs:
                mold_time = mold_release_times.get(job.mold, 0)
                if mold_time < best_mold_time:
                    best_job = job
                    best_mold_time = mold_time

            job = best_job
            remaining_jobs.remove(job)

            # Step 4: Assign to earliest available machine
            earliest_machine_idx = machine_release_times.index(min(machine_release_times))
            earliest_machine = machines[earliest_machine_idx]

            start_time = max(machine_release_times[earliest_machine_idx], mold_release_times.get(job.mold, 0))
            end_time = start_time + job.duration

            earliest_machine.assign_job(job)
            machine_release_times[earliest_machine_idx] = end_time
            mold_release_times[job.mold] = end_time

            job_schedule.append({
                'job_id': job.job_id,
                'machine_id': earliest_machine.machine_id,
                'start_time': start_time,
                'end_time': end_time,
                'mold_id': job.mold,
                'duration': job.duration
            })

        makespan = max(job['end_time'] for job in job_schedule)
        return machines, job_schedule, makespan
