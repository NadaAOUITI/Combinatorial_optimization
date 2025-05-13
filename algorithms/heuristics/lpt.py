# lpt.py
from models.machine import Machine

class LPTHeuristic:
    """
    Longest Processing Time first, with tie-breakers:
    - If durations are equal, choose job with available mold first.
    - If both molds are available, choose the one with earliest mold release.
    """
    def optimize(self, instance):
        machines = [Machine(i + 1) for i in range(instance.num_machines)]
        jobs = instance.jobs.copy()
        job_schedule = []

        mold_release_times = {}
        machine_release_times = [0] * instance.num_machines

        # Sort jobs primarily by descending duration
        # We'll do tie-breaking during scheduling, not here
        jobs.sort(key=lambda job: job.duration, reverse=True)

        while jobs:
            best_job = None
            best_priority = None

            for job in jobs:
                mold_release = mold_release_times.get(job.mold, 0)

                # Priority logic for tie-breaking
                priority = (
                    job.duration,                    # First: longer duration
                    job.mold in mold_release_times,  # Second: mold is available now
                    -mold_release                    # Third: mold becomes available earlier
                )

                if best_priority is None or priority > best_priority:
                    best_priority = priority
                    best_job = job

            # Choose best job and assign to earliest free machine
            earliest_machine_idx = machine_release_times.index(min(machine_release_times))
            start_time = machine_release_times[earliest_machine_idx]
            mold_ready_time = mold_release_times.get(best_job.mold, 0)
            start_time = max(start_time, mold_ready_time)

            end_time = start_time + best_job.duration
            chosen_machine = machines[earliest_machine_idx]
            chosen_machine.assign_job(best_job)

            machine_release_times[earliest_machine_idx] = end_time
            mold_release_times[best_job.mold] = end_time
            jobs.remove(best_job)

            job_schedule.append({
                'job_id': best_job.job_id,
                'machine_id': chosen_machine.machine_id,
                'start_time': start_time,
                'end_time': end_time,
                'mold_id': best_job.mold,
                'duration': best_job.duration
            })

        makespan = max(job['end_time'] for job in job_schedule)
        return machines, job_schedule, makespan
