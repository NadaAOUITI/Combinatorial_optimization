# idle_filling.py
from models.machine import Machine

class IdleFillingHeuristic:
    """
    At each step, pick the (job, machine) pair that
    - can start the earliest (minimizes machine idle time)
    - subject to mold availability
    Tie-break by earliest finish time, then by machine load.
    """
    def optimize(self, instance):
        machines = [Machine(i+1) for i in range(instance.num_machines)]
        jobs = instance.jobs.copy()
        job_schedule = []

        mold_release = {}                       # mold → time when mold is free
        machine_release = [0]*instance.num_machines  # machine → time when machine is free

        while jobs:
            best = None
            # We will choose (job, machine_idx, start, end, idle_gap)
            best_tuple = (float('inf'),   # idle_gap
                          float('inf'),   # end_time
                          float('inf'))   # machine_release (for tie-break)

            for job in jobs:
                mold_free = mold_release.get(job.mold, 0)
                for idx in range(instance.num_machines):
                    m_free = machine_release[idx]
                    # if mold not free until later, machine must wait:
                    start = max(m_free, mold_free)
                    idle_gap = start - m_free
                    end = start + job.duration

                    candidate = (idle_gap, end, m_free)
                    if candidate < best_tuple:
                        best_tuple = candidate
                        best = (job, idx, start, end)

            # assign best
            job, mi, st, et = best
            machines[mi].assign_job(job)
            machine_release[mi] = et
            mold_release[job.mold] = et
            jobs.remove(job)

            job_schedule.append({
                'job_id':    job.job_id,
                'machine_id':machines[mi].machine_id,
                'start_time':st,
                'end_time':  et,
                'mold_id':   job.mold,
                'duration':  job.duration
            })

        makespan = max(j['end_time'] for j in job_schedule)
        return machines, job_schedule, makespan
