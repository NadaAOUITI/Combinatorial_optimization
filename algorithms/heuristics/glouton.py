from models.machine import Machine

class GreedyHeuristic:
    def optimize(self, instance):
        machines = [Machine(i + 1) for i in range(instance.num_machines)]
        jobs = instance.jobs.copy()  # Copy the list to avoid modifying the original
        job_schedule = []

        mold_release_times = {}
        machine_release_times = [0] * instance.num_machines

        while jobs:
            best_choice = None
            best_end_time = float('inf')

            for job in jobs:
                for idx, machine in enumerate(machines):
                    # Find when the machine will be available
                    start_time = machine_release_times[idx]
                    
                    # Ensure that the mold is available for the job (mold conflict check)
                    if job.mold in mold_release_times:
                        mold_available_time = mold_release_times[job.mold]
                        start_time = max(start_time, mold_available_time)

                    end_time = start_time + job.duration

                    # Update the best choice based on minimum end time
                    if best_choice is None or end_time < best_end_time or (
                        end_time == best_end_time and machine_release_times[idx] < machine_release_times[best_choice[1]]
                    ):
                        best_choice = (job, idx, start_time, end_time)
                        best_end_time = end_time

            # Once the best job and machine are chosen
            job, machine_idx, start_time, end_time = best_choice
            chosen_machine = machines[machine_idx]

            # Assign the job to the machine and update relevant times
            chosen_machine.assign_job(job)
            machine_release_times[machine_idx] = end_time
            mold_release_times[job.mold] = end_time  # Update the mold's availability

            # Remove the assigned job from the list
            jobs.remove(job)

            # Record the job's schedule details
            job_schedule.append({
                'job_id': job.job_id,
                'machine_id': chosen_machine.machine_id,
                'start_time': start_time,
                'end_time': end_time,
                'mold_id': job.mold,
                'duration': job.duration
            })

        # Calculate the makespan (maximum completion time)
        makespan = max(job['end_time'] for job in job_schedule)
        return machines, job_schedule, makespan
