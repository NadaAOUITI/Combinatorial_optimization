
from .base import Metaheuristic
import copy, math, random
from tqdm import tqdm
from algorithms.heuristics.IdleFillingHeuristic import IdleFillingHeuristic
from algorithms.metaheuristics.neighborsgeneration import generate_neighbor, calculate_makespan, schedule_to_machines
import matplotlib.pyplot as plt
from algorithms.heuristics.lpt import LPTHeuristic
from algorithms.heuristics.spt import SPTHeuristic

class SimulatedAnnealing(Metaheuristic):
    def __init__(self, initial_temp=700, cooling_rate=0.99, min_temp=1, max_iterations=500, class_id=1, enable_hypertuning=True):
        self.initial_temp = initial_temp
        self.cooling_rate = cooling_rate
        self.min_temp = min_temp
        self.max_iterations = max_iterations
        self.class_id = class_id  # Store class_id for hypertuning
        self.enable_hypertuning = enable_hypertuning  # Flag to enable/disable hypertuning
        
        # For tracking temperature vs makespan
        self.temp_history = []
        self.makespan_history = []

    def optimize(self, instance):
        # Apply hypertuning only if enabled
        if self.enable_hypertuning:
            self.hypertune_parameters(instance)
            
        idlefilling = IdleFillingHeuristic()
        machines, best_schedule, best_makespan = idlefilling.optimize(instance)
        current_schedule = copy.deepcopy(best_schedule)
        current_makespan = best_makespan
        
        # Record initial state
        self.temp_history.append(self.initial_temp)
        self.makespan_history.append(best_makespan)
        
        temperature = self.initial_temp
        # To see if my neighbors are useful
        makespan_counts = {}  # Dictionary to count makespan frequencies
        progress_bar = tqdm(desc="Simulated Annealing", total=int(math.log(self.min_temp/self.initial_temp, self.cooling_rate)))

        #------------------------adding strategic restart to avoid local minima
        """no_improvement_count = 0
        max_no_improvement = 900  """
        # -----------------------------strategic restart------------------

        while temperature > self.min_temp:
            iterations = 0
            while iterations < self.max_iterations:
                neighbor_schedule = generate_neighbor(current_schedule, instance.num_machines)
                neighbor_makespan = calculate_makespan(neighbor_schedule)
                
                # Count makespan frequencies
                if neighbor_makespan in makespan_counts:
                    makespan_counts[neighbor_makespan] += 1
                else:
                    makespan_counts[neighbor_makespan] = 1

                delta = neighbor_makespan - current_makespan

                if delta < 0 or random.random() < math.exp(-delta / temperature):
                    current_schedule = copy.deepcopy(neighbor_schedule)
                    current_makespan = neighbor_makespan
                    if current_makespan < best_makespan:
                        best_schedule = copy.deepcopy(current_schedule)
                        best_makespan = current_makespan
                        # Record when we find a new best solution
                        self.temp_history.append(temperature)
                        self.makespan_history.append(best_makespan)
                    #-----restart logic--------
                    """    no_improvement_count = 0
                    else:
                        no_improvement_count += 1"""
                    #------End of --------------


                iterations += 1
            
            # Record temperature and current best makespan at each cooling step
            self.temp_history.append(temperature)
            self.makespan_history.append(best_makespan)
                
            temperature *= self.cooling_rate
            # 🔁 Strategic Restart: Reset current state if stagnation
            """if no_improvement_count >= max_no_improvement:
                print("🔄 Restart triggered: No improvement in recent iterations.")
                machines, current_schedule, current_makespan = idlefilling.optimize(instance)
                no_improvement_count = 0  # reset counter"""
            #------------------end of restart logic------------------
            progress_bar.update(1)
        
        progress_bar.close()

        # Display makespan frequencies
        print("\n📊 Makespan Frequencies Across All Neighbors:")
        for mk, count in sorted(makespan_counts.items()):
            print(f"Makespan {mk} : {count} neighbors")

        best_machines = schedule_to_machines(best_schedule, instance.num_machines, instance.jobs)
        
        return best_machines, best_schedule, best_makespan
    
    def plot_temperature_vs_makespan(self, save_path=None):
        
        plt.figure(figsize=(10, 6))
        plt.plot(self.temp_history, self.makespan_history, 'b-', marker='o')
        plt.xlabel('Temperature')
        plt.ylabel('Best Makespan')
        plt.title('Temperature vs. Best Makespan in Simulated Annealing')
        plt.xscale('log')  # Log scale for temperature
        plt.grid(True)
        
        if save_path:
            plt.savefig(save_path)
        
        plt.show()

    def hypertune_parameters(self, instance):
        # Apply hypertuning for unifrom distribution classes--------------------------------------------------
        if self.class_id in [1,2,3]:
            self.initial_temp = 500.0
            if instance.num_molds<5:            
                self.cooling_rate = 0.9
                if instance.num_jobs<200:
                    self.max_iterations =100.0
                    self.min_temp = 0.1
                else:
                    self.max_iterations =500.0
                    self.min_temp = 5.0
            else:
                self.cooling_rate = 0.97
                self.max_iterations =300.0
                self.min_temp = 1.0
        # Apply hypertuning for bimodal distribution classes--------------------------------------------------
        else :
            if instance.num_molds<5:
                self.initial_temp = 2000.0
                self.max_iterations =100.0
                if instance.num_jobs<200:   
                    self.cooling_rate = 0.95    
                    self.min_temp = 0.1
                else:#jobs are bigger,molds are the same
                    self.cooling_rate = 0.9
                    self.min_temp = 1.0
            else :
                self.initial_temp = 500.0
                self.cooling_rate = 0.97
                self.max_iterations =700.0
                self.min_temp = 1.0
