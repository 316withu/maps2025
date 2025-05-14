import pandas as pd
import random
import copy
import matplotlib.pyplot as plt
import time

random.seed(42)

plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

df = pd.read_csv("jobs_50_weight.csv")

class Job:
    def __init__(self, id, processing_time, due_date, weight):
        self.id = id
        self.processing_time = processing_time
        self.due_date = due_date
        self.weight = weight

    def __repr__(self):
        return str(self.id)

job_list = [Job(row['id'], row['processing_time'], row['due_date'], row['weight']) for _, row in df.iterrows()]

NUM_MACHINES = 3

class Chromosome:
    def __init__(self, job_list, num_machines):
        self.job_list = job_list
        self.num_machines = num_machines
        shuffled_jobs = random.sample(self.job_list, len(self.job_list))
        self.chromosome = [(job, random.choice(range(self.num_machines))) for job in shuffled_jobs]
        self.schedule = None
        self.total_tardiness = None

    def schedule_jobs(self):
        self.schedule = {m: [] for m in range(self.num_machines)}
        for job, machine in self.chromosome:
            self.schedule[machine].append(job)
        return self.schedule

    def calculate_total_tardiness(self):
        if self.schedule is None:
            self.schedule_jobs()
        total = 0
        for m in range(self.num_machines):
            time = 0
            for job in self.schedule[m]:
                time += job.processing_time
                tardiness = max(0, time - job.due_date)
                total += tardiness * job.weight
        self.total_tardiness = total
        return total

    def mutate(self):
        idx1, idx2 = random.sample(range(len(self.chromosome)), 2)
        self.chromosome[idx1], self.chromosome[idx2] = self.chromosome[idx2], self.chromosome[idx1]
        return self.chromosome

def two_point_crossover_operator(parent1, parent2):
    size = len(parent1.chromosome)
    p1, p2 = sorted(random.sample(range(size), 2))
    child_chromosome = [None] * size
    for i in range(p1, p2 + 1):
        child_chromosome[i] = parent1.chromosome[i]

    def get_ids(chromo):
        ids = set()
        for item in chromo:
            if item is not None:
                job, _ = item
                ids.add(job.id)
        return ids

    parent2_index = (p2 + 1) % size
    current_index = (p2 + 1) % size
    while None in child_chromosome:
        candidate = parent2.chromosome[parent2_index]
        if candidate[0].id not in get_ids(child_chromosome):
            child_chromosome[current_index] = candidate
            current_index = (current_index + 1) % size
        parent2_index = (parent2_index + 1) % size

    child = copy.deepcopy(parent1)
    child.chromosome = child_chromosome
    return child

def genetic_algorithm(job_list, num_machines, population_size=40, generations=20, elite_ratio=0.2,
                      crossover_rate=0.6, mutation_rate=0.6):
    population = [Chromosome(job_list, num_machines) for _ in range(population_size)]
    for chrom in population:
        chrom.schedule_jobs()
        chrom.calculate_total_tardiness()

    best_history = []

    for gen in range(generations):
        elite_count = int(elite_ratio * population_size)
        elite = sorted(population, key=lambda chrom: chrom.total_tardiness)[:elite_count]

        new_population = elite.copy()
        while len(new_population) < population_size:
            parent1, parent2 = random.sample(population, 2)
            if random.random() < crossover_rate:
                child = two_point_crossover_operator(parent1, parent2)
            else:
                child = copy.deepcopy(parent1)
            if random.random() < mutation_rate:
                child.mutate()
            child.schedule_jobs()
            child.calculate_total_tardiness()
            new_population.append(child)
        population = new_population

        best = min(population, key=lambda chrom: chrom.total_tardiness)
        best_history.append(best.total_tardiness)
        print(f"Generation {gen + 1}: Best Total Tardiness = {best.total_tardiness}")

    return population, best_history

def export_decision_log_excel_ga(best_solution, output="ga_decision_log.xlsx"):
    info = []
    for m, job_list in best_solution.schedule.items():
        time = 0
        for job in job_list:
            start = time
            end = start + job.processing_time
            tardiness = max(0, end - job.due_date)
            wt = tardiness * job.weight
            info.append((job.id, start, end, wt))
            time = end

    info.sort(key=lambda x: x[1])

    remaining = [row[0] for row in info]
    rows, total = [], 0
    for d, (jid, _, _, wt) in enumerate(info, 1):
        total += wt
        rows.append({
            "decision": d,
            "job_available": remaining.copy(),
            "job_chosen": jid,
            "WT": int(wt),
            "total_WT": int(total)
        })
        remaining.remove(jid)

    pd.DataFrame(rows).to_excel(output, index=False)
    print(f"[Excel] GA 의사결정 로그 저장 완료 → {output}")

if __name__ == "__main__":
    start_time = time.perf_counter()

    best_population, best_history = genetic_algorithm(job_list, NUM_MACHINES, population_size=200, generations=400,
                                                      elite_ratio=0.2, crossover_rate=0.5, mutation_rate=0.1)
    best_solution = min(best_population, key=lambda chrom: chrom.total_tardiness)

    print("\n최종 최적해의 총 가중 지연시간 (Total Weighted Tardiness):", best_solution.total_tardiness)
    print("각 머신별 스케줄 (id 순서):")
    if best_solution.schedule is None:
        best_solution.schedule_jobs()
    for m in range(NUM_MACHINES):
        print(f"Machine {m}: {[job.id for job in best_solution.schedule[m]]}")

    end_time = time.perf_counter()
    elapsed_time = end_time - start_time
    print(f"실행시간 : {elapsed_time:.2f}")


    export_decision_log_excel_ga(best_solution, output="ga_decision_log.xlsx")
