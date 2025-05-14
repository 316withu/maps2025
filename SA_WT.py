import pandas as pd
import random
import copy
import math
import matplotlib.pyplot as plt
import os
import time

plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

# CSV 파일로부터 작업 데이터를 읽어옵니다.
df = pd.read_csv("jobs_50_weight.csv")

# 작업 정보를 담는 Job 클래스 정의
class Job:
    def __init__(self, id, processing_time, due_date, weight):
        self.id = id
        self.processing_time = processing_time
        self.due_date = due_date
        self.weight = weight
    def __repr__(self):
        return str(self.id)

# CSV의 각 행을 Job 객체로 변환
job_list = [Job(row['id'], row['processing_time'], row['due_date'], row['weight']) for _, row in df.iterrows()]

# 머신 수
NUM_MACHINES = 3

# 염색체 클래스 정의
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
        if random.random() < 0.5:
            idx1, idx2 = random.sample(range(len(self.chromosome)), 2)
            self.chromosome[idx1], self.chromosome[idx2] = self.chromosome[idx2], self.chromosome[idx1]
        else:
            idx = random.randrange(len(self.chromosome))
            job, current_machine = self.chromosome[idx]
            new_machine = random.choice([m for m in range(self.num_machines) if m != current_machine])
            self.chromosome[idx] = (job, new_machine)
        return self.chromosome

def simulated_annealing(job_list, num_machines, initial_temp=5000, cooling_rate=0.99, max_iter=5000, temp_min=1, start_time=None, time_limit=None):
    current = Chromosome(job_list, num_machines)
    current.schedule_jobs()
    current.calculate_total_tardiness()
    best = copy.deepcopy(current)

    history = []
    T = initial_temp
    iter_count = 0

    while T > temp_min and iter_count < max_iter:
        if time_limit is not None and start_time is not None:
            if time.time() - start_time > time_limit:
                print(f"[⏰] 시간 제한 {time_limit}초 도달로 조기 종료합니다.")
                break

        neighbor = copy.deepcopy(current)
        neighbor.mutate()
        neighbor.schedule_jobs()
        neighbor.calculate_total_tardiness()

        delta = neighbor.total_tardiness - current.total_tardiness

        if delta < 0 or random.random() < math.exp(-delta / T):
            current = neighbor

        if current.total_tardiness < best.total_tardiness:
            best = copy.deepcopy(current)

        history.append(best.total_tardiness)
        T *= cooling_rate
        iter_count += 1

        if iter_count % 100 == 0:
            print(f"Iteration {iter_count}: Best Weighted Tardiness = {best.total_tardiness}")

    return best, history, iter_count

def export_decision_log_excel_sa(best_solution, output="sa_decision_log.xlsx"):
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
    print(f"[Excel] SA 의사결정 로그 저장 완료 → {output}")

if __name__ == "__main__":
    random.seed(42)

    TIME_LIMIT = 120
    start_time = time.time()

    best_solution, history, iter_count = simulated_annealing(
        job_list, NUM_MACHINES,
        initial_temp=50000,
        cooling_rate=0.999,
        max_iter=5000000,
        temp_min=0.1,
        start_time=start_time,
        time_limit=TIME_LIMIT
    )

    end_time = time.time()
    elapsed_time = end_time - start_time

    print("\nTotal Weighted Tardiness:", best_solution.total_tardiness)
    print('최종반복수:', iter_count)
    print(f"실행 시간: {elapsed_time:.4f} 초")

    print("각 머신별 스케줄 (id 순서):")
    if best_solution.schedule is None:
        best_solution.schedule_jobs()
    for m in range(NUM_MACHINES):
        print(f"Machine {m}: {[job.id for job in best_solution.schedule[m]]}")

    plt.figure(figsize=(8, 5))
    plt.plot(range(1, len(history) + 1), history, marker='o')
    plt.xlabel("Iteration")
    plt.ylabel("Best Weighted Tardiness")
    plt.title("Iteration별 가중 지연시간 (SA 성능)")
    plt.grid(True)
    plt.show()

    assignment_list = [[job.id, machine] for job, machine in best_solution.chromosome]
    assignment_list = list(map(list, zip(*assignment_list)))
    print("\n최종 최적해 작업/머신 할당 이중 리스트 (전치된 형태):")
    print(assignment_list)

    fig, ax = plt.subplots(figsize=(max(8, len(assignment_list[0]) * 1.2), 3))
    ax.axis('tight')
    ax.axis('off')
    table = ax.table(cellText=assignment_list, cellLoc='center', loc='center')
    plt.title("최종 최적해 염색체 구조 (SA)")
    plt.show()

    export_decision_log_excel_sa(best_solution, output="sa_decision_log.xlsx")
