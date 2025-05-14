import pandas as pd
import random
import copy
import math
import matplotlib.pyplot as plt
import os

# (필요 시) 한글 폰트
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

# =====================
# 1. 난수 시드 고정
# =====================
SEED = 42
random.seed(SEED)
# 만약 np.random을 활용한다면 아래도 추가:
# import numpy as np
# np.random.seed(SEED)

# ========== 2. CSV에서 작업 데이터 로드 (id, processing_time, due_date, weight) ========== #
df = pd.read_csv("jobs_50_weight.csv")

class Job:
    def __init__(self, id, processing_time, due_date, weight):
        self.id = id
        self.processing_time = processing_time
        self.due_date = due_date
        self.weight = weight  # 추가

    def __repr__(self):
        return f"Job(id={self.id}, weight={self.weight})"

# Job 리스트
job_list = [
    Job(row['id'], row['processing_time'], row['due_date'], row['weight'])
    for _, row in df.iterrows()
]

# 병렬 기계 수
NUM_MACHINES = 3

# ========== 3. 염색체(Chromosome) 클래스: (job, machine) 쌍으로 순서 표현 ========== #
class Chromosome:
    def __init__(self, job_list, num_machines):
        self.job_list = job_list
        self.num_machines = num_machines
        # 무작위 순서 + 무작위 머신 할당
        shuffled_jobs = random.sample(self.job_list, len(self.job_list))
        self.chromosome = [
            (job, random.choice(range(self.num_machines)))
            for job in shuffled_jobs
        ]
        self.schedule = None
        self.total_tardiness = None  # Weighted Tardiness 저장

    def schedule_jobs(self):
        """(job, machine) 리스트를 머신별로 분류"""
        self.schedule = {m: [] for m in range(self.num_machines)}
        for job, machine in self.chromosome:
            self.schedule[machine].append(job)
        return self.schedule

    def calculate_weighted_tardiness(self):
        """각 머신을 순차적으로 처리했을 때의 가중 지각 계산"""
        if self.schedule is None:
            self.schedule_jobs()

        total_wt = 0
        for m in range(self.num_machines):
            time = 0
            for job in self.schedule[m]:
                time += job.processing_time
                tardiness = max(0, time - job.due_date)
                total_wt += (job.weight * tardiness)
        self.total_tardiness = total_wt
        return total_wt

    def mutate(self):
        """
        간단한 돌연변이:
        (1) 두 작업 순서를 맞바꾸거나,
        (2) 한 작업의 머신 할당을 바꿈
        """
        if random.random() < 0.5:
            # (1) 두 작업 순서 교환
            idx1, idx2 = random.sample(range(len(self.chromosome)), 2)
            self.chromosome[idx1], self.chromosome[idx2] = \
                self.chromosome[idx2], self.chromosome[idx1]
        else:
            # (2) 머신 할당 변경
            idx = random.randrange(len(self.chromosome))
            job, current_machine = self.chromosome[idx]
            new_machine = random.choice(
                [m for m in range(self.num_machines) if m != current_machine]
            )
            self.chromosome[idx] = (job, new_machine)
        return self.chromosome

# ========== 4. PSO 알고리즘 (Discrete) ========== #
def particle_swarm_optimization(
    job_list,
    num_machines,
    swarm_size=30,
    max_iter=1000,
    c1=0.5,
    c2=0.5,
    random_rate=0.1
):
    """
    - swarm: 여러 Chromosome(입자)
    - personal_best, global_best
    - (작업 순서 + 머신 할당)을 PSO-like 방식으로 업데이트
    - 목적함수: Weighted Tardiness
    """
    # 초기 swarm
    swarm = []
    for _ in range(swarm_size):
        particle = Chromosome(job_list, num_machines)
        particle.schedule_jobs()
        particle.calculate_weighted_tardiness()
        # personal_best 초기화
        particle.personal_best = copy.deepcopy(particle)
        swarm.append(particle)

    # global_best 초기화
    global_best = min(swarm, key=lambda p: p.total_tardiness)
    history = [global_best.total_tardiness]

    # PSO 메인 루프
    for iter_count in range(max_iter):
        for particle in swarm:
            new_chromosome = copy.deepcopy(particle.chromosome)

            # ---- (A) 글로벌 베스트 방향 업데이트 ---- #
            for i in range(len(new_chromosome)):
                # (1) 작업 순서 swap
                if random.random() < c2:
                    target_job = global_best.chromosome[i][0]
                    # target_job 위치 찾기
                    cur_idx = next(
                        j for j, (jb, mac) in enumerate(new_chromosome)
                        if jb.id == target_job.id
                    )
                    # swap
                    new_chromosome[i], new_chromosome[cur_idx] = \
                        new_chromosome[cur_idx], new_chromosome[i]
                # (2) 머신 할당 동기화
                if random.random() < c2:
                    new_machine = global_best.chromosome[i][1]
                    job, _ = new_chromosome[i]
                    new_chromosome[i] = (job, new_machine)

            # ---- (B) 개인 베스트 방향 업데이트 ---- #
            for i in range(len(new_chromosome)):
                if random.random() < c1:
                    target_job = particle.personal_best.chromosome[i][0]
                    cur_idx = next(
                        j for j, (jb, mac) in enumerate(new_chromosome)
                        if jb.id == target_job.id
                    )
                    new_chromosome[i], new_chromosome[cur_idx] = \
                        new_chromosome[cur_idx], new_chromosome[i]
                if random.random() < c1:
                    new_machine = particle.personal_best.chromosome[i][1]
                    job, _ = new_chromosome[i]
                    new_chromosome[i] = (job, new_machine)

            # ---- (C) 랜덤 돌연변이 ---- #
            if random.random() < random_rate:
                if len(new_chromosome) >= 2:
                    idx1, idx2 = random.sample(range(len(new_chromosome)), 2)
                    new_chromosome[idx1], new_chromosome[idx2] = \
                        new_chromosome[idx2], new_chromosome[idx1]

            # 업데이트 적용
            particle.chromosome = new_chromosome
            particle.schedule_jobs()
            particle.calculate_weighted_tardiness()

            # personal_best 갱신
            if particle.total_tardiness < particle.personal_best.total_tardiness:
                particle.personal_best = copy.deepcopy(particle)

        # global_best 갱신
        current_best = min(swarm, key=lambda p: p.total_tardiness)
        if current_best.total_tardiness < global_best.total_tardiness:
            global_best = copy.deepcopy(current_best)

        history.append(global_best.total_tardiness)

        # 중간 진행상황
        if iter_count % 100 == 0:
            print(f"Iteration {iter_count}: Global Best Weighted Tardiness = {global_best.total_tardiness}")

    return global_best, history

# ========== 5. 결과 저장 함수 ========== #

def export_decision_log_excel_pso(best_solution, output="pso_decision_log.xlsx"):
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

    info.sort(key=lambda x: x[1])  # 시작시간 기준 정렬

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
    print(f"📄 PSO 의사결정 로그 저장 완료 → {output}")

# ========== 6. 메인 실행: PSO 수행  ========== #
if __name__ == "__main__":

    # -- 시드 고정 --
    SEED = 42
    random.seed(SEED)
    # np.random.seed(SEED)  # (NumPy 활용 시)

    best_solution, history = particle_swarm_optimization(
        job_list,
        NUM_MACHINES,
        swarm_size=30,
        max_iter=1000,
        c1=0.5,
        c2=0.5,
        random_rate=0.1
    )

    print("\nPSO 최종 최적해의 가중 지연시간(Weighted Tardiness):", best_solution.total_tardiness)

    export_decision_log_excel_pso(best_solution, output="pso_decision_log.xlsx")