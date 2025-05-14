import pandas as pd
import math
import numpy as np
import random
import os

# ========== 기본 파라미터 ========= #
NUM_MACHINES = 3
THETA = 0.1

df = pd.read_csv("jobs_50_weight.csv")

class Job:
    def __init__(self, id, processing_time, due_date, weight):
        self.id = id
        self.processing_time = processing_time
        self.due_date = due_date
        self.weight = weight

job_list = []
for _, row in df.iterrows():
    job_list.append(Job(
        id=row["id"],
        processing_time=row["processing_time"],
        due_date=row["due_date"],
        weight=row["weight"]
    ))

n_jobs = len(job_list)

def compute_DUI(job, current_time, p_bar, theta):
    w = job.weight
    p = job.processing_time
    d = job.due_date
    slack = d - current_time - p
    numerator = w / p if p > 0 else 0.0001
    exponent = - max(0, slack) / (theta * p_bar)
    return numerator * math.exp(exponent)

p_bar = np.mean([job.processing_time for job in job_list])

machine_next_time = [0] * NUM_MACHINES
scheduled = [False] * n_jobs
schedule_result = []

for _ in range(n_jobs):
    machine_idx = np.argmin(machine_next_time)
    current_time = machine_next_time[machine_idx]
    best_job_idx = None
    best_dui = -1
    for j_idx, job in enumerate(job_list):
        if not scheduled[j_idx]:
            dui_val = compute_DUI(job, current_time, p_bar, THETA)
            if dui_val > best_dui:
                best_dui = dui_val
                best_job_idx = j_idx
    chosen_job = job_list[best_job_idx]
    start_time = current_time
    finish_time = start_time + chosen_job.processing_time
    schedule_result.append((machine_idx, chosen_job.id, start_time, finish_time))
    machine_next_time[machine_idx] = finish_time
    scheduled[best_job_idx] = True

job_dict = {j.id: j for j in job_list}
info = []
for (m_id, j_id, st, ft) in schedule_result:
    job_obj = job_dict[j_id]
    tardiness = max(0, ft - job_obj.due_date)
    wt = job_obj.weight * tardiness
    info.append((j_id, st, ft, wt))

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

output_path = "dui_decision_log.xlsx"
pd.DataFrame(rows).to_excel(output_path, index=False)

print(f"\n[DUI Scheduling Result]")
print(f" - 총 작업 수: {n_jobs}")
print(f" - 총 가중 지각(Weighted Tardiness): {total}")
print(f"[Excel] DUI 의사결정 로그 저장 완료 → {output_path}")
