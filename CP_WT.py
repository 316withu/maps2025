import pandas as pd
import random
import copy
import matplotlib.pyplot as plt
import time
from ortools.sat.python import cp_model

random.seed(42)

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
n_jobs = len(job_list)
H = 100000

model = cp_model.CpModel()

presence_vars = [[model.NewBoolVar(f"presence_m{m}_j{j}") for j in range(n_jobs)] for m in range(NUM_MACHINES)]
start_vars = [[model.NewIntVar(0, H, f"start_m{m}_j{j}") for j in range(n_jobs)] for m in range(NUM_MACHINES)]
end_vars = [[model.NewIntVar(0, H, f"end_m{m}_j{j}") for j in range(n_jobs)] for m in range(NUM_MACHINES)]

processing_itv_vars = [
    [model.NewOptionalIntervalVar(start=start_vars[m][j], end=end_vars[m][j], size=job_list[j].processing_time,
                                  is_present=presence_vars[m][j], name=f"itv_m{m}_j{j}")
     for j in range(n_jobs)] for m in range(NUM_MACHINES)]

for m in range(NUM_MACHINES):
    model.AddNoOverlap(processing_itv_vars[m])

for j in range(n_jobs):
    model.Add(sum(presence_vars[m][j] for m in range(NUM_MACHINES)) == 1)

# ✅ Weighted Tardiness 목적함수 구현
wt_vars = []
for j in range(n_jobs):
    job_end = model.NewIntVar(0, H, f"job{j}_end")
    model.AddMaxEquality(job_end, [end_vars[m][j] for m in range(NUM_MACHINES)])

    tardiness = model.NewIntVar(0, H, f"tardiness_j{j}")
    model.Add(tardiness >= job_end - job_list[j].due_date)
    model.Add(tardiness >= 0)

    weighted_tardiness = model.NewIntVar(0, H * job_list[j].weight, f"wt_j{j}")
    model.AddMultiplicationEquality(weighted_tardiness, [tardiness, job_list[j].weight])

    wt_vars.append(weighted_tardiness)

model.Minimize(sum(wt_vars))

solver = cp_model.CpSolver()
solver.parameters.max_time_in_seconds = 300
status = solver.Solve(model)

if status in [cp_model.OPTIMAL, cp_model.FEASIBLE]:
    job_schedule = []
    for j in range(n_jobs):
        for m in range(NUM_MACHINES):
            if solver.Value(presence_vars[m][j]) == 1:
                start_time = solver.Value(start_vars[m][j])
                end_time = solver.Value(end_vars[m][j])
                job_schedule.append((j + 1, m + 1, start_time, end_time, job_list[j].weight,
                                     max(0, end_time - job_list[j].due_date)))

    job_schedule.sort(key=lambda x: x[2])

    rows, total = [], 0
    remaining = [job[0] for job in job_schedule]
    for d, (jid, mid, st, ft, wt, tardy) in enumerate(job_schedule, 1):
        total += wt * tardy
        rows.append({
            "decision": d,
            "job_available": remaining.copy(),
            "job_chosen": jid,
            "WT": wt * tardy,
            "total_WT": total
        })
        remaining.remove(jid)

    pd.DataFrame(rows).to_excel("CP_decision_log.xlsx", index=False)
    print("[Excel] CP 의사결정 로그 저장 완료 → CP_decision_log.xlsx")
else:
    print("❌ 해를 찾지 못했습니다.")
###