import pandas as pd

# Load job data
job_df = pd.read_csv("jobs_50_weight.csv")

class Job:
    def __init__(self, id, processing_time, due_date, weight):
        self.id = id
        self.processing_time = processing_time
        self.due_date = due_date
        self.weight = weight

    def __repr__(self):
        return str(self.id)

# Convert DataFrame to list of Job objects
job_list = [Job(row['id'], row['processing_time'], row['due_date'], row['weight']) for _, row in job_df.iterrows()]

# Number of machines
NUM_MACHINES = 3

# Sort jobs by processing time (SPT rule)
spt_sorted_jobs = sorted(job_list, key=lambda job: job.processing_time)

# Initialize machine availability times
machine_available_time = [0] * NUM_MACHINES

# Schedule result
schedule_result = []

for job in spt_sorted_jobs:
    # Select machine with earliest available time
    selected_machine = machine_available_time.index(min(machine_available_time))
    start_time = machine_available_time[selected_machine]
    end_time = start_time + job.processing_time
    machine_available_time[selected_machine] = end_time

    schedule_result.append((job.id, selected_machine + 1, start_time, end_time, job.weight, max(0, end_time - job.due_date)))

# Sort results by start time for better readability
schedule_result.sort(key=lambda x: x[2])

# Create decision log
rows = []
remaining = [job[0] for job in schedule_result]
total = 0
for d, (jid, mid, st, ft, wt, tardy) in enumerate(schedule_result, 1):
    total += wt * tardy
    rows.append({
        "decision": d,
        "job_available": remaining.copy(),
        "job_chosen": jid,
        "WT": wt * tardy,
        "total_WT": total
    })
    remaining.remove(jid)

# Save to Excel file
output_path = "spt_decision_log.xlsx"
pd.DataFrame(rows).to_excel(output_path, index=False)
print(f"[Excel] SPT 의사결정 로그 저장 완료 → {output_path}")
