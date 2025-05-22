import os
import pandas as pd
import time
import zipfile
from ortools.linear_solver import pywraplp

def solve_milp_with_order(filepath, save_dir="results_opt"):
    df = pd.read_csv(filepath, usecols=['id', 'machine', 'processing_time', 'due_date', 'weight'])

    machine_map = {name: i for i, name in enumerate(sorted(df['machine'].unique()))}
    rev_machine_map = {v: k for k, v in machine_map.items()}
    df['machine_id'] = df['machine'].map(machine_map)

    jobs = list(df['id'].unique())
    machines = list(df['machine_id'].unique())

    processing_time = {(row['id'], row['machine_id']): row['processing_time'] for _, row in df.iterrows()}
    due_date = {row['id']: row['due_date'] for _, row in df.iterrows()}
    weight = {row['id']: row['weight'] for _, row in df.iterrows()}

    solver = pywraplp.Solver.CreateSolver('CBC')
    solver.set_time_limit(300_000)

    big_M = sum(processing_time.values()) * 2

    start = {j: solver.NumVar(0.0, solver.infinity(), f'start_{j}') for j in jobs}
    tardiness = {j: solver.NumVar(0.0, solver.infinity(), f'tardiness_{j}') for j in jobs}
    x = {(j, m): solver.BoolVar(f'x_{j}_{m}') for (j, m) in processing_time.keys()}

    for j in jobs:
        solver.Add(solver.Sum(x[j, m] for m in machines if (j, m) in x) == 1)

    for j in jobs:
        proc_time = solver.Sum(x[j, m] * processing_time[(j, m)] for m in machines if (j, m) in x)
        solver.Add(tardiness[j] >= start[j] + proc_time - due_date[j])

    for i in range(len(jobs)):
        for j in range(i + 1, len(jobs)):
            j1, j2 = jobs[i], jobs[j]
            for m in machines:
                if (j1, m) in x and (j2, m) in x:
                    y = solver.BoolVar(f'order_{j1}_{j2}_{m}')
                    pt1 = processing_time[(j1, m)]
                    pt2 = processing_time[(j2, m)]
                    solver.Add(start[j1] + pt1 <= start[j2] + big_M * (1 - y))
                    solver.Add(start[j2] + pt2 <= start[j1] + big_M * y)
                    solver.Add(y <= x[j1, m])
                    solver.Add(y <= x[j2, m])

    solver.Minimize(solver.Sum(weight[j] * tardiness[j] for j in jobs))

    start_time = time.time()
    status = solver.Solve()
    exec_time = time.time() - start_time

    job_records = []

    if status in [pywraplp.Solver.OPTIMAL, pywraplp.Solver.FEASIBLE]:
        total_wtt = solver.Objective().Value()

        for j in jobs:
            for m in machines:
                if (j, m) in x and x[j, m].solution_value() > 0.5:
                    pt = processing_time[(j, m)]
                    s = start[j].solution_value()
                    job_records.append({
                        "job_id": j,
                        "machine": rev_machine_map[m],
                        "start": round(s, 2),
                        "finish": round(s + pt, 2),
                        "processing_time": pt,
                        "due_date": due_date[j],
                        "weight": weight[j],
                        "tardiness": round(max(0, s + pt - due_date[j]), 2)
                    })
                    break

        os.makedirs(save_dir, exist_ok=True)
        out_file = os.path.join(save_dir, f"result_{os.path.basename(filepath).replace('.csv', '.xlsx')}")
        pd.DataFrame(job_records).sort_values(by="start").to_excel(out_file, index=False)

        return {"filename": os.path.basename(filepath), "exec_time_sec": exec_time, "total_wtt": total_wtt}
    else:
        print(f"❌ {os.path.basename(filepath)}: No feasible solution.")
        return {"filename": os.path.basename(filepath), "exec_time_sec": exec_time, "total_wtt": None}


def zip_result_folder(folder_path, zip_name="milp_results.zip"):
    if not os.path.exists(folder_path):
        print(f"⚠️ 폴더 없음: {folder_path}")
        return
    with zipfile.ZipFile(zip_name, "w") as zipf:
        for file in os.listdir(folder_path):
            if file.endswith(".xlsx"):
                zipf.write(os.path.join(folder_path, file), arcname=file)
    print(f"📦 압축 완료: {zip_name}")


if __name__ == '__main__':
    import glob
    import multiprocessing
    multiprocessing.freeze_support()

    input_folder = "small_generated"
    save_dir = "results_opt"
    all_files = sorted(glob.glob(f"{input_folder}/*.csv"))

    results = []
    for f in all_files:
        results.append(solve_milp_with_order(f, save_dir))

    df_result = pd.DataFrame(results)

    print("\n MILP 실행 요약:")
    print(df_result)

    avg_exec_time = df_result['exec_time_sec'].mean()
    avg_total_wtt = df_result['total_wtt'].dropna().mean()
    print(f"\n📈 평균 실행 시간: {avg_exec_time:.4f}초")
    print(f"🎯 평균 총 WTT: {avg_total_wtt:.2f}")

    zip_result_folder(save_dir)