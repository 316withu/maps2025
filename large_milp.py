import pandas as pd
from ortools.linear_solver import pywraplp
import os
import time
import glob


def solve_milp_from_scratch(filepath):
    df = pd.read_csv(filepath)

    # 머신 인덱스 매핑
    machine_map = {name: i for i, name in enumerate(sorted(df['machine'].unique()))}
    df['machine_id'] = df['machine'].map(machine_map)

    jobs = list(df['id'].unique())
    machines = list(df['machine_id'].unique())

    # 처리시간 딕셔너리
    processing_time = {(j, m): 1e6 for j in jobs for m in machines}
    for _, row in df.iterrows():
        processing_time[(row['id'], row['machine_id'])] = row['processing_time']

    due_date = {row['id']: row['due_date'] for _, row in df.iterrows()}
    weight = {row['id']: row['weight'] for _, row in df.iterrows()}

    max_M = sum(processing_time[j, m] for j in jobs for m in machines)

    solver = pywraplp.Solver.CreateSolver('CBC')
    if not solver:
        raise RuntimeError("Solver not created")

    solver.set_time_limit(3600000) # 1시간 제한
    solver.SetSolverSpecificParametersAsString("log_level=1\nmip_gap=0.01")

    # 변수 선언
    start = {j: solver.NumVar(0.0, solver.infinity(), f'start_{j}') for j in jobs}
    tardiness = {j: solver.NumVar(0.0, solver.infinity(), f'tardiness_{j}') for j in jobs}
    x = {(j, m): solver.BoolVar(f'x_{j}_{m}') for j in jobs for m in machines}

    # 각 작업은 하나의 머신에만 할당
    for j in jobs:
        solver.Add(solver.Sum(x[j, m] for m in machines) == 1)

    # tardiness 정의
    for j in jobs:
        proc = solver.Sum(x[j, m] * processing_time[j, m] for m in machines)
        solver.Add(tardiness[j] >= start[j] + proc - due_date[j])

    # 순서 제약 (같은 머신에 할당된 경우만)
    for i in range(len(jobs)):
        for j in range(i + 1, len(jobs)):
            j1, j2 = jobs[i], jobs[j]
            for m in machines:
                y = solver.BoolVar(f'order_{j1}_{j2}_{m}')
                pt1 = processing_time[j1, m]
                pt2 = processing_time[j2, m]

                solver.Add(start[j1] + pt1 <= start[j2] + max_M * (1 - y))
                solver.Add(start[j2] + pt2 <= start[j1] + max_M * y)
                solver.Add(y <= x[j1, m])
                solver.Add(y <= x[j2, m])

    # 목적함수: WTT 최소화
    solver.Minimize(solver.Sum(weight[j] * tardiness[j] for j in jobs))

    # 실행
    start_time = time.time()
    status = solver.Solve()
    exec_time = time.time() - start_time

    if status == pywraplp.Solver.OPTIMAL or status == pywraplp.Solver.FEASIBLE:
        total_wtt = solver.Objective().Value()
        print(f'✅ {os.path.basename(filepath)} | 🕒 {exec_time:.2f}s | 🎯 최소 WTT: {total_wtt:.2f}')
        return {'filename': os.path.basename(filepath), 'exec_time_sec': exec_time, 'total_wtt': total_wtt}
    else:
        print(f'❌ {os.path.basename(filepath)} | No feasible solution')
        return {'filename': os.path.basename(filepath), 'exec_time_sec': exec_time, 'total_wtt': None}


if __name__ == '__main__':
    import multiprocessing
    multiprocessing.freeze_support()

    all_files = sorted(glob.glob("large_generated/*.csv"))  # 또는 ["/mnt/data/hard_01_82j_5m.csv"]

    results = []
    for f in all_files:
        result = solve_milp_from_scratch(f)
        results.append(result)

    df_result = pd.DataFrame(results)
    print("\n최적화 기반 MILP 결과 요약:")
    print(df_result)

    avg_exec_time = df_result['exec_time_sec'].mean()
    avg_total_wtt = df_result['total_wtt'].mean()
    print(f"\n📈 평균 실행 시간: {avg_exec_time:.2f}초")
    print(f"🎯 평균 총 WTT: {avg_total_wtt:.2f}")
