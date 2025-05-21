import pandas as pd
from ortools.linear_solver import pywraplp
import os
import time
import glob


def solve_milp_with_fixed_start(filepath):
    df = pd.read_csv(filepath)

    # 머신 이름 → 인덱스
    machine_map = {name: i for i, name in enumerate(sorted(df['machine'].unique()))}
    df['machine_id'] = df['machine'].map(machine_map)

    jobs = list(df['id'].unique())
    machines = list(df['machine_id'].unique())

    processing_time = {(j, m): 0 for j in jobs for m in machines}
    for _, row in df.iterrows():
        processing_time[(row['id'], row['machine_id'])] = row['processing_time']

    due_date = {row['id']: row['due_date'] for _, row in df.iterrows()}
    weight = {row['id']: row['weight'] for _, row in df.iterrows()}
    fixed_start = {row['id']: row['start'] for _, row in df.iterrows()}

    solver = pywraplp.Solver.CreateSolver('CBC')
    if not solver:
        raise RuntimeError("Solver not created")

    # 변수 선언
    tardiness = {j: solver.NumVar(0.0, solver.infinity(), f'tardiness_{j}') for j in jobs}
    x = {(j, m): solver.BoolVar(f'x_{j}_{m}') for j in jobs for m in machines}

    # 각 작업은 하나의 머신에만 할당
    for j in jobs:
        solver.Add(solver.Sum(x[j, m] for m in machines) == 1)

    # start[j]는 고정값으로 설정
    for j in jobs:
        job_proc = solver.Sum(x[j, m] * processing_time[j, m] for m in machines)
        solver.Add(tardiness[j] >= fixed_start[j] + job_proc - due_date[j])

    # 목적함수: WTT 최소화 (start 고정이므로 실제 지연 평가 목적)
    objective = solver.Sum(weight[j] * tardiness[j] for j in jobs)
    solver.Minimize(objective)

    # 실행
    start_time = time.time()
    status = solver.Solve()
    exec_time = time.time() - start_time

    if status == pywraplp.Solver.OPTIMAL or status == pywraplp.Solver.FEASIBLE:
        total_wtt = solver.Objective().Value()
        print(f'✅ {os.path.basename(filepath)} | 🕒 {exec_time:.2f}s | 📌 현실 스케줄 기준 WTT: {total_wtt:.2f}')
        return {'filename': os.path.basename(filepath), 'exec_time_sec': exec_time, 'total_wtt': total_wtt}
    else:
        print(f'❌ {os.path.basename(filepath)} | No feasible solution')
        return {'filename': os.path.basename(filepath), 'exec_time_sec': exec_time, 'total_wtt': None}


if __name__ == '__main__':
    import multiprocessing
    multiprocessing.freeze_support()

    # 파일 경로 지정
    all_files = sorted(glob.glob("large_generated/*.csv"))  # 또는 단일 테스트용 ["파일경로"]

    # 결과 수집
    results = []
    for f in all_files:
        result = solve_milp_with_fixed_start(f)
        results.append(result)

    df_result = pd.DataFrame(results)
    print("\n 현실 스케줄 기반 WTT 결과 요약:")
    print(df_result)

    avg_exec_time = df_result['exec_time_sec'].mean()
    avg_total_wtt = df_result['total_wtt'].mean()

    print(f"\n📈 평균 실행 시간: {avg_exec_time:.4f}초")
    print(f" 평균 총 WTT: {avg_total_wtt:.2f}")