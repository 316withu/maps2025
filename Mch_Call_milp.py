# scheduler_machine_call.py  (v1 SDK + few‑shot 학습 예시 포함)
import os, re, argparse, heapq, pandas as pd, openai
from dataclasses import dataclass
from typing import List
from dotenv import load_dotenv
load_dotenv()

# ────────────────────────────────────────────────────────────────────
# 1. 기본 데이터 구조
@dataclass
class Job:
    id: int; p: int; d: int; w: int = 1

DEFAULT_CSV  = "jobs_50_weight.csv"
DEFAULT_LOGX = "milp_decision_log.xlsx"

def load_jobs(csv) -> List[Job]:
    df = pd.read_csv(csv)
    return [Job(int(r.id), int(r.processing_time), int(r.due_date),
                int(r.weight) if "weight" in df.columns else 1)  # 가중치 없으면 1
            for _, r in df.iterrows()]  # _ 는 인덱스 받고 안쓰겠다는 의미

# ────────────────────────────────────────────────────────────────────
# 2. 엑셀 → few‑shot 예시
def load_training_data(xlsx) -> list[dict]:
    df = pd.read_excel(xlsx)
    ex = []
    for _, r in df.iterrows():  #_ 는 인덱스 받고 안쓰겠다는 의미
        user = (
            f"Remaining jobs: {r.job_available}\n"
            f"Current total WT: {r.total_WT}\n"
            "Choose ONE Job ID that should be processed next."
        )
        ex += [{"role":"user","content":user},
               {"role":"assistant","content":str(r.job_chosen)}]
    return ex

# ────────────────────────────────────────────────────────────────────
# 3. 머신‑콜용 프롬프트
PROMPT = """
Machine {mname} is idle at time {now}.
Remaining jobs ({n}):

{lines}

Choose exactly ONE Job ID that {mname} should process next so that the
final weighted total tardiness will be as small as possible.

Return ONLY the Job ID.
""".strip()

def build_prompt(machine, remaining, now):
    lines = "\n".join(
        f"- Job {j.id}: w={j.w}, p={j.p}, d={j.d}" for j in remaining
    )
    return PROMPT.format(mname=machine, now=now, n=len(remaining), lines=lines)

# ────────────────────────────────────────────────────────────────────
# 4. LLM 호출 (v1 SDK)  ▸ 머신 이름 추가
client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def llm_choose_one(machine, remaining, now, few_shot) -> int:
    messages = few_shot + [{"role":"user", "content": build_prompt(machine, remaining, now)}]
    reply = client.chat.completions.create(
        model="gpt-4o",
        temperature=0,
        max_tokens=20,
        messages=messages
    ).choices[0].message.content
    m = re.search(r"\d+", reply)
    return int(m.group()) if m else remaining[0].id

# ────────────────────────────────────────────────────────────────────
# 5. 머신‑콜 디스패처  ▸ heap 사용
def dispatch_machine_call(jobs: List[Job], start: int, few_shot) -> List[tuple]:
    mch_names = ["A", "B", "C"]
    # 우선순위 큐에 (next_available, mch_index) 저장
    heap = [(start, i) for i in range(3)]
    heapq.heapify(heap) # 리스트를 힙 구조로 변환

    remaining = jobs.copy()
    assignment = []              # (machine_name, job_id) 순서 기록

    while remaining:
        now, mi = heapq.heappop(heap) # 가장 빠른 원소 선택후 제거
        mch = mch_names[mi]
        cid = llm_choose_one(mch, remaining, now, few_shot)
        job = next((j for j in remaining if j.id == cid), remaining[0])
        remaining.remove(job)
        assignment.append((mch, job.id))
        heapq.heappush(heap, (now + job.p, mi))   # 머신 가용 시각 갱신하는 함수
    return assignment

# ────────────────────────────────────────────────────────────────────
# 6. WTT 계산 (머신‑Job 쌍 입력)
def wtt_pairs(pairs, jobs, start):
    jobmap = {j.id: j for j in jobs}
    avail   = {"A":start, "B":start, "C":start}
    total   = 0
    for mch, jid in pairs:
        j = jobmap[jid]
        begin = avail[mch]
        finish = begin + j.p
        avail[mch] = finish
        total += j.w * max(0, finish - j.d)
    return total

# ────────────────────────────────────────────────────────────────────
# 7. main
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv",  default=DEFAULT_CSV)
    ap.add_argument("--log",  default=DEFAULT_LOGX)
    ap.add_argument("--start", type=int, default=0)
    args = ap.parse_args()

    jobs   = load_jobs(args.csv)
    shots  = load_training_data(args.log)   # few‑shot 예시
    pairs  = dispatch_machine_call(jobs, args.start, shots)
    score  = wtt_pairs(pairs, jobs, args.start)

    # after pairs = dispatch_machine_call(...)

    # ── 머신별 Job 호출 리스트 생성 ───────────────────────────────
    by_machine = {"A": [], "B": [], "C": []}
    for mch, jid in pairs:
        by_machine[mch].append(jid)

    print("training 예시 메시지 :", len(shots) // 2)
    print(
        "머신별 호출 순서     :\n"
        f'A: {by_machine["A"]}\n'
        f'B: {by_machine["B"]}\n'
        f'C: {by_machine["C"]}'
    )
    print("Weighted Total Tardiness:", score)

